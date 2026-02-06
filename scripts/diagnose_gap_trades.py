#!/usr/bin/env python
"""
诊断跳空后的交易失败模式。

分析内容：
1. 跳空后 N 根 bar 内的交易胜率
2. 跳空对中枢的影响（是否跨越中枢区间）
3. 跳空后分型失效的比例
4. 不同跳空类型（向上/向下）的表现差异
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qp.datafeed.normalizer import (
    normalize_1m_bars, PALM_OIL_SESSIONS, compute_window_end, get_session_key
)


def build_5m_bars(df_1m: pd.DataFrame) -> pd.DataFrame:
    """将1分钟数据合成5分钟数据。"""
    bars_5m = []
    window_bar = None
    last_window_end = None
    
    for _, row in df_1m.iterrows():
        dt = row['datetime']
        window_end = compute_window_end(dt, PALM_OIL_SESSIONS, 5)
        if window_end is None:
            continue
            
        if window_bar is not None and last_window_end != window_end:
            bars_5m.append(window_bar.copy())
            window_bar = None
            
        if window_bar is None:
            window_bar = {
                'datetime': window_end,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'] if 'volume' in row else 0,
            }
            last_window_end = window_end
        else:
            window_bar['high'] = max(window_bar['high'], row['high'])
            window_bar['low'] = min(window_bar['low'], row['low'])
            window_bar['close'] = row['close']
            if 'volume' in row:
                window_bar['volume'] += row['volume']
    
    if window_bar:
        bars_5m.append(window_bar)
    
    return pd.DataFrame(bars_5m)


def detect_gaps_5m(df_5m: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """检测5分钟K线上的跳空事件。"""
    gaps = []
    
    # 计算 ATR
    df_5m = df_5m.copy()
    df_5m['prev_close'] = df_5m['close'].shift(1)
    df_5m['tr'] = np.maximum(
        df_5m['high'] - df_5m['low'],
        np.maximum(
            abs(df_5m['high'] - df_5m['prev_close']),
            abs(df_5m['low'] - df_5m['prev_close'])
        )
    )
    df_5m['atr'] = df_5m['tr'].rolling(atr_window).mean()
    
    prev_row = None
    for idx, row in df_5m.iterrows():
        if prev_row is None:
            prev_row = row
            continue
            
        time_diff = row['datetime'] - prev_row['datetime']
        gap = row['open'] - prev_row['close']
        gap_abs = abs(gap)
        
        atr = row['atr'] if pd.notna(row['atr']) and row['atr'] > 0 else 30
        gap_atr = gap_abs / atr
        
        if gap_atr >= 1.5:
            if time_diff > timedelta(hours=4):
                gap_type = 'holiday'
            elif time_diff > timedelta(hours=1):
                gap_type = 'session'
            else:
                gap_type = 'intraday'
                
            gaps.append({
                'idx': idx,
                'datetime': row['datetime'],
                'gap_type': gap_type,
                'gap_size': gap_abs,
                'gap_atr': gap_atr,
                'prev_close': prev_row['close'],
                'open_price': row['open'],
                'direction': 'up' if gap > 0 else 'down',
                'atr': atr,
            })
        
        prev_row = row
    
    return pd.DataFrame(gaps)


def simulate_chan_structure(df_5m: pd.DataFrame) -> dict:
    """
    模拟缠论结构的构建（简化版）。
    
    返回：
    - fractals: 分型列表
    - bi_points: 笔端点列表
    - pivots: 中枢列表
    """
    # 1. 包含处理
    processed = []
    inclusion_dir = 0
    
    for _, row in df_5m.iterrows():
        bar = {
            'datetime': row['datetime'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
        }
        
        if not processed:
            processed.append(bar)
            continue
            
        prev = processed[-1]
        
        contains = (prev['high'] >= bar['high'] and prev['low'] <= bar['low'])
        contained = (bar['high'] >= prev['high'] and bar['low'] <= prev['low'])
        
        if contains or contained:
            if inclusion_dir == 0:
                if len(processed) >= 2:
                    inclusion_dir = 1 if processed[-1]['close'] > processed[-2]['close'] else -1
                else:
                    inclusion_dir = 1
                    
            if inclusion_dir > 0:
                processed[-1]['high'] = max(prev['high'], bar['high'])
                processed[-1]['low'] = max(prev['low'], bar['low'])
            else:
                processed[-1]['high'] = min(prev['high'], bar['high'])
                processed[-1]['low'] = min(prev['low'], bar['low'])
            processed[-1]['close'] = bar['close']
            processed[-1]['datetime'] = bar['datetime']
        else:
            if bar['high'] > prev['high']:
                inclusion_dir = 1
            elif bar['low'] < prev['low']:
                inclusion_dir = -1
            processed.append(bar)
    
    # 2. 识别分型
    fractals = []
    for i in range(1, len(processed) - 1):
        prev, curr, next_ = processed[i-1], processed[i], processed[i+1]
        
        if curr['high'] > prev['high'] and curr['high'] > next_['high']:
            fractals.append({
                'datetime': curr['datetime'],
                'type': 'top',
                'price': curr['high'],
                'processed_idx': i,
            })
        elif curr['low'] < prev['low'] and curr['low'] < next_['low']:
            fractals.append({
                'datetime': curr['datetime'],
                'type': 'bottom',
                'price': curr['low'],
                'processed_idx': i,
            })
    
    # 3. 构建严格笔（简化：min_bi_gap=4）
    min_bi_gap = 4
    bi_points = []
    last_fractal = None
    
    for f in fractals:
        if last_fractal is None:
            bi_points.append(f)
            last_fractal = f
            continue
            
        # 同向分型跳过
        if f['type'] == last_fractal['type']:
            # 更新：顶取更高，底取更低
            if f['type'] == 'top' and f['price'] > last_fractal['price']:
                bi_points[-1] = f
                last_fractal = f
            elif f['type'] == 'bottom' and f['price'] < last_fractal['price']:
                bi_points[-1] = f
                last_fractal = f
            continue
        
        # 检查 gap
        gap = f['processed_idx'] - last_fractal['processed_idx']
        if gap >= min_bi_gap:
            bi_points.append(f)
            last_fractal = f
    
    # 4. 识别中枢（3笔重叠）
    pivots = []
    if len(bi_points) >= 4:
        for i in range(len(bi_points) - 3):
            # 取3笔的端点
            p1, p2, p3, p4 = bi_points[i:i+4]
            
            # 计算重叠区间
            highs = [p1['price'] if p1['type'] == 'top' else p2['price'],
                     p2['price'] if p2['type'] == 'top' else p3['price'],
                     p3['price'] if p3['type'] == 'top' else p4['price']]
            lows = [p1['price'] if p1['type'] == 'bottom' else p2['price'],
                    p2['price'] if p2['type'] == 'bottom' else p3['price'],
                    p3['price'] if p3['type'] == 'bottom' else p4['price']]
            
            zg = min(highs)  # 中枢高点
            zd = max(lows)   # 中枢低点
            
            if zg > zd:  # 有效中枢
                pivots.append({
                    'datetime': p1['datetime'],
                    'zg': zg,
                    'zd': zd,
                    'range': zg - zd,
                    'start_idx': i,
                })
    
    return {
        'processed_count': len(processed),
        'fractals': fractals,
        'bi_points': bi_points,
        'pivots': pivots,
    }


def analyze_gap_impact_on_structure(df_5m: pd.DataFrame, gaps: pd.DataFrame, structure: dict) -> list:
    """分析跳空对结构的具体影响。"""
    impacts = []
    
    pivots = structure['pivots']
    bi_points = structure['bi_points']
    
    for _, gap in gaps.iterrows():
        gap_time = gap['datetime']
        gap_idx = gap['idx']
        
        impact = {
            'datetime': gap_time,
            'gap_type': gap['gap_type'],
            'gap_atr': gap['gap_atr'],
            'direction': gap['direction'],
            
            # 跳空是否跨越中枢
            'crosses_pivot': False,
            'pivot_crossed': None,
            
            # 跳空后的结构变化
            'fractal_formed_within_3bars': False,
            'bi_formed_within_5bars': False,
            
            # 跳空后10根bar的表现
            'post_10bar_return': 0,
            'post_10bar_max_adverse': 0,
        }
        
        # 检查是否跨越中枢
        for pivot in pivots:
            if pd.Timestamp(pivot['datetime']) < gap_time:
                zg, zd = pivot['zg'], pivot['zd']
                prev_close = gap['prev_close']
                open_price = gap['open_price']
                
                # 检查是否从中枢一侧跳到另一侧
                prev_in_pivot = zd <= prev_close <= zg
                open_in_pivot = zd <= open_price <= zg
                
                if prev_in_pivot != open_in_pivot or \
                   (prev_close < zd and open_price > zg) or \
                   (prev_close > zg and open_price < zd):
                    impact['crosses_pivot'] = True
                    impact['pivot_crossed'] = {
                        'zg': zg, 'zd': zd,
                        'prev_close': prev_close,
                        'open_price': open_price,
                    }
                    break
        
        # 检查跳空后的表现
        post_bars = df_5m.iloc[gap_idx:gap_idx+10]
        if len(post_bars) >= 10:
            open_price = post_bars['open'].iloc[0]
            close_price = post_bars['close'].iloc[9]
            impact['post_10bar_return'] = (close_price - open_price) / open_price * 100
            
            if gap['direction'] == 'up':
                # 向上跳空后的最大回撤
                min_low = post_bars['low'].min()
                impact['post_10bar_max_adverse'] = (min_low - open_price) / open_price * 100
            else:
                # 向下跳空后的最大反弹
                max_high = post_bars['high'].max()
                impact['post_10bar_max_adverse'] = (max_high - open_price) / open_price * 100
        
        impacts.append(impact)
    
    return impacts


def main():
    """主函数：分析跳空对缠论结构的影响。"""
    data_dir = PROJECT_ROOT / "data" / "analyse" / "wind"
    output_dir = PROJECT_ROOT / "experiments" / "iter21_holiday_gap"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_impacts = []
    contract_summary = {}
    
    for csv_file in sorted(data_dir.glob("*.csv")):
        contract = csv_file.stem.split("_")[0]
        print(f"\n分析 {contract}...")
        
        # 读取数据
        df_1m = pd.read_csv(csv_file)
        df_1m = normalize_1m_bars(df_1m, PALM_OIL_SESSIONS)
        
        # 合成5分钟
        df_5m = build_5m_bars(df_1m)
        df_5m = df_5m.reset_index(drop=True)
        
        # 检测跳空
        gaps = detect_gaps_5m(df_5m)
        
        if len(gaps) == 0:
            print(f"  无显著跳空")
            continue
            
        # 构建缠论结构
        structure = simulate_chan_structure(df_5m)
        
        # 分析跳空影响
        impacts = analyze_gap_impact_on_structure(df_5m, gaps, structure)
        
        # 统计
        crosses_pivot_count = sum(1 for i in impacts if i['crosses_pivot'])
        
        # 跳空后表现统计
        up_gaps = [i for i in impacts if i['direction'] == 'up']
        down_gaps = [i for i in impacts if i['direction'] == 'down']
        
        up_continuation = sum(1 for i in up_gaps if i['post_10bar_return'] > 0) / len(up_gaps) if up_gaps else 0
        down_continuation = sum(1 for i in down_gaps if i['post_10bar_return'] < 0) / len(down_gaps) if down_gaps else 0
        
        print(f"  跳空数: {len(gaps)}, 跨越中枢: {crosses_pivot_count}")
        print(f"  向上跳空延续率: {up_continuation*100:.1f}%")
        print(f"  向下跳空延续率: {down_continuation*100:.1f}%")
        
        contract_summary[contract] = {
            'total_gaps': len(gaps),
            'crosses_pivot': crosses_pivot_count,
            'up_gaps': len(up_gaps),
            'down_gaps': len(down_gaps),
            'up_continuation_rate': up_continuation,
            'down_continuation_rate': down_continuation,
            'pivots_count': len(structure['pivots']),
            'bi_points_count': len(structure['bi_points']),
        }
        
        for i in impacts:
            i['contract'] = contract
        all_impacts.extend(impacts)
    
    # 汇总分析
    print("\n" + "="*60)
    print("跳空对中枢的影响汇总")
    print("="*60)
    
    total_gaps = len(all_impacts)
    crosses_pivot = sum(1 for i in all_impacts if i['crosses_pivot'])
    
    print(f"总跳空: {total_gaps}")
    print(f"跨越中枢: {crosses_pivot} ({crosses_pivot/total_gaps*100:.1f}%)")
    
    # 跨越中枢的跳空后表现
    crossing_gaps = [i for i in all_impacts if i['crosses_pivot']]
    non_crossing_gaps = [i for i in all_impacts if not i['crosses_pivot']]
    
    if crossing_gaps:
        crossing_returns = [i['post_10bar_return'] for i in crossing_gaps if i['post_10bar_return'] != 0]
        print(f"\n跨越中枢的跳空后10bar平均收益: {np.mean(crossing_returns):.3f}%")
        print(f"跨越中枢的跳空后10bar最大回撤: {np.mean([i['post_10bar_max_adverse'] for i in crossing_gaps]):.3f}%")
    
    if non_crossing_gaps:
        non_crossing_returns = [i['post_10bar_return'] for i in non_crossing_gaps if i['post_10bar_return'] != 0]
        print(f"\n未跨越中枢的跳空后10bar平均收益: {np.mean(non_crossing_returns):.3f}%")
    
    # 保存结果
    with open(output_dir / "gap_structure_analysis.json", "w") as f:
        json.dump({
            'summary': {
                'total_gaps': total_gaps,
                'crosses_pivot': crosses_pivot,
                'crosses_pivot_pct': crosses_pivot / total_gaps if total_gaps > 0 else 0,
            },
            'contract_summary': contract_summary,
        }, f, indent=2, default=str)
    
    print(f"\n结果已保存到 {output_dir / 'gap_structure_analysis.json'}")


if __name__ == "__main__":
    main()
