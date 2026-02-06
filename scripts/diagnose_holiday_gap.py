#!/usr/bin/env python
"""
诊断节假日跳空对缠论分型的影响。

分析内容：
1. 检测所有跳空事件（session boundary 跳空 vs 日内跳空）
2. 分析跳空前后的分型/笔/中枢状态变化
3. 找出跳空后的失败交易模式
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qp.datafeed.normalizer import (
    normalize_1m_bars, PALM_OIL_SESSIONS, compute_window_end, get_session_key
)

def detect_gaps(df_1m: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """
    检测跳空事件。
    
    返回包含以下列的 DataFrame:
    - datetime: 跳空发生时间
    - gap_type: 'holiday' (停盘后) / 'session' (交易时段切换) / 'intraday' (日内)
    - gap_size: 跳空幅度（点）
    - gap_atr: 跳空幅度/ATR
    - prev_close: 前收盘
    - open_price: 开盘价
    - direction: 'up' / 'down'
    """
    gaps = []
    
    # 计算 ATR（用于归一化跳空幅度）
    df_1m = df_1m.copy()
    df_1m['prev_close'] = df_1m['close'].shift(1)
    df_1m['tr'] = np.maximum(
        df_1m['high'] - df_1m['low'],
        np.maximum(
            abs(df_1m['high'] - df_1m['prev_close']),
            abs(df_1m['low'] - df_1m['prev_close'])
        )
    )
    df_1m['atr'] = df_1m['tr'].rolling(atr_window * 5).mean()  # 用5m bar数量
    
    prev_row = None
    for idx, row in df_1m.iterrows():
        if prev_row is None:
            prev_row = row
            continue
            
        # 计算时间差
        time_diff = row['datetime'] - prev_row['datetime']
        gap = row['open'] - prev_row['close']
        gap_abs = abs(gap)
        
        # ATR 归一化
        atr = row['atr'] if pd.notna(row['atr']) and row['atr'] > 0 else 50  # 默认 50 点
        gap_atr = gap_abs / atr
        
        # 跳空检测阈值：1.5x ATR
        if gap_atr >= 1.5:
            # 判断跳空类型
            if time_diff > timedelta(hours=4):
                gap_type = 'holiday'  # 停盘超过4小时（含隔夜和节假日）
            elif time_diff > timedelta(hours=1):
                gap_type = 'session'  # 交易时段切换（如午休）
            else:
                gap_type = 'intraday'  # 日内跳空（罕见）
                
            gaps.append({
                'datetime': row['datetime'],
                'gap_type': gap_type,
                'gap_size': gap_abs,
                'gap_atr': gap_atr,
                'prev_close': prev_row['close'],
                'open_price': row['open'],
                'direction': 'up' if gap > 0 else 'down',
                'time_diff_hours': time_diff.total_seconds() / 3600,
            })
        
        prev_row = row
    
    return pd.DataFrame(gaps)


def analyze_gap_impact_on_fractal(df_1m: pd.DataFrame, gaps: pd.DataFrame) -> dict:
    """
    分析跳空对分型的影响。
    
    检查每个跳空后：
    1. 第一根K线是否被包含处理
    2. 包含处理方向是否合理
    3. 是否立即形成分型
    4. 分型的有效性（后续是否被破坏）
    """
    results = {
        'total_gaps': len(gaps),
        'holiday_gaps': len(gaps[gaps['gap_type'] == 'holiday']),
        'session_gaps': len(gaps[gaps['gap_type'] == 'session']),
        'intraday_gaps': len(gaps[gaps['gap_type'] == 'intraday']),
        'up_gaps': len(gaps[gaps['direction'] == 'up']),
        'down_gaps': len(gaps[gaps['direction'] == 'down']),
        'avg_gap_atr': gaps['gap_atr'].mean() if len(gaps) > 0 else 0,
        'max_gap_atr': gaps['gap_atr'].max() if len(gaps) > 0 else 0,
        'gap_details': [],
    }
    
    # 按时间排序
    df_1m = df_1m.sort_values('datetime').reset_index(drop=True)
    
    for _, gap in gaps.iterrows():
        gap_time = gap['datetime']
        
        # 找到跳空后的 K 线位置
        idx = df_1m[df_1m['datetime'] >= gap_time].index.min()
        if pd.isna(idx) or idx < 5:
            continue
            
        # 取跳空前后各5根K线
        pre_bars = df_1m.iloc[max(0, idx-5):idx]
        post_bars = df_1m.iloc[idx:min(len(df_1m), idx+10)]
        
        # 分析特征
        detail = {
            'datetime': gap_time,
            'gap_type': gap['gap_type'],
            'gap_atr': gap['gap_atr'],
            'direction': gap['direction'],
            # 跳空前的趋势
            'pre_trend': 'up' if pre_bars['close'].iloc[-1] > pre_bars['close'].iloc[0] else 'down',
            # 跳空后第一根K线的特征
            'first_bar_range': post_bars['high'].iloc[0] - post_bars['low'].iloc[0] if len(post_bars) > 0 else 0,
            # 跳空后10根K线的方向
            'post_10bar_trend': 'up' if len(post_bars) >= 10 and post_bars['close'].iloc[9] > post_bars['open'].iloc[0] else 'down' if len(post_bars) >= 10 else 'unknown',
            # 跳空方向与后续趋势是否一致
            'gap_continuation': None,
        }
        
        if detail['post_10bar_trend'] != 'unknown':
            detail['gap_continuation'] = detail['direction'] == detail['post_10bar_trend']
            
        results['gap_details'].append(detail)
    
    # 统计跳空延续率
    continuations = [d['gap_continuation'] for d in results['gap_details'] if d['gap_continuation'] is not None]
    results['continuation_rate'] = sum(continuations) / len(continuations) if continuations else 0
    
    return results


def simulate_fractal_processing(df_1m: pd.DataFrame) -> dict:
    """
    模拟分型处理，记录每根K线的处理过程。
    
    返回：
    - processed_bars: 包含处理后的K线列表
    - fractals: 识别出的分型列表
    - inclusion_events: 包含处理事件列表
    """
    # 合成5分钟K线
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
            }
            last_window_end = window_end
        else:
            window_bar['high'] = max(window_bar['high'], row['high'])
            window_bar['low'] = min(window_bar['low'], row['low'])
            window_bar['close'] = row['close']
    
    if window_bar:
        bars_5m.append(window_bar)
    
    # 包含处理
    processed = []
    inclusion_events = []
    inclusion_dir = 0  # 0=未定, 1=向上, -1=向下
    
    for bar in bars_5m:
        if not processed:
            processed.append(bar.copy())
            continue
            
        prev = processed[-1]
        
        # 检查是否包含
        contains = (prev['high'] >= bar['high'] and prev['low'] <= bar['low'])
        contained = (bar['high'] >= prev['high'] and bar['low'] <= prev['low'])
        
        if contains or contained:
            # 包含处理
            if inclusion_dir == 0:
                # 初次包含，用前一根K线方向
                if len(processed) >= 2:
                    inclusion_dir = 1 if processed[-1]['close'] > processed[-2]['close'] else -1
                else:
                    inclusion_dir = 1  # 默认向上
                    
            inclusion_events.append({
                'datetime': bar['datetime'],
                'type': 'contains' if contains else 'contained',
                'direction': inclusion_dir,
                'prev_high': prev['high'],
                'prev_low': prev['low'],
                'bar_high': bar['high'],
                'bar_low': bar['low'],
            })
            
            # 合并
            if inclusion_dir > 0:  # 向上
                processed[-1]['high'] = max(prev['high'], bar['high'])
                processed[-1]['low'] = max(prev['low'], bar['low'])
            else:  # 向下
                processed[-1]['high'] = min(prev['high'], bar['high'])
                processed[-1]['low'] = min(prev['low'], bar['low'])
            processed[-1]['close'] = bar['close']
            processed[-1]['datetime'] = bar['datetime']
        else:
            # 不包含，更新方向
            if bar['high'] > prev['high']:
                inclusion_dir = 1
            elif bar['low'] < prev['low']:
                inclusion_dir = -1
            processed.append(bar.copy())
    
    # 识别分型
    fractals = []
    for i in range(1, len(processed) - 1):
        prev, curr, next_ = processed[i-1], processed[i], processed[i+1]
        
        if curr['high'] > prev['high'] and curr['high'] > next_['high']:
            fractals.append({
                'datetime': curr['datetime'],
                'type': 'top',
                'price': curr['high'],
                'index': i,
            })
        elif curr['low'] < prev['low'] and curr['low'] < next_['low']:
            fractals.append({
                'datetime': curr['datetime'],
                'type': 'bottom',
                'price': curr['low'],
                'index': i,
            })
    
    return {
        'bars_5m_count': len(bars_5m),
        'processed_count': len(processed),
        'inclusion_events_count': len(inclusion_events),
        'fractals_count': len(fractals),
        'fractals': fractals[:20],  # 只返回前20个
        'inclusion_events': inclusion_events[:20],
    }


def main():
    """主函数：分析所有合约的跳空影响。"""
    data_dir = PROJECT_ROOT / "data" / "analyse" / "wind"
    
    all_results = {}
    
    # 分析每个合约
    for csv_file in sorted(data_dir.glob("*.csv")):
        contract = csv_file.stem.split("_")[0]
        print(f"\n{'='*60}")
        print(f"分析 {contract}")
        print(f"{'='*60}")
        
        # 读取数据
        df = pd.read_csv(csv_file)
        df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
        
        # 检测跳空
        gaps = detect_gaps(df)
        print(f"  检测到 {len(gaps)} 个跳空事件")
        
        if len(gaps) > 0:
            # 分析跳空影响
            impact = analyze_gap_impact_on_fractal(df, gaps)
            
            print(f"  - 节假日跳空: {impact['holiday_gaps']}")
            print(f"  - 时段跳空: {impact['session_gaps']}")
            print(f"  - 日内跳空: {impact['intraday_gaps']}")
            print(f"  - 向上跳空: {impact['up_gaps']}, 向下跳空: {impact['down_gaps']}")
            print(f"  - 平均跳空幅度: {impact['avg_gap_atr']:.2f}x ATR")
            print(f"  - 最大跳空幅度: {impact['max_gap_atr']:.2f}x ATR")
            print(f"  - 跳空延续率: {impact['continuation_rate']*100:.1f}%")
            
            # 显示最大的几个跳空
            gaps_sorted = gaps.sort_values('gap_atr', ascending=False)
            print(f"\n  Top 5 跳空事件:")
            for _, g in gaps_sorted.head(5).iterrows():
                print(f"    {g['datetime']}: {g['gap_type']} {g['direction']} "
                      f"{g['gap_size']:.0f}点 ({g['gap_atr']:.1f}x ATR)")
            
            all_results[contract] = impact
        
        # 模拟分型处理（检查包含处理问题）
        fractal_info = simulate_fractal_processing(df)
        print(f"\n  分型处理统计:")
        print(f"    - 5m K线: {fractal_info['bars_5m_count']}")
        print(f"    - 包含处理后: {fractal_info['processed_count']}")
        print(f"    - 包含事件: {fractal_info['inclusion_events_count']}")
        print(f"    - 分型数量: {fractal_info['fractals_count']}")
    
    # 汇总统计
    print(f"\n{'='*60}")
    print("汇总统计")
    print(f"{'='*60}")
    
    total_gaps = sum(r['total_gaps'] for r in all_results.values())
    total_holiday = sum(r['holiday_gaps'] for r in all_results.values())
    total_session = sum(r['session_gaps'] for r in all_results.values())
    
    print(f"总跳空事件: {total_gaps}")
    print(f"  - 节假日跳空: {total_holiday} ({total_holiday/total_gaps*100:.1f}%)")
    print(f"  - 时段跳空: {total_session} ({total_session/total_gaps*100:.1f}%)")
    
    # 跳空延续率分析
    all_continuations = []
    for r in all_results.values():
        for d in r['gap_details']:
            if d['gap_continuation'] is not None:
                all_continuations.append(d['gap_continuation'])
    
    if all_continuations:
        overall_continuation = sum(all_continuations) / len(all_continuations)
        print(f"\n跳空延续率（跳空方向与后续10bar趋势一致）: {overall_continuation*100:.1f}%")
        print(f"  → 如果 <50%，说明跳空后反转概率更高，应该反向交易")
        print(f"  → 如果 >50%，说明跳空延续概率更高，应该顺向交易")


if __name__ == "__main__":
    main()
