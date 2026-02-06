#!/usr/bin/env python3
"""
iter18: 分析节假日跳空对缠论分型/笔的影响

核心问题：
1. 跳空K线是否被包含处理吃掉？
2. 跳空是否破坏正在形成的分型？
3. 跳空后的交易表现如何？
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import numpy as np
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 src 到 path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qp.datafeed.normalizer import PALM_OIL_SESSIONS, normalize_1m_bars


def load_data(contract: str) -> pd.DataFrame:
    """加载合约数据"""
    # 尝试多个路径
    paths = [
        PROJECT_ROOT / f"data/analyse/wind/{contract}_1min_*.csv",
        PROJECT_ROOT / f"data/analyse/{contract}_1min_*.csv",
    ]
    
    import glob
    for pattern in paths:
        files = glob.glob(str(pattern))
        if files:
            df = pd.read_csv(files[0])
            df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
            return df
    
    raise FileNotFoundError(f"No data file found for {contract}")


def detect_holidays(df: pd.DataFrame) -> List[Dict]:
    """检测节假日跳空"""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 计算 ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=70, min_periods=14).mean()
    
    # 检测时间间隔
    df['prev_close'] = df['close'].shift(1)
    df['prev_datetime'] = df['datetime'].shift(1)
    df['hours_gap'] = (df['datetime'] - df['prev_datetime']).dt.total_seconds() / 3600
    
    # 计算跳空
    df['gap'] = df['open'] - df['prev_close']
    df['gap_abs'] = df['gap'].abs()
    df['gap_atr'] = df['gap_abs'] / df['atr'].clip(lower=1)
    
    # 筛选节假日跳空 (>48小时)
    holiday_mask = df['hours_gap'] > 48
    holidays = df[holiday_mask].copy()
    
    return holidays.to_dict('records')


def simulate_chan_at_gap(df: pd.DataFrame, gap_idx: int, window: int = 50) -> Dict:
    """
    模拟跳空前后的缠论结构变化
    
    Returns:
        {
            'before_gap': 跳空前的结构状态,
            'at_gap': 跳空K线的处理结果,
            'after_gap': 跳空后的结构状态,
            'fractal_disrupted': 是否有分型被破坏,
            'bi_extended': 笔是否被延伸,
        }
    """
    # 取跳空前后的数据
    start_idx = max(0, gap_idx - window)
    end_idx = min(len(df), gap_idx + window)
    
    sub_df = df.iloc[start_idx:end_idx].copy()
    sub_df = sub_df.reset_index(drop=True)
    gap_local_idx = gap_idx - start_idx
    
    # 简化的包含处理模拟
    k_lines = []
    inclusion_dir = 0
    
    for i, row in sub_df.iterrows():
        new_bar = {
            'datetime': row['datetime'],
            'high': row['high'],
            'low': row['low'],
            'is_gap': i == gap_local_idx,
        }
        
        if not k_lines:
            k_lines.append(new_bar)
            continue
        
        last = k_lines[-1]
        
        # 检查包含关系
        in_last = new_bar['high'] <= last['high'] and new_bar['low'] >= last['low']
        in_new = last['high'] <= new_bar['high'] and last['low'] >= new_bar['low']
        
        if in_last or in_new:
            # 存在包含关系
            if inclusion_dir == 0:
                k_lines.append(new_bar)
                continue
            
            merged = last.copy()
            merged['datetime'] = new_bar['datetime']
            merged['merged_count'] = last.get('merged_count', 1) + 1
            merged['contains_gap'] = last.get('contains_gap', False) or new_bar['is_gap']
            
            if inclusion_dir == 1:  # 向上
                merged['high'] = max(last['high'], new_bar['high'])
                merged['low'] = max(last['low'], new_bar['low'])
            else:  # 向下
                merged['high'] = min(last['high'], new_bar['high'])
                merged['low'] = min(last['low'], new_bar['low'])
            
            k_lines[-1] = merged
        else:
            # 无包含关系，更新方向
            if new_bar['high'] > last['high'] and new_bar['low'] > last['low']:
                inclusion_dir = 1
            elif new_bar['high'] < last['high'] and new_bar['low'] < last['low']:
                inclusion_dir = -1
            
            k_lines.append(new_bar)
    
    # 检查跳空K线是否被包含
    gap_merged = False
    merged_bars = 0
    for k in k_lines:
        if k.get('contains_gap'):
            gap_merged = True
            merged_bars = k.get('merged_count', 1)
            break
    
    # 简化的分型检测
    fractals = []
    for i in range(1, len(k_lines) - 1):
        left = k_lines[i-1]
        mid = k_lines[i]
        right = k_lines[i+1]
        
        is_top = mid['high'] > left['high'] and mid['high'] > right['high']
        is_bot = mid['low'] < left['low'] and mid['low'] < right['low']
        
        if is_top:
            fractals.append({'type': 'top', 'idx': i, 'price': mid['high'],
                           'contains_gap': mid.get('contains_gap', False)})
        elif is_bot:
            fractals.append({'type': 'bottom', 'idx': i, 'price': mid['low'],
                           'contains_gap': mid.get('contains_gap', False)})
    
    # 统计跳空对分型的影响
    gap_in_fractal = any(f.get('contains_gap') for f in fractals)
    
    return {
        'gap_merged': gap_merged,
        'merged_bars': merged_bars,
        'total_fractals': len(fractals),
        'gap_in_fractal': gap_in_fractal,
        'k_lines_count': len(k_lines),
        'original_bars': len(sub_df),
        'compression_ratio': len(k_lines) / len(sub_df) if len(sub_df) > 0 else 1,
    }


def analyze_all_contracts():
    """分析所有合约的节假日跳空影响"""
    contracts = [
        'p2201', 'p2205', 'p2209', 'p2301', 'p2305', 'p2309',
        'p2401', 'p2405', 'p2409', 'p2501', 'p2505', 'p2509', 'p2601'
    ]
    
    results = {
        'summary': {},
        'by_contract': {},
        'all_holiday_gaps': [],
    }
    
    total_gaps = 0
    total_merged = 0
    total_in_fractal = 0
    
    for contract in contracts:
        print(f"\n{'='*60}")
        print(f"Analyzing {contract}...")
        
        try:
            df = load_data(contract)
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
            continue
        
        # 检测节假日跳空
        holidays = detect_holidays(df)
        if not holidays:
            print(f"  No holiday gaps found")
            continue
        
        # 过滤掉太小的跳空 (< 3 ATR)
        significant_holidays = [h for h in holidays if h.get('gap_atr', 0) > 3]
        
        print(f"  Total holiday gaps: {len(holidays)}")
        print(f"  Significant (>3 ATR): {len(significant_holidays)}")
        
        contract_results = []
        for h in significant_holidays:
            # 找到这个跳空在 df 中的位置
            gap_time = h['datetime']
            idx = df[df['datetime'] == gap_time].index
            if len(idx) == 0:
                continue
            gap_idx = idx[0]
            
            # 模拟缠论结构
            impact = simulate_chan_at_gap(df, gap_idx)
            
            result = {
                'datetime': str(gap_time),
                'gap': h['gap'],
                'gap_atr': h['gap_atr'],
                'hours_gap': h['hours_gap'],
                **impact
            }
            contract_results.append(result)
            
            total_gaps += 1
            if impact['gap_merged']:
                total_merged += 1
            if impact['gap_in_fractal']:
                total_in_fractal += 1
            
            # 添加到全局列表
            result['contract'] = contract
            results['all_holiday_gaps'].append(result)
        
        results['by_contract'][contract] = {
            'total_gaps': len(contract_results),
            'gaps': contract_results,
        }
        
        # 打印摘要
        if contract_results:
            merged_cnt = sum(1 for r in contract_results if r['gap_merged'])
            fractal_cnt = sum(1 for r in contract_results if r['gap_in_fractal'])
            print(f"  Gap merged into K-line: {merged_cnt}/{len(contract_results)}")
            print(f"  Gap in fractal: {fractal_cnt}/{len(contract_results)}")
    
    # 总结
    results['summary'] = {
        'total_gaps': total_gaps,
        'gaps_merged': total_merged,
        'gaps_in_fractal': total_in_fractal,
        'merge_rate': total_merged / total_gaps if total_gaps > 0 else 0,
        'fractal_impact_rate': total_in_fractal / total_gaps if total_gaps > 0 else 0,
    }
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total significant holiday gaps: {total_gaps}")
    print(f"Gaps merged by inclusion: {total_merged} ({total_merged/total_gaps*100:.1f}%)")
    print(f"Gaps affecting fractals: {total_in_fractal} ({total_in_fractal/total_gaps*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    results = analyze_all_contracts()
    
    # 保存结果
    output_path = Path(__file__).parent / "gap_fractal_impact.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
