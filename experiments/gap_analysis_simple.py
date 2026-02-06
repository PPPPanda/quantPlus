#!/usr/bin/env python3
"""
跳空分析脚本（简化版）- 不依赖 vnpy

分析目标：
1. 识别所有跳空事件（session 间隔导致的 gap）
2. 分析跳空特征（幅度、时间、方向）
3. 为策略优化提供数据支持
"""

import os
import sys
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


def detect_gaps(df: pd.DataFrame, atr_mult: float = 3.0) -> pd.DataFrame:
    """
    检测跳空事件
    
    Args:
        df: 1分钟K线数据，需包含 datetime, open, high, low, close, volume
        atr_mult: 极端跳空阈值（×ATR）
    
    Returns:
        跳空事件 DataFrame
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 计算 ATR（使用滚动14*5=70根1分钟bar，约等于14根5分钟bar）
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
    df['time_gap'] = (df['datetime'] - df['prev_datetime']).dt.total_seconds() / 60
    
    # 识别 session 跳空（时间间隔 > 5 分钟）
    session_gap_mask = df['time_gap'] > 5
    
    # 计算跳空幅度
    df['gap'] = df['open'] - df['prev_close']
    df['gap_abs'] = df['gap'].abs()
    df['gap_atr'] = df['gap_abs'] / df['atr'].clip(lower=1)  # 避免除零
    
    # 筛选有意义的跳空（session 间隔 + gap > 0.3 ATR）
    gap_events = df[session_gap_mask & (df['gap_atr'] > 0.3)].copy()
    
    # 标记跳空类型
    gap_events['gap_type'] = np.where(gap_events['gap'] > 0, 'gap_up', 'gap_down')
    gap_events['is_extreme'] = gap_events['gap_atr'] > atr_mult
    
    # 判断跳空时间类型
    def classify_gap_time(row):
        dt = row['datetime']
        prev_dt = row['prev_datetime']
        gap_hours = row['time_gap'] / 60
        
        # 隔夜（收盘到下一交易日开盘）
        if gap_hours > 10:
            return 'overnight'
        # 午休（11:30-13:30）
        elif gap_hours > 1.5 and dt.hour == 13:
            return 'lunch_break'
        # 夜盘切换（15:00-21:00）
        elif gap_hours > 5 and dt.hour == 21:
            return 'day_to_night'
        # 节假日
        elif gap_hours > 24:
            return 'holiday'
        else:
            return 'other'
    
    gap_events['gap_time_type'] = gap_events.apply(classify_gap_time, axis=1)
    
    return gap_events[['datetime', 'prev_datetime', 'prev_close', 'open', 'close', 
                       'gap', 'gap_abs', 'gap_atr', 'atr', 'gap_type', 'is_extreme', 
                       'time_gap', 'gap_time_type']]


def analyze_contract_gaps(data_file: Path, contract: str) -> dict:
    """分析单个合约的跳空情况"""
    print(f"\n{'='*60}")
    print(f"Processing: {contract} - {data_file.name}")
    print(f"{'='*60}")
    
    # 加载数据
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"  Data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Total bars: {len(df)}")
    
    # 检测跳空
    gap_events = detect_gaps(df)
    
    if gap_events.empty:
        print(f"  No significant gaps detected")
        return {
            'contract': contract,
            'total_bars': len(df),
            'total_gaps': 0,
            'extreme_gaps': 0,
            'gaps': []
        }
    
    # 统计
    total_gaps = len(gap_events)
    extreme_gaps = gap_events['is_extreme'].sum()
    avg_gap = gap_events['gap_abs'].mean()
    avg_gap_atr = gap_events['gap_atr'].mean()
    max_gap = gap_events['gap_abs'].max()
    max_gap_atr = gap_events['gap_atr'].max()
    
    # 按类型统计
    by_type = gap_events.groupby('gap_type').agg({
        'gap': ['count', 'mean', 'sum'],
        'gap_atr': 'mean',
        'is_extreme': 'sum'
    }).round(2)
    
    by_time_type = gap_events.groupby('gap_time_type').agg({
        'gap': ['count', 'mean', 'sum'],
        'gap_atr': 'mean',
        'is_extreme': 'sum'
    }).round(2)
    
    print(f"\n  Total gaps: {total_gaps}")
    print(f"  Extreme gaps (>3 ATR): {extreme_gaps}")
    print(f"  Avg gap: {avg_gap:.0f} pts ({avg_gap_atr:.2f} ATR)")
    print(f"  Max gap: {max_gap:.0f} pts ({max_gap_atr:.2f} ATR)")
    
    print(f"\n  By direction:")
    print(f"    Gap up: {(gap_events['gap_type'] == 'gap_up').sum()}")
    print(f"    Gap down: {(gap_events['gap_type'] == 'gap_down').sum()}")
    
    print(f"\n  By time type:")
    for tt in ['overnight', 'lunch_break', 'day_to_night', 'holiday', 'other']:
        count = (gap_events['gap_time_type'] == tt).sum()
        if count > 0:
            subset = gap_events[gap_events['gap_time_type'] == tt]
            print(f"    {tt}: {count} gaps, avg={subset['gap'].mean():.0f} pts")
    
    # 详细极端跳空列表
    extreme_list = []
    if extreme_gaps > 0:
        print(f"\n  ⚠️ Extreme gaps (>3 ATR):")
        for _, row in gap_events[gap_events['is_extreme']].iterrows():
            direction = "↑" if row['gap'] > 0 else "↓"
            print(f"    {row['datetime']} {direction} {row['gap']:.0f} pts ({row['gap_atr']:.1f}×ATR) [{row['gap_time_type']}]")
            extreme_list.append({
                'datetime': str(row['datetime']),
                'prev_datetime': str(row['prev_datetime']),
                'gap': float(row['gap']),
                'gap_atr': float(row['gap_atr']),
                'gap_type': row['gap_type'],
                'gap_time_type': row['gap_time_type'],
            })
    
    # 返回结果
    gap_list = []
    for _, row in gap_events.iterrows():
        gap_list.append({
            'datetime': str(row['datetime']),
            'prev_datetime': str(row['prev_datetime']),
            'prev_close': float(row['prev_close']),
            'open': float(row['open']),
            'gap': float(row['gap']),
            'gap_atr': float(row['gap_atr']),
            'gap_type': row['gap_type'],
            'is_extreme': bool(row['is_extreme']),
            'gap_time_type': row['gap_time_type'],
        })
    
    return {
        'contract': contract,
        'total_bars': len(df),
        'total_gaps': total_gaps,
        'extreme_gaps': int(extreme_gaps),
        'avg_gap': float(avg_gap),
        'avg_gap_atr': float(avg_gap_atr),
        'max_gap': float(max_gap),
        'max_gap_atr': float(max_gap_atr),
        'gap_up_count': int((gap_events['gap_type'] == 'gap_up').sum()),
        'gap_down_count': int((gap_events['gap_type'] == 'gap_down').sum()),
        'gaps': gap_list,
        'extreme_list': extreme_list,
    }


def main():
    """主函数"""
    # 数据路径
    wind_dir = PROJECT_ROOT / "data" / "analyse" / "wind"
    xt_dir = PROJECT_ROOT / "data" / "analyse"
    
    # 全量 Wind 合约
    contracts = [
        ("p2201", wind_dir / "p2201_1min_202108-202112.csv"),
        ("p2205", wind_dir / "p2205_1min_202112-202204.csv"),
        ("p2209", wind_dir / "p2209_1min_202204-202208.csv"),
        ("p2301", wind_dir / "p2301_1min_202208-202212.csv"),
        ("p2305", wind_dir / "p2305_1min_202212-202304.csv"),
        ("p2309", wind_dir / "p2309_1min_202304-202308.csv"),
        ("p2401", wind_dir / "p2401_1min_202308-202312.csv"),
        ("p2405", wind_dir / "p2405_1min_202312-202404.csv"),
        ("p2409", wind_dir / "p2409_1min_202401-202408.csv"),
        ("p2501", wind_dir / "p2501_1min_202404-202412.csv"),
        ("p2505", wind_dir / "p2505_1min_202412-202504.csv"),
        ("p2509", wind_dir / "p2509_1min_202504-202508.csv"),
    ]
    
    # p2601 来自 XT（非 Wind）
    p2601_file = xt_dir / "p2601_1min_202507-202512.csv"
    if p2601_file.exists():
        contracts.append(("p2601", p2601_file))
    
    all_results = []
    all_extreme_gaps = []
    
    for contract, data_file in contracts:
        if not data_file.exists():
            print(f"Skipping {contract}: file not found at {data_file}")
            continue
        
        try:
            result = analyze_contract_gaps(data_file, contract)
            all_results.append(result)
            
            # 收集所有极端跳空
            for eg in result.get('extreme_list', []):
                eg['contract'] = contract
                all_extreme_gaps.append(eg)
                
        except Exception as e:
            print(f"Error processing {contract}: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总统计
    print("\n" + "="*80)
    print("SUMMARY: Gap Statistics Across All Contracts")
    print("="*80)
    
    total_gaps = sum(r['total_gaps'] for r in all_results)
    total_extreme = sum(r['extreme_gaps'] for r in all_results)
    
    print(f"\nTotal gaps across all contracts: {total_gaps}")
    print(f"Total extreme gaps (>3 ATR): {total_extreme}")
    
    # 按合约的极端跳空统计
    print("\n" + "-"*60)
    print("Extreme Gaps by Contract:")
    print("-"*60)
    
    for result in all_results:
        if result['extreme_gaps'] > 0:
            print(f"  {result['contract']}: {result['extreme_gaps']} extreme gaps")
    
    # 所有极端跳空详细列表
    print("\n" + "-"*60)
    print("All Extreme Gaps (sorted by ATR magnitude):")
    print("-"*60)
    
    all_extreme_gaps.sort(key=lambda x: abs(x['gap_atr']), reverse=True)
    for eg in all_extreme_gaps[:30]:  # Top 30
        direction = "📈" if eg['gap'] > 0 else "📉"
        print(f"  {direction} {eg['contract']} @ {eg['datetime']}: {eg['gap']:.0f} pts ({eg['gap_atr']:.1f}×ATR) [{eg['gap_time_type']}]")
    
    # 分析跳空对策略的潜在影响
    print("\n" + "="*80)
    print("ANALYSIS: Potential Impact on Strategy")
    print("="*80)
    
    # 统计跳空方向分布
    total_gap_up = sum(r['gap_up_count'] for r in all_results)
    total_gap_down = sum(r['gap_down_count'] for r in all_results)
    
    print(f"\nGap direction distribution:")
    print(f"  Gap up: {total_gap_up} ({total_gap_up/total_gaps*100:.1f}%)")
    print(f"  Gap down: {total_gap_down} ({total_gap_down/total_gaps*100:.1f}%)")
    
    # 分析极端跳空中多头/空头方向
    extreme_up = sum(1 for eg in all_extreme_gaps if eg['gap'] > 0)
    extreme_down = sum(1 for eg in all_extreme_gaps if eg['gap'] < 0)
    
    print(f"\nExtreme gap direction:")
    print(f"  Extreme gap up: {extreme_up}")
    print(f"  Extreme gap down: {extreme_down}")
    
    # 按时间类型分析极端跳空
    print(f"\nExtreme gaps by time type:")
    time_types = {}
    for eg in all_extreme_gaps:
        tt = eg['gap_time_type']
        if tt not in time_types:
            time_types[tt] = {'count': 0, 'total_gap': 0}
        time_types[tt]['count'] += 1
        time_types[tt]['total_gap'] += eg['gap']
    
    for tt, stats in sorted(time_types.items(), key=lambda x: -x[1]['count']):
        avg_gap = stats['total_gap'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {tt}: {stats['count']} gaps, avg={avg_gap:.0f} pts")
    
    # 策略影响分析
    print("\n" + "-"*60)
    print("Strategy Impact Analysis:")
    print("-"*60)
    
    print("""
对多头持仓的影响：
  - 极端向上跳空（gap_up）：对多头持仓有利，浮盈瞬间扩大
  - 极端向下跳空（gap_down）：对多头持仓不利，可能直接击穿止损

对策略信号的影响：
  - 跳空破坏缠论连续性假设
  - 分型/笔/中枢结构可能失真
  - 背驰判断的 MACD 面积计算被扭曲

S26 保护机制的作用：
  - 极端跳空后暂停 3 根 5m bar 信号
  - 等待结构重新稳定后再入场
  - 避免在混乱期盲目追涨杀跌
""")
    
    # 保存结果到 JSON
    output_file = PROJECT_ROOT / "experiments" / "gap_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_gaps': total_gaps,
                'total_extreme_gaps': total_extreme,
                'gap_up_count': total_gap_up,
                'gap_down_count': total_gap_down,
            },
            'by_contract': all_results,
            'all_extreme_gaps': all_extreme_gaps,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()
