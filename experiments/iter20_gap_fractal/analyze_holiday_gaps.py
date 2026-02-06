#!/usr/bin/env python3
"""
分析节假日跳空对缠论分型的影响
iter20: 新视角研究（不使用S26/S27）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# 中国法定节假日（2021-2025）
CHINESE_HOLIDAYS = {
    # 2022年节假日
    '2022-01-01', '2022-01-02', '2022-01-03',  # 元旦
    '2022-01-31', '2022-02-01', '2022-02-02', '2022-02-03', '2022-02-04', '2022-02-05', '2022-02-06',  # 春节
    '2022-04-03', '2022-04-04', '2022-04-05',  # 清明
    '2022-04-30', '2022-05-01', '2022-05-02', '2022-05-03', '2022-05-04',  # 劳动节
    '2022-06-03', '2022-06-04', '2022-06-05',  # 端午
    '2022-09-10', '2022-09-11', '2022-09-12',  # 中秋
    '2022-10-01', '2022-10-02', '2022-10-03', '2022-10-04', '2022-10-05', '2022-10-06', '2022-10-07',  # 国庆
    # 2023年
    '2023-01-01', '2023-01-02',
    '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26', '2023-01-27',
    '2023-04-05',
    '2023-04-29', '2023-04-30', '2023-05-01', '2023-05-02', '2023-05-03',
    '2023-06-22', '2023-06-23', '2023-06-24',
    '2023-09-29', '2023-09-30', '2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05', '2023-10-06',
    # 2024年
    '2024-01-01',
    '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16', '2024-02-17',
    '2024-04-04', '2024-04-05', '2024-04-06',
    '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05',
    '2024-06-08', '2024-06-09', '2024-06-10',
    '2024-09-15', '2024-09-16', '2024-09-17',
    '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06', '2024-10-07',
    # 2025年
    '2025-01-01',
    '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04',
    '2025-04-04', '2025-04-05', '2025-04-06',
    '2025-05-01', '2025-05-02', '2025-05-03', '2025-05-04', '2025-05-05',
    '2025-06-01', '2025-06-02',
    '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05', '2025-10-06', '2025-10-07',
}

def is_holiday(date_str):
    """判断是否为节假日"""
    return date_str in CHINESE_HOLIDAYS

def is_weekend(dt):
    """判断是否为周末"""
    return dt.weekday() >= 5

def get_holiday_gap_sessions(df):
    """识别节假日后的跳空交易日"""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d')
    
    # 按日聚合
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'datetime': 'first'
    }).reset_index()
    daily['date_str'] = daily['date'].astype(str)
    
    gap_sessions = []
    for i in range(1, len(daily)):
        curr = daily.iloc[i]
        prev = daily.iloc[i-1]
        
        # 计算交易日间隔
        days_gap = (curr['date'] - prev['date']).days
        
        # 检查是否跨越节假日
        crossed_holiday = False
        crossed_weekend = False
        for d in range(1, days_gap):
            check_date = prev['date'] + timedelta(days=d)
            if is_holiday(str(check_date)):
                crossed_holiday = True
            if check_date.weekday() >= 5:
                crossed_weekend = True
        
        # 只关注节假日跳空（不是普通周末）
        if crossed_holiday or days_gap >= 3:
            gap = curr['open'] - prev['close']
            gap_pct = gap / prev['close'] * 100
            
            gap_sessions.append({
                'date': str(curr['date']),
                'prev_date': str(prev['date']),
                'days_gap': days_gap,
                'crossed_holiday': crossed_holiday,
                'prev_close': prev['close'],
                'open': curr['open'],
                'gap': gap,
                'gap_pct': gap_pct,
                'gap_direction': 'up' if gap > 0 else 'down',
            })
    
    return gap_sessions

def analyze_fractal_around_gap(df, gap_date, window=20):
    """分析跳空前后的分型变化"""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d')
    
    gap_df = df[df['date_str'] == gap_date]
    if gap_df.empty:
        return None
    
    gap_start_idx = gap_df.index[0]
    
    # 取跳空前后的数据
    start_idx = max(0, gap_start_idx - window * 5)  # 约window个5m bar
    end_idx = min(len(df), gap_start_idx + window * 5)
    
    analysis_df = df.iloc[start_idx:end_idx].copy()
    
    # 简化的5m K线合成
    analysis_df['5m_group'] = analysis_df['datetime'].dt.floor('5min')
    bars_5m = analysis_df.groupby('5m_group').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }).reset_index()
    bars_5m.columns = ['datetime', 'open', 'high', 'low', 'close']
    
    # 简化的分型识别（无包含处理）
    fractals = []
    for i in range(1, len(bars_5m) - 1):
        left = bars_5m.iloc[i-1]
        mid = bars_5m.iloc[i]
        right = bars_5m.iloc[i+1]
        
        if mid['high'] > left['high'] and mid['high'] > right['high']:
            fractals.append({
                'datetime': mid['datetime'],
                'type': 'top',
                'price': mid['high']
            })
        elif mid['low'] < left['low'] and mid['low'] < right['low']:
            fractals.append({
                'datetime': mid['datetime'],
                'type': 'bottom',
                'price': mid['low']
            })
    
    # 标记跳空点
    gap_datetime = pd.Timestamp(gap_date)
    
    # 分析跳空前后的分型分布
    before_gap = [f for f in fractals if f['datetime'].date() < gap_datetime.date()]
    after_gap = [f for f in fractals if f['datetime'].date() >= gap_datetime.date()]
    
    return {
        'gap_date': gap_date,
        'total_fractals': len(fractals),
        'fractals_before': len(before_gap),
        'fractals_after': len(after_gap),
        'bars_5m_count': len(bars_5m),
        'first_fractal_after': after_gap[0] if after_gap else None,
    }

def main():
    data_dir = Path('/mnt/e/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse')
    
    # 分析所有棕榈油合约
    results = {}
    contracts = ['p2209', 'p2401', 'p2405', 'p2601']  # 基准合约
    
    for contract in contracts:
        csv_files = list(data_dir.glob(f'{contract}*.csv'))
        if not csv_files:
            print(f"No data for {contract}")
            continue
        
        df = pd.read_csv(csv_files[0])
        print(f"\n=== {contract} ===")
        print(f"数据范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        
        gaps = get_holiday_gap_sessions(df)
        print(f"节假日跳空次数: {len(gaps)}")
        
        contract_gaps = []
        for gap in gaps:
            print(f"\n{gap['prev_date']} -> {gap['date']}: "
                  f"间隔{gap['days_gap']}天, "
                  f"跳空{gap['gap']:.0f}点 ({gap['gap_pct']:.2f}%), "
                  f"方向: {gap['gap_direction']}")
            
            # 分析分型
            fractal_analysis = analyze_fractal_around_gap(df, gap['date'])
            if fractal_analysis:
                gap['fractal_analysis'] = fractal_analysis
                print(f"  跳空后首个分型: {fractal_analysis['first_fractal_after']}")
            
            contract_gaps.append(gap)
        
        results[contract] = contract_gaps
    
    # 保存结果
    output_path = Path('/mnt/e/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/experiments/iter20_gap_fractal/holiday_gaps_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n结果已保存到: {output_path}")
    
    # 统计摘要
    print("\n=== 统计摘要 ===")
    total_gaps = sum(len(g) for g in results.values())
    print(f"总跳空次数: {total_gaps}")
    
    # 跳空方向统计
    up_gaps = sum(1 for gaps in results.values() for g in gaps if g['gap_direction'] == 'up')
    down_gaps = total_gaps - up_gaps
    print(f"向上跳空: {up_gaps}, 向下跳空: {down_gaps}")
    
    # 跳空幅度统计
    all_gaps = [g['gap_pct'] for gaps in results.values() for g in gaps]
    if all_gaps:
        print(f"跳空幅度: 最小{min(all_gaps):.2f}%, 最大{max(all_gaps):.2f}%, 平均{np.mean(all_gaps):.2f}%")

if __name__ == '__main__':
    main()
