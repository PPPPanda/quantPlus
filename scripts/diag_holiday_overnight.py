"""诊断脚本：分析节假日跳空和持仓过夜收益."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 中国期货市场主要节假日（按年份）
HOLIDAYS_2022 = [
    # 春节
    ("2022-01-31", "2022-02-06"),
    # 清明
    ("2022-04-03", "2022-04-05"),
    # 五一
    ("2022-04-30", "2022-05-04"),
    # 端午
    ("2022-06-03", "2022-06-05"),
    # 中秋
    ("2022-09-10", "2022-09-12"),
    # 国庆
    ("2022-10-01", "2022-10-07"),
]

HOLIDAYS_2023 = [
    # 春节
    ("2023-01-21", "2023-01-27"),
    # 清明
    ("2023-04-05", "2023-04-05"),
    # 五一
    ("2023-04-29", "2023-05-03"),
    # 端午
    ("2023-06-22", "2023-06-24"),
    # 中秋+国庆
    ("2023-09-29", "2023-10-06"),
]

HOLIDAYS_2024 = [
    # 春节
    ("2024-02-10", "2024-02-17"),
    # 清明
    ("2024-04-04", "2024-04-06"),
    # 五一
    ("2024-05-01", "2024-05-05"),
    # 端午
    ("2024-06-08", "2024-06-10"),
    # 中秋
    ("2024-09-15", "2024-09-17"),
    # 国庆
    ("2024-10-01", "2024-10-07"),
]

HOLIDAYS_2025 = [
    # 春节
    ("2025-01-28", "2025-02-04"),
    # 清明
    ("2025-04-04", "2025-04-06"),
    # 五一
    ("2025-05-01", "2025-05-05"),
    # 端午
    ("2025-05-31", "2025-06-02"),
    # 中秋+国庆
    ("2025-10-01", "2025-10-08"),
]

ALL_HOLIDAYS = HOLIDAYS_2022 + HOLIDAYS_2023 + HOLIDAYS_2024 + HOLIDAYS_2025

def is_holiday_gap(date1, date2):
    """检查两个日期之间是否包含节假日."""
    d1 = pd.to_datetime(date1).date()
    d2 = pd.to_datetime(date2).date()
    
    for h_start, h_end in ALL_HOLIDAYS:
        h_s = pd.to_datetime(h_start).date()
        h_e = pd.to_datetime(h_end).date()
        # 如果 d1 <= h_start 且 d2 >= h_end（跨越节假日）
        if d1 <= h_s and d2 >= h_e:
            return True
        # 如果 d1 在节假日前一天，d2 在节假日后
        if d1 < h_s and d2 > h_e:
            return True
    return False

def analyze_gaps(csv_path):
    """分析某合约的跳空情况."""
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 按天聚合
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    gaps = []
    for i in range(1, len(daily)):
        prev_close = daily.iloc[i-1]['close']
        curr_open = daily.iloc[i]['open']
        gap = curr_open - prev_close
        gap_pct = gap / prev_close * 100
        
        prev_date = daily.iloc[i-1]['date']
        curr_date = daily.iloc[i]['date']
        days_diff = (curr_date - prev_date).days
        
        is_holiday = is_holiday_gap(prev_date, curr_date) or days_diff > 3
        
        if abs(gap) >= 20:  # 显著跳空
            gaps.append({
                'prev_date': prev_date,
                'curr_date': curr_date,
                'days_off': days_diff,
                'prev_close': prev_close,
                'curr_open': curr_open,
                'gap': gap,
                'gap_pct': gap_pct,
                'is_holiday': is_holiday
            })
    
    return gaps

def main():
    ROOT = Path(__file__).parent.parent
    
    # 与 run_13bench.py 保持一致的数据路径
    BENCHMARKS = [
        ("p2201.DCE", ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv"),
        ("p2205.DCE", ROOT / "data/analyse/wind/p2205_1min_202112-202204.csv"),
        ("p2209.DCE", ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"),
        ("p2301.DCE", ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv"),
        ("p2305.DCE", ROOT / "data/analyse/wind/p2305_1min_202212-202304.csv"),
        ("p2309.DCE", ROOT / "data/analyse/wind/p2309_1min_202304-202308.csv"),
        ("p2401.DCE", ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv"),
        ("p2405.DCE", ROOT / "data/analyse/wind/p2405_1min_202312-202404.csv"),
        ("p2409.DCE", ROOT / "data/analyse/wind/p2409_1min_202401-202408.csv"),
        ("p2501.DCE", ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv"),
        ("p2505.DCE", ROOT / "data/analyse/wind/p2505_1min_202412-202504.csv"),
        ("p2509.DCE", ROOT / "data/analyse/wind/p2509_1min_202504-202508.csv"),
        ("p2601.DCE", ROOT / "data/analyse/p2601_1min_202507-202512.csv"),
    ]
    
    all_gaps = []
    
    for contract, csv_path in BENCHMARKS:
        
        if not csv_path.exists():
            print(f"[SKIP] {contract}: CSV not found")
            continue
        
        gaps = analyze_gaps(csv_path)
        for g in gaps:
            g['contract'] = contract
        all_gaps.extend(gaps)
        
        # 统计
        holiday_gaps = [g for g in gaps if g['is_holiday']]
        normal_gaps = [g for g in gaps if not g['is_holiday']]
        
        print(f"\n=== {contract} ===")
        print(f"Total gaps (>=20pts): {len(gaps)}")
        print(f"  Holiday gaps: {len(holiday_gaps)}")
        print(f"  Normal gaps: {len(normal_gaps)}")
        
        if holiday_gaps:
            avg_holiday_gap = np.mean([abs(g['gap']) for g in holiday_gaps])
            print(f"  Avg holiday gap: {avg_holiday_gap:.1f} pts")
        
        if normal_gaps:
            avg_normal_gap = np.mean([abs(g['gap']) for g in normal_gaps])
            print(f"  Avg normal gap: {avg_normal_gap:.1f} pts")
    
    # 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    holiday_gaps = [g for g in all_gaps if g['is_holiday']]
    normal_gaps = [g for g in all_gaps if not g['is_holiday']]
    
    print(f"Total holiday gaps: {len(holiday_gaps)}")
    print(f"Total normal gaps: {len(normal_gaps)}")
    
    if holiday_gaps:
        up_holiday = [g for g in holiday_gaps if g['gap'] > 0]
        down_holiday = [g for g in holiday_gaps if g['gap'] < 0]
        print(f"  Holiday UP gaps: {len(up_holiday)}, avg={np.mean([g['gap'] for g in up_holiday]):.1f}" if up_holiday else "  Holiday UP gaps: 0")
        print(f"  Holiday DOWN gaps: {len(down_holiday)}, avg={np.mean([g['gap'] for g in down_holiday]):.1f}" if down_holiday else "  Holiday DOWN gaps: 0")
    
    # 最大跳空
    print("\nTop 10 largest gaps:")
    sorted_gaps = sorted(all_gaps, key=lambda x: abs(x['gap']), reverse=True)[:10]
    for g in sorted_gaps:
        h_flag = "[HOLIDAY]" if g['is_holiday'] else ""
        print(f"  {g['contract']} {g['prev_date']}→{g['curr_date']}: {g['gap']:+.0f} pts ({g['gap_pct']:+.1f}%) {h_flag}")

if __name__ == "__main__":
    main()
