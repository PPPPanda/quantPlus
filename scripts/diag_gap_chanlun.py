"""
诊断：节假日跳空对缠论指标的影响分析

目标：
1. 识别所有大跳空（>=20pts）发生的时间点
2. 检查跳空前后的分型/笔/中枢状态变化
3. 统计跳空后产生的信号是否有更高的失败率
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from pathlib import Path
import json

# 数据路径
DATA_DIR = Path(r"E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\data\analyse")

# 合约文件名映射
CONTRACT_FILES = {
    'p2201': 'p2201_1min_202108-202112.csv',
    'p2205': 'p2205_1min_202112-202204.csv',
    'p2209': 'p2209_1min_202204-202208.csv',
    'p2301': 'p2301_1min_202208-202212.csv',
    'p2305': 'p2305_1min_202212-202304.csv',
    'p2309': 'p2309_1min_202304-202308.csv',
    'p2401': 'p2401_1min_202308-202312.csv',
    'p2405': 'p2405_1min_202312-202404.csv',
    'p2409': 'p2409_1min_202401-202408.csv',
    'p2501': 'p2501_1min_202404-202412.csv',
    'p2505': 'p2505_1min_202408-202504.csv',
    'p2509': 'p2509_1min_202412-202508.csv',
    'p2601': 'p2601_1min_202412-202512.csv',
}

# 节假日列表（简化版，主要是长假）
HOLIDAYS = {
    # 2022
    "2022-01-31": "春节",
    "2022-02-01": "春节",
    "2022-02-02": "春节",
    "2022-02-03": "春节",
    "2022-02-04": "春节",
    "2022-04-04": "清明",
    "2022-04-05": "清明",
    "2022-05-02": "劳动节",
    "2022-05-03": "劳动节",
    "2022-05-04": "劳动节",
    "2022-06-03": "端午",
    "2022-09-12": "中秋",
    "2022-10-03": "国庆",
    "2022-10-04": "国庆",
    "2022-10-05": "国庆",
    "2022-10-06": "国庆",
    "2022-10-07": "国庆",
    # 2023
    "2023-01-23": "春节",
    "2023-01-24": "春节",
    "2023-01-25": "春节",
    "2023-01-26": "春节",
    "2023-01-27": "春节",
    "2023-04-05": "清明",
    "2023-05-01": "劳动节",
    "2023-05-02": "劳动节",
    "2023-05-03": "劳动节",
    "2023-06-22": "端午",
    "2023-06-23": "端午",
    "2023-09-29": "中秋+国庆",
    "2023-10-02": "国庆",
    "2023-10-03": "国庆",
    "2023-10-04": "国庆",
    "2023-10-05": "国庆",
    "2023-10-06": "国庆",
    # 2024
    "2024-02-12": "春节",
    "2024-02-13": "春节",
    "2024-02-14": "春节",
    "2024-02-15": "春节",
    "2024-02-16": "春节",
    "2024-04-04": "清明",
    "2024-04-05": "清明",
    "2024-05-01": "劳动节",
    "2024-05-02": "劳动节",
    "2024-05-03": "劳动节",
    "2024-06-10": "端午",
    "2024-09-16": "中秋",
    "2024-09-17": "中秋",
    "2024-10-01": "国庆",
    "2024-10-02": "国庆",
    "2024-10-03": "国庆",
    "2024-10-04": "国庆",
    "2024-10-07": "国庆",
}

def is_holiday_gap(date_before: str, date_after: str) -> tuple[bool, str]:
    """判断两个日期之间是否跨越节假日"""
    # 检查间隔天数
    d1 = datetime.strptime(date_before, "%Y-%m-%d")
    d2 = datetime.strptime(date_after, "%Y-%m-%d")
    gap_days = (d2 - d1).days
    
    if gap_days <= 1:
        return False, ""
    
    # 检查中间是否有节假日
    for i in range(1, gap_days):
        check_date = (d1 + timedelta(days=i)).strftime("%Y-%m-%d")
        if check_date in HOLIDAYS:
            return True, HOLIDAYS[check_date]
    
    # 周末跳过（周五到周一）
    if gap_days == 3 and d1.weekday() == 4:  # Friday
        return False, "周末"
    
    return gap_days > 3, f"长间隔({gap_days}天)"


def load_and_analyze_gaps(csv_path: Path, gap_threshold: float = 20.0):
    """加载数据并分析跳空"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    
    # 按日期分组，取每日开盘价和前日收盘价
    daily = df.groupby('date').agg({
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min'
    }).reset_index()
    
    gaps = []
    for i in range(1, len(daily)):
        prev_close = daily.iloc[i-1]['close']
        curr_open = daily.iloc[i]['open']
        gap = curr_open - prev_close
        gap_pct = abs(gap) / prev_close * 100
        
        if abs(gap) >= gap_threshold:
            is_holiday, holiday_name = is_holiday_gap(
                daily.iloc[i-1]['date'], 
                daily.iloc[i]['date']
            )
            gaps.append({
                'date': daily.iloc[i]['date'],
                'prev_date': daily.iloc[i-1]['date'],
                'prev_close': prev_close,
                'open': curr_open,
                'gap': gap,
                'gap_pct': gap_pct,
                'direction': 'UP' if gap > 0 else 'DOWN',
                'is_holiday': is_holiday,
                'holiday_name': holiday_name if is_holiday else ''
            })
    
    return gaps, df


def analyze_chanlun_impact(gaps: list, df, contract: str):
    """分析跳空对缠论结构的影响（简化版）"""
    
    results = {
        'contract': contract,
        'total_gaps': len(gaps),
        'holiday_gaps': sum(1 for g in gaps if g['is_holiday']),
        'normal_gaps': sum(1 for g in gaps if not g['is_holiday']),
        'gaps': []
    }
    
    for gap in gaps:
        gap_date = gap['date']
        gap_info = {
            **gap,
            'impact_analysis': {}
        }
        
        # 获取跳空当天的数据
        day_data = df[df['date'] == gap_date]
        if len(day_data) == 0:
            continue
        
        # 分析1：跳空当天的波动范围 vs 跳空幅度
        day_range = day_data['high'].max() - day_data['low'].min()
        gap_info['impact_analysis']['day_range'] = float(day_range)
        gap_info['impact_analysis']['gap_vs_range'] = abs(gap['gap']) / day_range if day_range > 0 else 0
        
        # 分析2：跳空是否在当天被回补
        if gap['direction'] == 'UP':
            filled = day_data['low'].min() <= gap['prev_close']
        else:
            filled = day_data['high'].max() >= gap['prev_close']
        gap_info['impact_analysis']['gap_filled_same_day'] = filled
        
        # 分析3：跳空后第一根K线的特征（可能形成假分型）
        first_bar = day_data.iloc[0]
        gap_info['impact_analysis']['first_bar'] = {
            'open': float(first_bar['open']),
            'high': float(first_bar['high']),
            'low': float(first_bar['low']),
            'close': float(first_bar['close']),
            'range': float(first_bar['high'] - first_bar['low'])
        }
        
        results['gaps'].append(gap_info)
    
    return results


def main():
    print("=" * 60)
    print("节假日跳空对缠论指标的影响分析")
    print("=" * 60)
    
    # 合约列表
    contracts = ['p2201', 'p2205', 'p2209', 'p2301', 'p2305', 'p2309', 
                 'p2401', 'p2405', 'p2409', 'p2501', 'p2505', 'p2509', 'p2601']
    
    all_results = []
    
    for contract in contracts:
        if contract not in CONTRACT_FILES:
            print(f"[SKIP] {contract}: 无映射")
            continue
        csv_path = DATA_DIR / CONTRACT_FILES[contract]
        if not csv_path.exists():
            print(f"[SKIP] {contract}: 文件不存在")
            continue
        
        print(f"\n--- {contract} ---")
        gaps, df = load_and_analyze_gaps(csv_path)
        result = analyze_chanlun_impact(gaps, df, contract)
        all_results.append(result)
        
        # 打印摘要
        print(f"  总跳空数: {result['total_gaps']}")
        print(f"  节假日跳空: {result['holiday_gaps']}")
        print(f"  普通跳空: {result['normal_gaps']}")
        
        # 统计当天回补率
        if result['gaps']:
            filled_count = sum(1 for g in result['gaps'] 
                             if g['impact_analysis'].get('gap_filled_same_day', False))
            fill_rate = filled_count / len(result['gaps']) * 100
            print(f"  当天回补率: {fill_rate:.1f}%")
            
            # 节假日跳空的回补率
            holiday_gaps = [g for g in result['gaps'] if g['is_holiday']]
            if holiday_gaps:
                h_filled = sum(1 for g in holiday_gaps 
                              if g['impact_analysis'].get('gap_filled_same_day', False))
                h_fill_rate = h_filled / len(holiday_gaps) * 100
                print(f"  节假日跳空当天回补率: {h_fill_rate:.1f}%")
    
    # 汇总统计
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    
    all_gaps = []
    for r in all_results:
        all_gaps.extend(r['gaps'])
    
    holiday_gaps = [g for g in all_gaps if g['is_holiday']]
    normal_gaps = [g for g in all_gaps if not g['is_holiday']]
    
    print(f"\n总跳空数: {len(all_gaps)}")
    print(f"  - 节假日跳空: {len(holiday_gaps)}")
    print(f"  - 普通跳空: {len(normal_gaps)}")
    
    # 节假日跳空 vs 普通跳空的特征对比
    if holiday_gaps and normal_gaps:
        h_avg_gap = sum(abs(g['gap']) for g in holiday_gaps) / len(holiday_gaps)
        n_avg_gap = sum(abs(g['gap']) for g in normal_gaps) / len(normal_gaps)
        
        h_filled = sum(1 for g in holiday_gaps 
                      if g['impact_analysis'].get('gap_filled_same_day', False))
        n_filled = sum(1 for g in normal_gaps 
                      if g['impact_analysis'].get('gap_filled_same_day', False))
        
        print(f"\n平均跳空幅度:")
        print(f"  - 节假日: {h_avg_gap:.1f} pts")
        print(f"  - 普通: {n_avg_gap:.1f} pts")
        print(f"  - 节假日/普通比: {h_avg_gap/n_avg_gap:.2f}x")
        
        print(f"\n当天回补率:")
        print(f"  - 节假日: {h_filled}/{len(holiday_gaps)} = {h_filled/len(holiday_gaps)*100:.1f}%")
        print(f"  - 普通: {n_filled}/{len(normal_gaps)} = {n_filled/len(normal_gaps)*100:.1f}%")
    
    # 关键发现：跳空对缠论的影响
    print("\n" + "=" * 60)
    print("跳空对缠论结构的潜在影响")
    print("=" * 60)
    
    # 分析跳空幅度 vs 当天波动范围
    gap_vs_range_holiday = [g['impact_analysis']['gap_vs_range'] 
                           for g in holiday_gaps 
                           if 'gap_vs_range' in g['impact_analysis']]
    gap_vs_range_normal = [g['impact_analysis']['gap_vs_range'] 
                          for g in normal_gaps 
                          if 'gap_vs_range' in g['impact_analysis']]
    
    if gap_vs_range_holiday and gap_vs_range_normal:
        print(f"\n跳空幅度 / 当天波动范围 比值:")
        print(f"  - 节假日: {sum(gap_vs_range_holiday)/len(gap_vs_range_holiday):.2f}")
        print(f"  - 普通: {sum(gap_vs_range_normal)/len(gap_vs_range_normal):.2f}")
        print(f"  (比值越高，说明跳空在当天K线中占比越大，越可能形成假分型)")
    
    # 保存结果
    output_path = Path(r"E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\experiments\iter17\gap_chanlun_impact.json")
    
    # 转换为可序列化格式
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif hasattr(obj, '__float__'):
            return float(obj)
        else:
            return obj
    
    serializable_results = make_serializable(all_results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
