#!/usr/bin/env python3
"""分析跳空事件与回测亏损的时间关联."""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# 路径配置 - 使用相对路径
SCRIPT_DIR = Path(__file__).parent
GAP_DATA = SCRIPT_DIR / "analysis_results.json"
OUTPUT_DIR = SCRIPT_DIR


def load_gap_data():
    """加载跳空分析数据."""
    with open(GAP_DATA) as f:
        return json.load(f)


def analyze_p2401_gaps():
    """分析 p2401 的跳空详情."""
    data = load_gap_data()
    p2401 = data["p2401_analysis"]
    
    print("=" * 60)
    print("p2401 跳空事件详情")
    print("=" * 60)
    
    gaps = []
    for gap in p2401["gap_details"]:
        dt = datetime.fromisoformat(gap["datetime"])
        gaps.append({
            "datetime": dt,
            "gap": gap["gap"],
            "gap_atr": gap["gap_atr"],
            "hours_gap": gap["hours_gap"],
            "in_fractal": gap["gap_in_fractal"],
            "compression_ratio": gap["compression_ratio"]
        })
    
    df = pd.DataFrame(gaps)
    df = df.sort_values("datetime")
    
    # 统计
    affected = df[df["in_fractal"] == True]
    not_affected = df[df["in_fractal"] == False]
    
    print(f"\n总跳空事件: {len(df)}")
    print(f"影响分型: {len(affected)} ({len(affected)/len(df)*100:.1f}%)")
    print(f"未影响分型: {len(not_affected)} ({len(not_affected)/len(df)*100:.1f}%)")
    
    print(f"\n影响分型的跳空 ATR 分布:")
    print(f"  平均: {affected['gap_atr'].mean():.2f}")
    print(f"  最大: {affected['gap_atr'].max():.2f}")
    print(f"  最小: {affected['gap_atr'].min():.2f}")
    
    print(f"\n未影响分型的跳空 ATR 分布:")
    print(f"  平均: {not_affected['gap_atr'].mean():.2f}")
    print(f"  最大: {not_affected['gap_atr'].max():.2f}")
    print(f"  最小: {not_affected['gap_atr'].min():.2f}")
    
    # 详细列表
    print("\n详细跳空列表:")
    print("-" * 80)
    print(f"{'日期':<20} {'跳空(点)':<10} {'ATR倍数':<10} {'时间间隔':<12} {'影响分型':<10}")
    print("-" * 80)
    for _, row in df.iterrows():
        affect_str = "[Y]" if row["in_fractal"] else "[N]"
        print(f"{row['datetime'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{row['gap']:>+8.0f}  "
              f"{row['gap_atr']:>8.2f}  "
              f"{row['hours_gap']:>10.0f}h "
              f"{affect_str:<10}")
    
    return df


def analyze_risk_scores():
    """分析各合约风险评分."""
    data = load_gap_data()
    scores = data["risk_scores"]
    
    print("\n" + "=" * 60)
    print("合约风险评分排名")
    print("=" * 60)
    
    # 按总分排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["total_score"], reverse=True)
    
    print(f"\n{'排名':<6} {'合约':<10} {'总分':<10} {'影响率':<10} {'ATR分':<10} {'频率分':<10}")
    print("-" * 66)
    for i, (contract, s) in enumerate(sorted_scores, 1):
        risk = "[H]" if s["total_score"] > 70 else ("[M]" if s["total_score"] > 60 else "[L]")
        print(f"{risk} {i:<4} {contract:<10} {s['total_score']:<10.1f} "
              f"{s['impact_rate']:<10.1f} {s['atr_score']:<10.1f} {s['freq_score']:<10.1f}")


def analyze_threshold_optimization():
    """分析阈值优化建议."""
    data = load_gap_data()
    suggestions = data["threshold_suggestions"]
    
    print("\n" + "=" * 60)
    print("阈值优化建议")
    print("=" * 60)
    
    print(f"\n跳空 ATR 分布 (分位数):")
    for k, v in suggestions["percentiles"].items():
        print(f"  {k}: {v:.2f} ATR")
    
    print("\n最佳阈值组合 (按区分度排序):")
    print("-" * 90)
    print(f"{'T1阈值':<10} {'T2阈值':<10} {'T1数':<8} {'T2数':<8} {'T3数':<8} "
          f"{'T1影响%':<10} {'T2影响%':<10} {'T3影响%':<10} {'区分度':<10}")
    print("-" * 90)
    
    for t in suggestions["threshold_analysis"][:5]:
        print(f"{t['tier1_threshold']:<10} {t['tier2_threshold']:<10} "
              f"{t['tier1_count']:<8} {t['tier2_count']:<8} {t['tier3_count']:<8} "
              f"{t['tier1_impact']:<10.1f} {t['tier2_impact']:<10.1f} {t['tier3_impact']:<10.1f} "
              f"{t['discrimination']:<10.1f}")


def main():
    print("节假日跳空-亏损关联分析")
    print("=" * 60)
    
    # 1. p2401 跳空详情
    p2401_gaps = analyze_p2401_gaps()
    
    # 2. 风险评分
    analyze_risk_scores()
    
    # 3. 阈值优化
    analyze_threshold_optimization()
    
    # 输出结论
    print("\n" + "=" * 60)
    print("关键结论")
    print("=" * 60)
    print("""
1. p2401 分型影响率 81.8%（最高），风险评分 76.0（最高）
   - 这与其作为唯一 FAIL 合约的事实高度相关
   
2. 中等跳空(3-10 ATR)最危险
   - p2401 的 9 个影响分型的跳空全部在这个范围
   - 最大跳空(16.71 ATR)反而未影响分型
   
3. 阈值优化建议
   - 当前: tier1=10, tier2=30
   - 建议: tier1=6, tier2=15 (区分度提升 3x)
   
4. 下一步：关联 p2401 回测交易记录
   - 验证跳空后的交易是否是主要亏损来源
   - 计算：如果跳过跳空后 N 根 bar 的交易，PnL 变化
""")


if __name__ == "__main__":
    main()
