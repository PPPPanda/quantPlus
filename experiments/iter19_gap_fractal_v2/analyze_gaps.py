#!/usr/bin/env python3
"""
节假日跳空数据分析 - 分合约诊断
"""

import json
from pathlib import Path
from collections import defaultdict
import statistics

# 数据路径
GAP_ANALYSIS_PATH = Path("/mnt/e/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/experiments/gap_analysis_results.json")
FRACTAL_IMPACT_PATH = Path("/mnt/e/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/experiments/iter18_gap_fractal/gap_fractal_impact.json")

def load_data():
    with open(GAP_ANALYSIS_PATH) as f:
        gap_data = json.load(f)
    with open(FRACTAL_IMPACT_PATH) as f:
        fractal_data = json.load(f)
    return gap_data, fractal_data

def analyze_by_contract(fractal_data):
    """分合约统计分型影响率"""
    results = {}
    for contract, data in fractal_data['by_contract'].items():
        gaps = data['gaps']
        total = len(gaps)
        in_fractal = sum(1 for g in gaps if g['gap_in_fractal'])
        impact_rate = in_fractal / total if total > 0 else 0
        
        # 计算 gap_atr 分布
        gap_atrs = [g['gap_atr'] for g in gaps]
        avg_gap_atr = statistics.mean(gap_atrs) if gap_atrs else 0
        max_gap_atr = max(gap_atrs) if gap_atrs else 0
        min_gap_atr = min(gap_atrs) if gap_atrs else 0
        
        results[contract] = {
            'total_gaps': total,
            'gaps_in_fractal': in_fractal,
            'impact_rate': impact_rate,
            'avg_gap_atr': avg_gap_atr,
            'max_gap_atr': max_gap_atr,
            'min_gap_atr': min_gap_atr,
            'gaps': gaps
        }
    return results

def analyze_gap_atr_distribution(fractal_data):
    """分析 gap_atr 分布"""
    all_gaps = []
    for contract, data in fractal_data['by_contract'].items():
        for g in data['gaps']:
            g['contract'] = contract
            all_gaps.append(g)
    
    # 分级统计
    tier1_count = sum(1 for g in all_gaps if g['gap_atr'] < 10)
    tier2_count = sum(1 for g in all_gaps if 10 <= g['gap_atr'] < 30)
    tier3_count = sum(1 for g in all_gaps if g['gap_atr'] >= 30)
    
    # 分级分型影响率
    tier1_gaps = [g for g in all_gaps if g['gap_atr'] < 10]
    tier2_gaps = [g for g in all_gaps if 10 <= g['gap_atr'] < 30]
    tier3_gaps = [g for g in all_gaps if g['gap_atr'] >= 30]
    
    tier1_impact = sum(1 for g in tier1_gaps if g['gap_in_fractal']) / len(tier1_gaps) if tier1_gaps else 0
    tier2_impact = sum(1 for g in tier2_gaps if g['gap_in_fractal']) / len(tier2_gaps) if tier2_gaps else 0
    tier3_impact = sum(1 for g in tier3_gaps if g['gap_in_fractal']) / len(tier3_gaps) if tier3_gaps else 0
    
    return {
        'total': len(all_gaps),
        'tier1': {'count': tier1_count, 'impact_rate': tier1_impact, 'threshold': '<10 ATR'},
        'tier2': {'count': tier2_count, 'impact_rate': tier2_impact, 'threshold': '10-30 ATR'},
        'tier3': {'count': tier3_count, 'impact_rate': tier3_impact, 'threshold': '>=30 ATR'},
        'all_gaps': all_gaps
    }

def analyze_p2401(fractal_data, gap_data):
    """深入分析 p2401"""
    p2401_data = fractal_data['by_contract'].get('p2401', {})
    gaps = p2401_data.get('gaps', [])
    
    # 基本统计
    total = len(gaps)
    in_fractal = sum(1 for g in gaps if g['gap_in_fractal'])
    impact_rate = in_fractal / total if total > 0 else 0
    
    # 详细分析每个跳空
    gap_details = []
    for g in gaps:
        gap_details.append({
            'datetime': g['datetime'],
            'gap': g['gap'],
            'gap_atr': g['gap_atr'],
            'hours_gap': g['hours_gap'],
            'gap_in_fractal': g['gap_in_fractal'],
            'compression_ratio': g['compression_ratio'],
            'total_fractals': g['total_fractals']
        })
    
    # 与其他合约对比
    other_contracts = {}
    for contract, data in fractal_data['by_contract'].items():
        if contract != 'p2401':
            gaps_list = data['gaps']
            other_total = len(gaps_list)
            other_in_fractal = sum(1 for g in gaps_list if g['gap_in_fractal'])
            other_impact_rate = other_in_fractal / other_total if other_total > 0 else 0
            other_contracts[contract] = {
                'total': other_total,
                'in_fractal': other_in_fractal,
                'impact_rate': other_impact_rate
            }
    
    return {
        'total_gaps': total,
        'gaps_in_fractal': in_fractal,
        'impact_rate': impact_rate,
        'gap_details': gap_details,
        'comparison': other_contracts
    }

def calculate_risk_score(contract_stats):
    """计算每个合约的跳空风险评分"""
    risk_scores = {}
    for contract, stats in contract_stats.items():
        # 风险评分 = 影响率 * 0.5 + 平均跳空ATR权重 * 0.3 + 跳空频率权重 * 0.2
        impact_score = stats['impact_rate'] * 100
        atr_score = min(stats['avg_gap_atr'] / 10 * 100, 100)  # 归一化到100
        freq_score = min(stats['total_gaps'] / 15 * 100, 100)  # 假设15个跳空是高频
        
        risk_score = impact_score * 0.5 + atr_score * 0.3 + freq_score * 0.2
        risk_scores[contract] = {
            'total_score': round(risk_score, 2),
            'impact_score': round(impact_score, 2),
            'atr_score': round(atr_score, 2),
            'freq_score': round(freq_score, 2),
            'impact_rate': round(stats['impact_rate'] * 100, 2),
            'avg_gap_atr': round(stats['avg_gap_atr'], 2),
            'total_gaps': stats['total_gaps']
        }
    
    # 按风险评分排序
    sorted_scores = dict(sorted(risk_scores.items(), key=lambda x: x[1]['total_score'], reverse=True))
    return sorted_scores

def suggest_thresholds(gap_distribution):
    """基于数据分布建议阈值"""
    all_gaps = gap_distribution['all_gaps']
    gap_atrs = sorted([g['gap_atr'] for g in all_gaps])
    
    # 计算分位数
    p25 = gap_atrs[int(len(gap_atrs) * 0.25)]
    p50 = gap_atrs[int(len(gap_atrs) * 0.50)]
    p75 = gap_atrs[int(len(gap_atrs) * 0.75)]
    p90 = gap_atrs[int(len(gap_atrs) * 0.90)]
    
    # 分析不同阈值下的分型影响率
    thresholds_analysis = []
    for t1 in [5, 6, 7, 8, 10]:
        for t2 in [15, 20, 25, 30]:
            if t1 >= t2:
                continue
            tier1 = [g for g in all_gaps if g['gap_atr'] < t1]
            tier2 = [g for g in all_gaps if t1 <= g['gap_atr'] < t2]
            tier3 = [g for g in all_gaps if g['gap_atr'] >= t2]
            
            t1_impact = sum(1 for g in tier1 if g['gap_in_fractal']) / len(tier1) if tier1 else 0
            t2_impact = sum(1 for g in tier2 if g['gap_in_fractal']) / len(tier2) if tier2 else 0
            t3_impact = sum(1 for g in tier3 if g['gap_in_fractal']) / len(tier3) if tier3 else 0
            
            # 计算区分度（各层级影响率差异）
            discrimination = abs(t1_impact - t2_impact) + abs(t2_impact - t3_impact)
            
            thresholds_analysis.append({
                'tier1_threshold': t1,
                'tier2_threshold': t2,
                'tier1_count': len(tier1),
                'tier2_count': len(tier2),
                'tier3_count': len(tier3),
                'tier1_impact': round(t1_impact * 100, 2),
                'tier2_impact': round(t2_impact * 100, 2),
                'tier3_impact': round(t3_impact * 100, 2),
                'discrimination': round(discrimination * 100, 2)
            })
    
    # 按区分度排序
    thresholds_analysis.sort(key=lambda x: x['discrimination'], reverse=True)
    
    return {
        'percentiles': {
            'p25': round(p25, 2),
            'p50': round(p50, 2),
            'p75': round(p75, 2),
            'p90': round(p90, 2)
        },
        'threshold_analysis': thresholds_analysis[:10]  # 前10个最佳组合
    }

def main():
    print("=" * 60)
    print("节假日跳空数据分析 - 分合约诊断")
    print("=" * 60)
    
    gap_data, fractal_data = load_data()
    
    # 1. 分合约统计
    print("\n1. 分合约分型影响率统计")
    print("-" * 40)
    contract_stats = analyze_by_contract(fractal_data)
    for contract, stats in sorted(contract_stats.items(), key=lambda x: x[1]['impact_rate'], reverse=True):
        print(f"{contract}: 总跳空={stats['total_gaps']}, 影响分型={stats['gaps_in_fractal']}, "
              f"影响率={stats['impact_rate']*100:.1f}%, 平均ATR={stats['avg_gap_atr']:.2f}")
    
    # 2. 跳空ATR分布
    print("\n2. 跳空ATR分布分析")
    print("-" * 40)
    gap_distribution = analyze_gap_atr_distribution(fractal_data)
    print(f"总跳空数: {gap_distribution['total']}")
    print(f"Tier1 (<10 ATR): {gap_distribution['tier1']['count']} 个, 影响率={gap_distribution['tier1']['impact_rate']*100:.1f}%")
    print(f"Tier2 (10-30 ATR): {gap_distribution['tier2']['count']} 个, 影响率={gap_distribution['tier2']['impact_rate']*100:.1f}%")
    print(f"Tier3 (>=30 ATR): {gap_distribution['tier3']['count']} 个, 影响率={gap_distribution['tier3']['impact_rate']*100:.1f}%")
    
    # 3. p2401 深入分析
    print("\n3. p2401 深入分析")
    print("-" * 40)
    p2401_analysis = analyze_p2401(fractal_data, gap_data)
    print(f"p2401 总跳空: {p2401_analysis['total_gaps']}")
    print(f"p2401 影响分型: {p2401_analysis['gaps_in_fractal']}")
    print(f"p2401 影响率: {p2401_analysis['impact_rate']*100:.1f}%")
    print("\np2401 各跳空详情:")
    for detail in p2401_analysis['gap_details']:
        status = "✓ 影响分型" if detail['gap_in_fractal'] else "✗ 未影响"
        print(f"  {detail['datetime']}: gap={detail['gap']}, ATR={detail['gap_atr']:.2f}, {status}")
    
    # 4. 风险评分
    print("\n4. 合约跳空风险评分")
    print("-" * 40)
    risk_scores = calculate_risk_score(contract_stats)
    for contract, scores in risk_scores.items():
        print(f"{contract}: 总分={scores['total_score']:.1f} "
              f"(影响={scores['impact_score']:.1f}, ATR={scores['atr_score']:.1f}, 频率={scores['freq_score']:.1f})")
    
    # 5. 阈值建议
    print("\n5. 分级阈值建议")
    print("-" * 40)
    threshold_suggestions = suggest_thresholds(gap_distribution)
    print(f"数据分位数: P25={threshold_suggestions['percentiles']['p25']}, "
          f"P50={threshold_suggestions['percentiles']['p50']}, "
          f"P75={threshold_suggestions['percentiles']['p75']}, "
          f"P90={threshold_suggestions['percentiles']['p90']}")
    print("\n最佳阈值组合（按区分度排序）:")
    for i, t in enumerate(threshold_suggestions['threshold_analysis'][:5], 1):
        print(f"  {i}. T1={t['tier1_threshold']}, T2={t['tier2_threshold']}: "
              f"Tier1影响={t['tier1_impact']:.1f}%({t['tier1_count']}), "
              f"Tier2影响={t['tier2_impact']:.1f}%({t['tier2_count']}), "
              f"Tier3影响={t['tier3_impact']:.1f}%({t['tier3_count']}), "
              f"区分度={t['discrimination']:.1f}")
    
    # 输出完整结果
    output = {
        'contract_stats': {k: {kk: vv for kk, vv in v.items() if kk != 'gaps'} for k, v in contract_stats.items()},
        'gap_distribution': {k: v for k, v in gap_distribution.items() if k != 'all_gaps'},
        'p2401_analysis': p2401_analysis,
        'risk_scores': risk_scores,
        'threshold_suggestions': threshold_suggestions
    }
    
    output_path = Path("/mnt/e/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/experiments/iter19_gap_fractal_v2/analysis_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {output_path}")

if __name__ == '__main__':
    main()
