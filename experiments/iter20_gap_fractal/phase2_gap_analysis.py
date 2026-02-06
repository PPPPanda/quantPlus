#!/usr/bin/env python3
"""
Phase 2: 跳空数据深度分析
基于已收集的跳空数据，分析失败模式
"""

import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

GAPS_FILE = Path(__file__).parent / "holiday_gaps_analysis.json"
OUTPUT_FILE = Path(__file__).parent / "phase2_analysis_results.json"

def load_gaps():
    with open(GAPS_FILE) as f:
        return json.load(f)

def analyze_contract(contract: str, gaps: list) -> dict:
    """分析单个合约的跳空特征"""
    if not gaps:
        return {}
    
    analysis = {
        'total_gaps': len(gaps),
        'holiday_gaps': 0,
        'weekend_gaps': 0,
        'large_gaps': 0,  # >1%
        'medium_gaps': 0, # 0.5-1%
        'small_gaps': 0,  # <0.5%
        
        # 方向分析
        'up_gaps': 0,
        'down_gaps': 0,
        
        # 分型一致性
        'consistent': 0,
        'inconsistent': 0,
        
        # 节假日 vs 周末
        'holiday_consistent': 0,
        'holiday_total': 0,
        'weekend_consistent': 0,
        'weekend_total': 0,
        
        # 大跳空延迟
        'large_gap_immediate': 0,  # 开盘即分型
        'large_gap_delayed': 0,    # 延迟分型
        
        # 详细记录
        'failure_cases': [],  # 不一致的情况
        'holiday_failures': [],
    }
    
    for gap in gaps:
        gap_pct = abs(gap['gap_pct'])
        
        # 分类
        if gap['crossed_holiday']:
            analysis['holiday_gaps'] += 1
            analysis['holiday_total'] += 1
        else:
            analysis['weekend_gaps'] += 1
            analysis['weekend_total'] += 1
        
        if gap_pct > 1.0:
            analysis['large_gaps'] += 1
        elif gap_pct > 0.5:
            analysis['medium_gaps'] += 1
        else:
            analysis['small_gaps'] += 1
        
        if gap['gap_direction'] == 'up':
            analysis['up_gaps'] += 1
        else:
            analysis['down_gaps'] += 1
        
        # 分型一致性分析
        fa = gap.get('fractal_analysis', {}).get('first_fractal_after', {})
        if fa:
            gap_dir = gap['gap_direction']
            frac_type = fa.get('type', '')
            
            # 顺势判定：向上跳空→顶分型，向下跳空→底分型
            is_consistent = (gap_dir == 'up' and frac_type == 'top') or \
                           (gap_dir == 'down' and frac_type == 'bottom')
            
            if is_consistent:
                analysis['consistent'] += 1
                if gap['crossed_holiday']:
                    analysis['holiday_consistent'] += 1
                else:
                    analysis['weekend_consistent'] += 1
            else:
                analysis['inconsistent'] += 1
                analysis['failure_cases'].append({
                    'date': gap['date'],
                    'gap_direction': gap_dir,
                    'gap_pct': gap['gap_pct'],
                    'first_fractal_type': frac_type,
                    'is_holiday': gap['crossed_holiday'],
                })
                if gap['crossed_holiday']:
                    analysis['holiday_failures'].append({
                        'date': gap['date'],
                        'gap_pct': gap['gap_pct'],
                        'gap_direction': gap_dir,
                        'fractal_type': frac_type,
                    })
            
            # 大跳空延迟分析
            if gap_pct > 1.0:
                frac_time = fa.get('datetime', '')
                if frac_time.endswith('09:00:00') or frac_time.endswith('09:05:00'):
                    analysis['large_gap_immediate'] += 1
                else:
                    analysis['large_gap_delayed'] += 1
    
    # 计算比率
    total = analysis['consistent'] + analysis['inconsistent']
    analysis['consistency_rate'] = analysis['consistent'] / total if total > 0 else 0
    
    if analysis['holiday_total'] > 0:
        analysis['holiday_consistency_rate'] = analysis['holiday_consistent'] / analysis['holiday_total']
    else:
        analysis['holiday_consistency_rate'] = 0
    
    if analysis['weekend_total'] > 0:
        analysis['weekend_consistency_rate'] = analysis['weekend_consistent'] / analysis['weekend_total']
    else:
        analysis['weekend_consistency_rate'] = 0
    
    if analysis['large_gaps'] > 0:
        analysis['large_gap_immediate_rate'] = analysis['large_gap_immediate'] / analysis['large_gaps']
    else:
        analysis['large_gap_immediate_rate'] = 0
    
    return analysis


def identify_failure_patterns(all_analysis: dict) -> dict:
    """识别跨合约的失败模式"""
    patterns = {
        'pattern_1_holiday_reversal': {
            'description': '节假日跳空后首个分型反向（最危险）',
            'cases': [],
            'impact': 'HIGH',
        },
        'pattern_2_large_gap_delay': {
            'description': '大跳空(>1%)分型延迟，入场时机错误',
            'cases': [],
            'impact': 'MEDIUM',
        },
        'pattern_3_weekend_fake_fractal': {
            'description': '普通周末的假分型（较少）',
            'cases': [],
            'impact': 'LOW',
        },
    }
    
    for contract, analysis in all_analysis.items():
        # Pattern 1: 节假日反向
        for failure in analysis.get('holiday_failures', []):
            patterns['pattern_1_holiday_reversal']['cases'].append({
                'contract': contract,
                **failure
            })
        
        # Pattern 2: 大跳空延迟
        if analysis.get('large_gap_delayed', 0) > 0:
            patterns['pattern_2_large_gap_delay']['cases'].append({
                'contract': contract,
                'delayed_count': analysis['large_gap_delayed'],
                'total_large': analysis['large_gaps'],
            })
        
        # Pattern 3: 周末假分型
        for failure in analysis.get('failure_cases', []):
            if not failure['is_holiday']:
                patterns['pattern_3_weekend_fake_fractal']['cases'].append({
                    'contract': contract,
                    **failure
                })
    
    return patterns


def generate_recommendations(patterns: dict, all_analysis: dict) -> list:
    """基于失败模式生成修复建议"""
    recommendations = []
    
    # R1: 节假日处理
    holiday_failures = len(patterns['pattern_1_holiday_reversal']['cases'])
    if holiday_failures > 0:
        recommendations.append({
            'id': 'R1',
            'priority': 'HIGH',
            'title': '节假日跳空专项处理',
            'problem': f'{holiday_failures}次节假日跳空后分型反向',
            'solution': [
                '方案A: 延长节后冷却期 (gap_cooldown_bars: 6→12)',
                '方案B: 节后首个分型降权 (gap_confidence_mult: 0.4→0.2)',
                '方案C: 节后中枢状态重置 (会话边界机制)',
            ],
            'expected_improvement': '减少节后30分钟内的假信号',
        })
    
    # R2: 大跳空延迟
    delay_cases = patterns['pattern_2_large_gap_delay']['cases']
    if delay_cases:
        total_delayed = sum(c['delayed_count'] for c in delay_cases)
        recommendations.append({
            'id': 'R2',
            'priority': 'MEDIUM',
            'title': '大跳空分型确认机制',
            'problem': f'{total_delayed}次大跳空分型延迟形成',
            'solution': [
                '方案A: 大跳空后延长确认期 (gap_confirm_bars: 2→4)',
                '方案B: 大跳空后ATR加权 (已有S27, 需验证)',
            ],
            'expected_improvement': '避免追高/追低入场',
        })
    
    # R3: 汇总数据
    avg_holiday = sum(a['holiday_consistency_rate'] for a in all_analysis.values()) / len(all_analysis)
    avg_weekend = sum(a['weekend_consistency_rate'] for a in all_analysis.values()) / len(all_analysis)
    gap_diff = avg_weekend - avg_holiday
    
    recommendations.append({
        'id': 'R3',
        'priority': 'INFO',
        'title': '关键数据汇总',
        'data': {
            'holiday_consistency': f'{avg_holiday:.1%}',
            'weekend_consistency': f'{avg_weekend:.1%}',
            'gap': f'{gap_diff:.1%}',
        },
        'insight': f'节假日比周末低{gap_diff:.1%}的一致性，这是核心问题',
    })
    
    return recommendations


def main():
    print("=== Phase 2: 跳空数据深度分析 ===\n")
    
    gaps = load_gaps()
    print(f"加载跳空数据: {sum(len(v) for v in gaps.values())} 次\n")
    
    all_analysis = {}
    
    for contract in ['p2209', 'p2401', 'p2405', 'p2601']:
        print(f"--- {contract} ---")
        contract_gaps = gaps.get(contract, [])
        analysis = analyze_contract(contract, contract_gaps)
        all_analysis[contract] = analysis
        
        print(f"  总跳空: {analysis['total_gaps']}")
        print(f"  节假日: {analysis['holiday_gaps']} | 周末: {analysis['weekend_gaps']}")
        print(f"  大/中/小: {analysis['large_gaps']}/{analysis['medium_gaps']}/{analysis['small_gaps']}")
        print(f"  方向一致率: {analysis['consistency_rate']:.1%}")
        print(f"    - 节假日: {analysis['holiday_consistency_rate']:.1%}")
        print(f"    - 周末: {analysis['weekend_consistency_rate']:.1%}")
        print(f"  大跳空即时率: {analysis['large_gap_immediate_rate']:.1%}")
        print(f"  失败案例: {len(analysis['failure_cases'])}例")
        print()
    
    # 识别失败模式
    print("=== 失败模式识别 ===")
    patterns = identify_failure_patterns(all_analysis)
    
    for name, pattern in patterns.items():
        print(f"\n{pattern['description']}")
        print(f"  影响级别: {pattern['impact']}")
        print(f"  案例数: {len(pattern['cases'])}")
        if pattern['cases'] and len(pattern['cases']) <= 5:
            for case in pattern['cases'][:3]:
                print(f"    - {case}")
    
    # 生成建议
    print("\n=== 修复建议 ===")
    recommendations = generate_recommendations(patterns, all_analysis)
    
    for rec in recommendations:
        print(f"\n[{rec['id']}] {rec['title']} (优先级: {rec['priority']})")
        if 'problem' in rec:
            print(f"  问题: {rec['problem']}")
        if 'solution' in rec:
            for sol in rec['solution']:
                print(f"  → {sol}")
        if 'data' in rec:
            for k, v in rec['data'].items():
                print(f"  {k}: {v}")
        if 'insight' in rec:
            print(f"  💡 {rec['insight']}")
    
    # 保存结果
    result = {
        'analysis': all_analysis,
        'patterns': {k: {'description': v['description'], 'impact': v['impact'], 'case_count': len(v['cases'])} for k, v in patterns.items()},
        'recommendations': recommendations,
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n\n结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
