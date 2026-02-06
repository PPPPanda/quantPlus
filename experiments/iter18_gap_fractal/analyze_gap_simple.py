#!/usr/bin/env python3
"""
iter18: 分析节假日跳空对缠论分型/笔的影响（简化版，不依赖vnpy）
"""

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict


def parse_datetime(s):
    """解析日期时间字符串"""
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s[:19], fmt)
        except:
            continue
    return None


def load_csv(filepath):
    """加载CSV文件"""
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def calculate_atr(rows, window=70):
    """计算ATR"""
    prev_close = None
    tr_values = []
    
    for i, row in enumerate(rows):
        high = float(row.get('high', 0))
        low = float(row.get('low', 0))
        close = float(row.get('close', 0))
        
        if prev_close is None:
            prev_close = close
            row['atr'] = 0
            continue
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
        
        if len(tr_values) >= window:
            row['atr'] = sum(tr_values[-window:]) / window
        else:
            row['atr'] = sum(tr_values) / len(tr_values) if tr_values else 0
        
        prev_close = close
    
    return rows


def find_holiday_gaps(rows):
    """找出节假日跳空（>48小时）"""
    holidays = []
    prev_row = None
    
    for i, row in enumerate(rows):
        dt = parse_datetime(row.get('datetime', ''))
        if dt is None:
            continue
        
        if prev_row is not None:
            prev_dt = parse_datetime(prev_row.get('datetime', ''))
            if prev_dt is not None:
                hours_gap = (dt - prev_dt).total_seconds() / 3600
                
                if hours_gap > 48:  # 节假日
                    prev_close = float(prev_row.get('close', 0))
                    open_price = float(row.get('open', 0))
                    gap = open_price - prev_close
                    atr = float(row.get('atr', 1)) or 1
                    gap_atr = abs(gap) / atr
                    
                    if gap_atr > 3:  # 只关注显著跳空
                        holidays.append({
                            'idx': i,
                            'datetime': str(dt),
                            'prev_datetime': str(prev_dt),
                            'gap': gap,
                            'gap_atr': gap_atr,
                            'hours_gap': hours_gap,
                            'open': open_price,
                            'prev_close': prev_close,
                            'atr': atr,
                        })
        
        prev_row = row
    
    return holidays


def simulate_inclusion(rows, start_idx, end_idx):
    """
    模拟包含处理
    
    返回：
    - k_lines: 包含处理后的K线列表
    - gap_info: 跳空K线的处理信息
    """
    k_lines = []
    inclusion_dir = 0  # 0=未定, 1=向上, -1=向下
    
    gap_idx_local = None  # 跳空在k_lines中的位置
    gap_merged = False
    merged_count = 0
    
    for i in range(start_idx, min(end_idx, len(rows))):
        row = rows[i]
        high = float(row.get('high', 0))
        low = float(row.get('low', 0))
        is_gap_bar = (i == start_idx + (end_idx - start_idx) // 2)  # 中间位置假设为跳空
        
        new_bar = {
            'idx': i,
            'high': high,
            'low': low,
            'is_gap': is_gap_bar,
            'merged_count': 1,
        }
        
        if not k_lines:
            k_lines.append(new_bar)
            if is_gap_bar:
                gap_idx_local = 0
            continue
        
        last = k_lines[-1]
        
        # 检查包含关系
        in_last = new_bar['high'] <= last['high'] and new_bar['low'] >= last['low']
        in_new = last['high'] <= new_bar['high'] and last['low'] >= new_bar['low']
        
        if in_last or in_new:
            # 存在包含关系
            if inclusion_dir == 0:
                # 方向未定，不合并
                k_lines.append(new_bar)
                if is_gap_bar:
                    gap_idx_local = len(k_lines) - 1
                continue
            
            # 合并
            merged = last.copy()
            merged['merged_count'] = last.get('merged_count', 1) + 1
            merged['contains_gap'] = last.get('contains_gap', False) or is_gap_bar
            
            if inclusion_dir == 1:  # 向上
                merged['high'] = max(last['high'], new_bar['high'])
                merged['low'] = max(last['low'], new_bar['low'])
            else:  # 向下
                merged['high'] = min(last['high'], new_bar['high'])
                merged['low'] = min(last['low'], new_bar['low'])
            
            k_lines[-1] = merged
            
            if is_gap_bar:
                gap_merged = True
                merged_count = merged['merged_count']
        else:
            # 无包含关系，更新方向
            if new_bar['high'] > last['high'] and new_bar['low'] > last['low']:
                inclusion_dir = 1
            elif new_bar['high'] < last['high'] and new_bar['low'] < last['low']:
                inclusion_dir = -1
            
            k_lines.append(new_bar)
            if is_gap_bar:
                gap_idx_local = len(k_lines) - 1
    
    return k_lines, {
        'gap_merged': gap_merged,
        'merged_count': merged_count,
        'gap_idx_local': gap_idx_local,
    }


def detect_fractals(k_lines):
    """检测分型"""
    fractals = []
    
    for i in range(1, len(k_lines) - 1):
        left = k_lines[i - 1]
        mid = k_lines[i]
        right = k_lines[i + 1]
        
        is_top = mid['high'] > left['high'] and mid['high'] > right['high']
        is_bot = mid['low'] < left['low'] and mid['low'] < right['low']
        
        if is_top:
            fractals.append({
                'type': 'top',
                'idx': i,
                'price': mid['high'],
                'contains_gap': mid.get('contains_gap', False) or mid.get('is_gap', False),
            })
        elif is_bot:
            fractals.append({
                'type': 'bottom',
                'idx': i,
                'price': mid['low'],
                'contains_gap': mid.get('contains_gap', False) or mid.get('is_gap', False),
            })
    
    return fractals


def analyze_gap_impact(rows, gap_info, window=30):
    """分析单个跳空的影响"""
    gap_idx = gap_info['idx']
    
    # 取跳空前后的数据
    start_idx = max(0, gap_idx - window)
    end_idx = min(len(rows), gap_idx + window)
    
    # 模拟包含处理
    k_lines, inclusion_info = simulate_inclusion(rows, start_idx, end_idx)
    
    # 检测分型
    fractals = detect_fractals(k_lines)
    
    # 统计
    gap_in_fractal = any(f.get('contains_gap', False) for f in fractals)
    
    return {
        **gap_info,
        'gap_merged': inclusion_info['gap_merged'],
        'merged_count': inclusion_info['merged_count'],
        'total_fractals': len(fractals),
        'gap_in_fractal': gap_in_fractal,
        'k_lines_count': len(k_lines),
        'original_bars': end_idx - start_idx,
        'compression_ratio': len(k_lines) / (end_idx - start_idx) if (end_idx - start_idx) > 0 else 1,
    }


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent.parent
    data_dirs = [
        project_root / "data/analyse/wind",
        project_root / "data/analyse",
    ]
    
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
        
        # 找数据文件
        filepath = None
        for data_dir in data_dirs:
            import glob
            pattern = str(data_dir / f"{contract}*_1min*.csv")
            files = glob.glob(pattern)
            if files:
                filepath = files[0]
                break
        
        if not filepath:
            print(f"  Skipped: no data file found")
            continue
        
        # 加载数据
        rows = load_csv(filepath)
        print(f"  Loaded {len(rows)} bars")
        
        # 计算ATR
        rows = calculate_atr(rows)
        
        # 找节假日跳空
        holidays = find_holiday_gaps(rows)
        print(f"  Found {len(holidays)} significant holiday gaps (>3 ATR, >48h)")
        
        if not holidays:
            continue
        
        # 分析每个跳空的影响
        contract_results = []
        for h in holidays:
            impact = analyze_gap_impact(rows, h)
            contract_results.append(impact)
            
            total_gaps += 1
            if impact['gap_merged']:
                total_merged += 1
            if impact['gap_in_fractal']:
                total_in_fractal += 1
            
            # 添加到全局
            impact['contract'] = contract
            results['all_holiday_gaps'].append(impact)
        
        results['by_contract'][contract] = {
            'total_gaps': len(contract_results),
            'gaps': contract_results,
        }
        
        # 打印摘要
        merged_cnt = sum(1 for r in contract_results if r['gap_merged'])
        fractal_cnt = sum(1 for r in contract_results if r['gap_in_fractal'])
        print(f"  Gap merged: {merged_cnt}/{len(contract_results)}")
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
    print(f"Gaps merged by inclusion: {total_merged} ({total_merged/total_gaps*100:.1f}%)" if total_gaps > 0 else "")
    print(f"Gaps affecting fractals: {total_in_fractal} ({total_in_fractal/total_gaps*100:.1f}%)" if total_gaps > 0 else "")
    
    # 保存结果
    output_path = Path(__file__).parent / "gap_fractal_impact.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
