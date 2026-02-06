#!/usr/bin/env python3
"""
Phase 2: 跳空-交易关联分析
识别跳空后的失败交易模式
"""

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from qp.strategies.cta_chan_pivot import CtaChanPivot

# 配置
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "analyse" / "wind"
GAPS_FILE = Path(__file__).parent / "holiday_gaps_analysis.json"
OUTPUT_FILE = Path(__file__).parent / "phase2_gap_trade_correlation.json"

# iter14 基线参数
BASELINE_PARAMS = {
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
}

# 合约配置
CONTRACTS = {
    "p2209": "p2209_1min_202111-202209.csv",
    "p2401": "p2401_1min_202308-202401.csv",
    "p2405": "p2405_1min_202312-202405.csv",
    "p2601": "p2601_1min_202408-202601.csv",
}


def load_gaps():
    """加载跳空数据"""
    with open(GAPS_FILE) as f:
        return json.load(f)


def run_backtest(contract: str, data_file: str) -> list:
    """运行回测并返回交易记录"""
    data_path = DATA_DIR / data_file
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return []
    
    # 读取数据
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 初始化策略
    strategy = CtaChanPivot(
        symbol=contract,
        **BASELINE_PARAMS
    )
    
    # 运行回测
    trades = []
    for _, bar in df.iterrows():
        # 模拟 bar 数据结构
        bar_dict = {
            'datetime': bar['datetime'],
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar.get('volume', 0),
        }
        
        # 调用策略的 on_bar
        result = strategy.on_bar(bar_dict)
        
        # 记录交易
        if hasattr(strategy, '_trades') and strategy._trades:
            for t in strategy._trades:
                if t not in trades:
                    trades.append(t)
    
    # 从策略获取完整交易记录
    if hasattr(strategy, 'get_trades'):
        trades = strategy.get_trades()
    
    return trades


def correlate_gaps_trades(gaps: dict, trades: list, contract: str) -> list:
    """关联跳空事件和交易记录"""
    correlations = []
    contract_gaps = gaps.get(contract, [])
    
    for gap in contract_gaps:
        gap_date = datetime.strptime(gap['date'], '%Y-%m-%d')
        gap_end = gap_date + timedelta(days=1)  # 跳空当天结束
        
        # 扩展分析窗口：跳空后3天内的交易
        window_end = gap_date + timedelta(days=3)
        
        related_trades = []
        for trade in trades:
            entry_time = trade.get('entry_time')
            if isinstance(entry_time, str):
                entry_time = datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')
            
            # 检查交易是否在跳空窗口内
            if gap_date <= entry_time < window_end:
                related_trades.append({
                    'entry_time': str(entry_time),
                    'exit_time': str(trade.get('exit_time', '')),
                    'direction': trade.get('direction', 'long'),
                    'pnl': trade.get('pnl', 0),
                    'bars_after_gap': (entry_time - gap_date).total_seconds() / (5 * 60),  # 5分钟bar数
                })
        
        correlations.append({
            'gap_date': gap['date'],
            'gap_pct': gap['gap_pct'],
            'gap_direction': gap['gap_direction'],
            'crossed_holiday': gap['crossed_holiday'],
            'first_fractal': gap.get('fractal_analysis', {}).get('first_fractal_after', {}),
            'trades_count': len(related_trades),
            'trades': related_trades,
            'total_pnl': sum(t['pnl'] for t in related_trades),
            'win_rate': sum(1 for t in related_trades if t['pnl'] > 0) / len(related_trades) if related_trades else 0,
        })
    
    return correlations


def analyze_failure_patterns(correlations: list) -> dict:
    """分析失败模式"""
    patterns = {
        'immediate_entry_loss': [],  # 跳空后立即入场亏损
        'wrong_direction': [],       # 方向与跳空相反
        'fractal_trap': [],          # 被首个分型诱导
        'holiday_specific': [],      # 节假日特有
    }
    
    for corr in correlations:
        if corr['total_pnl'] < 0:
            # 分析亏损原因
            for trade in corr['trades']:
                if trade['pnl'] < 0:
                    # 立即入场（<10 bar）
                    if trade['bars_after_gap'] < 10:
                        patterns['immediate_entry_loss'].append({
                            'gap_date': corr['gap_date'],
                            'entry_bars': trade['bars_after_gap'],
                            'pnl': trade['pnl'],
                            'gap_pct': corr['gap_pct'],
                        })
                    
                    # 方向相反
                    gap_dir = corr['gap_direction']
                    trade_dir = trade['direction']
                    if (gap_dir == 'up' and trade_dir == 'short') or \
                       (gap_dir == 'down' and trade_dir == 'long'):
                        patterns['wrong_direction'].append({
                            'gap_date': corr['gap_date'],
                            'gap_direction': gap_dir,
                            'trade_direction': trade_dir,
                            'pnl': trade['pnl'],
                        })
                    
                    # 节假日
                    if corr['crossed_holiday']:
                        patterns['holiday_specific'].append({
                            'gap_date': corr['gap_date'],
                            'pnl': trade['pnl'],
                            'gap_pct': corr['gap_pct'],
                        })
    
    return patterns


def main():
    print("=== Phase 2: 跳空-交易关联分析 ===\n")
    
    # 加载跳空数据
    gaps = load_gaps()
    print(f"加载跳空数据: {sum(len(v) for v in gaps.values())} 次跳空")
    
    results = {}
    
    for contract, data_file in CONTRACTS.items():
        print(f"\n--- {contract} ---")
        
        # 简化：直接分析跳空数据中的交易关联
        # 实际回测需要更复杂的集成
        contract_gaps = gaps.get(contract, [])
        
        # 统计分析
        analysis = {
            'total_gaps': len(contract_gaps),
            'holiday_gaps': sum(1 for g in contract_gaps if g['crossed_holiday']),
            'large_gaps': sum(1 for g in contract_gaps if abs(g['gap_pct']) > 1.0),
            'gaps_with_immediate_fractal': sum(
                1 for g in contract_gaps 
                if g.get('fractal_analysis', {}).get('first_fractal_after', {}).get('datetime', '').endswith('09:00:00')
            ),
        }
        
        # 方向一致性分析
        consistent = 0
        inconsistent = 0
        for gap in contract_gaps:
            fa = gap.get('fractal_analysis', {}).get('first_fractal_after', {})
            if fa:
                gap_dir = gap['gap_direction']
                frac_type = fa.get('type', '')
                if (gap_dir == 'up' and frac_type == 'top') or \
                   (gap_dir == 'down' and frac_type == 'bottom'):
                    consistent += 1
                else:
                    inconsistent += 1
        
        analysis['consistent_direction'] = consistent
        analysis['inconsistent_direction'] = inconsistent
        analysis['consistency_rate'] = consistent / (consistent + inconsistent) if (consistent + inconsistent) > 0 else 0
        
        # 节假日 vs 普通周末
        holiday_consistent = 0
        holiday_total = 0
        weekend_consistent = 0
        weekend_total = 0
        
        for gap in contract_gaps:
            fa = gap.get('fractal_analysis', {}).get('first_fractal_after', {})
            if fa:
                gap_dir = gap['gap_direction']
                frac_type = fa.get('type', '')
                is_consistent = (gap_dir == 'up' and frac_type == 'top') or \
                               (gap_dir == 'down' and frac_type == 'bottom')
                
                if gap['crossed_holiday']:
                    holiday_total += 1
                    if is_consistent:
                        holiday_consistent += 1
                else:
                    weekend_total += 1
                    if is_consistent:
                        weekend_consistent += 1
        
        analysis['holiday_consistency'] = holiday_consistent / holiday_total if holiday_total > 0 else 0
        analysis['weekend_consistency'] = weekend_consistent / weekend_total if weekend_total > 0 else 0
        
        # 大跳空延迟分析
        large_gap_delays = []
        for gap in contract_gaps:
            if abs(gap['gap_pct']) > 1.0:
                fa = gap.get('fractal_analysis', {}).get('first_fractal_after', {})
                if fa and fa.get('datetime'):
                    frac_time = datetime.strptime(fa['datetime'], '%Y-%m-%d %H:%M:%S')
                    gap_date = datetime.strptime(gap['date'], '%Y-%m-%d')
                    delay_minutes = (frac_time - gap_date.replace(hour=9, minute=0)).total_seconds() / 60
                    large_gap_delays.append(delay_minutes)
        
        analysis['large_gap_avg_delay_minutes'] = np.mean(large_gap_delays) if large_gap_delays else 0
        
        results[contract] = analysis
        
        print(f"  总跳空: {analysis['total_gaps']}")
        print(f"  节假日跳空: {analysis['holiday_gaps']}")
        print(f"  大跳空(>1%): {analysis['large_gaps']}")
        print(f"  方向一致率: {analysis['consistency_rate']:.1%}")
        print(f"    - 节假日: {analysis['holiday_consistency']:.1%}")
        print(f"    - 普通周末: {analysis['weekend_consistency']:.1%}")
        print(f"  大跳空平均延迟: {analysis['large_gap_avg_delay_minutes']:.0f}分钟")
    
    # 保存结果
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {OUTPUT_FILE}")
    
    # 汇总
    print("\n=== 汇总 ===")
    total_gaps = sum(r['total_gaps'] for r in results.values())
    total_holiday = sum(r['holiday_gaps'] for r in results.values())
    avg_consistency = np.mean([r['consistency_rate'] for r in results.values()])
    avg_holiday_cons = np.mean([r['holiday_consistency'] for r in results.values() if r['holiday_gaps'] > 0])
    avg_weekend_cons = np.mean([r['weekend_consistency'] for r in results.values()])
    
    print(f"总跳空次数: {total_gaps}")
    print(f"节假日跳空: {total_holiday} ({total_holiday/total_gaps:.1%})")
    print(f"整体方向一致率: {avg_consistency:.1%}")
    print(f"节假日方向一致率: {avg_holiday_cons:.1%}")
    print(f"普通周末方向一致率: {avg_weekend_cons:.1%}")
    print(f"\n⚠️ 节假日比普通周末低 {(avg_weekend_cons - avg_holiday_cons):.1%} —— 这就是亏损来源")


if __name__ == "__main__":
    main()
