#!/usr/bin/env python3
"""
跳空分析脚本 - 分析全量 Wind 数据 + p2601 的跳空对策略的影响

分析目标：
1. 识别所有跳空事件（session 间隔导致的 gap）
2. 统计跳空时策略是否持仓
3. 分析跳空导致的盈亏
4. 区分跳空带来的盈利和亏损
"""

import os
import sys
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vnpy_ctabacktester import BacktestingEngine
from vnpy.trader.constant import Interval
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import PALM_OIL_SESSIONS


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
    
    # 计算 ATR（14 周期）
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14 * 5).mean()  # 5分钟合成，所以乘5
    
    # 检测 session 变化（日内交易中断）
    df['prev_close'] = df['close'].shift(1)
    df['prev_datetime'] = df['datetime'].shift(1)
    df['time_gap'] = (df['datetime'] - df['prev_datetime']).dt.total_seconds() / 60
    
    # 识别 session 跳空（时间间隔 > 5 分钟）
    session_gap_mask = df['time_gap'] > 5
    
    # 计算跳空幅度
    df['gap'] = df['open'] - df['prev_close']
    df['gap_abs'] = df['gap'].abs()
    df['gap_atr'] = df['gap_abs'] / df['atr']
    
    # 筛选有意义的跳空（session 间隔 + gap > 0.5 ATR）
    gap_events = df[session_gap_mask & (df['gap_atr'] > 0.5)].copy()
    
    # 标记跳空类型
    gap_events['gap_type'] = np.where(gap_events['gap'] > 0, 'gap_up', 'gap_down')
    gap_events['is_extreme'] = gap_events['gap_atr'] > atr_mult
    
    return gap_events[['datetime', 'prev_datetime', 'prev_close', 'open', 'close', 
                       'gap', 'gap_abs', 'gap_atr', 'atr', 'gap_type', 'is_extreme', 'time_gap']]


def run_backtest_with_trade_log(data_file: str, contract: str) -> tuple:
    """
    运行回测并返回交易记录
    
    Returns:
        (trades_df, daily_results, gap_events)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {contract}")
    print(f"{'='*60}")
    
    # 加载数据
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 检测跳空
    gap_events = detect_gaps(df)
    print(f"  Gap events detected: {len(gap_events)}")
    print(f"  Extreme gaps (>3 ATR): {gap_events['is_extreme'].sum()}")
    
    # 设置回测引擎
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=f"{contract}.DCE",
        interval=Interval.MINUTE,
        start=df['datetime'].min(),
        end=df['datetime'].max(),
        rate=0.0001,
        slippage=2,
        size=10,
        pricetick=2,
        capital=1000000,
    )
    
    # 加载策略（使用 iter14 基线参数）
    engine.add_strategy(CtaChanPivotStrategy, {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_window": 14,
        "atr_trailing_mult": 3.0,
        "atr_activate_mult": 2.5,
        "atr_entry_filter": 2.0,
        "min_bi_gap": 4,
        "pivot_valid_range": 6,
        "fixed_volume": 1,
        "cooldown_losses": 2,
        "cooldown_bars": 20,
        "circuit_breaker_losses": 7,
        "circuit_breaker_bars": 70,
        "lock_profit_atr": 0.0,
        "min_hold_bars": 2,
        "max_pullback_atr": 3.2,
        "use_bi_trailing": True,
        "stop_buffer_atr_pct": 0.02,
        "max_pivot_entries": 2,
        "pivot_reentry_atr": 0.6,
        "dedup_bars": 0,
        "dedup_atr_mult": 1.5,
        "div_mode": 1,
        "div_threshold": 0.39,
        "seg_enabled": False,
        "hist_gate": 0,
        "gap_extreme_atr": 0.0,  # 禁用 S26 以观察原始跳空影响
        "gap_cooldown_bars": 0,
        "debug": False,
        "debug_enabled": False,
    })
    
    # 加载数据到引擎
    engine.load_data()
    
    # 运行回测
    engine.run_backtesting()
    
    # 获取交易记录
    trades = engine.get_all_trades()
    daily = engine.get_all_daily_results()
    
    # 转换为 DataFrame
    trades_df = pd.DataFrame([{
        'datetime': t.datetime,
        'direction': t.direction.value,
        'offset': t.offset.value,
        'price': t.price,
        'volume': t.volume,
    } for t in trades])
    
    return trades_df, daily, gap_events, engine


def analyze_gap_impact(trades_df: pd.DataFrame, gap_events: pd.DataFrame, 
                       contract: str) -> dict:
    """
    分析跳空对交易的影响
    
    Returns:
        分析结果字典
    """
    if trades_df.empty:
        return {
            'contract': contract,
            'total_trades': 0,
            'gap_affected_trades': 0,
            'gap_profit': 0,
            'gap_loss': 0,
            'details': []
        }
    
    trades_df = trades_df.copy()
    trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
    
    # 重建持仓状态
    positions = []
    current_pos = 0
    entry_price = 0
    entry_time = None
    
    for _, trade in trades_df.iterrows():
        if trade['offset'] == '开':
            current_pos = 1 if trade['direction'] == '多' else -1
            entry_price = trade['price']
            entry_time = trade['datetime']
        else:  # 平仓
            exit_price = trade['price']
            exit_time = trade['datetime']
            if current_pos == 1:
                pnl = (exit_price - entry_price) * 10  # 棕榈油 10 元/点
            else:
                pnl = (entry_price - exit_price) * 10
            
            positions.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': '多' if current_pos == 1 else '空',
                'pnl': pnl,
            })
            current_pos = 0
    
    positions_df = pd.DataFrame(positions)
    
    if positions_df.empty:
        return {
            'contract': contract,
            'total_trades': 0,
            'gap_affected_trades': 0,
            'gap_profit': 0,
            'gap_loss': 0,
            'details': []
        }
    
    # 分析每个持仓是否受跳空影响
    gap_details = []
    total_gap_profit = 0
    total_gap_loss = 0
    
    for _, pos in positions_df.iterrows():
        # 检查持仓期间是否有跳空
        gaps_in_position = gap_events[
            (gap_events['datetime'] >= pos['entry_time']) &
            (gap_events['datetime'] <= pos['exit_time'])
        ]
        
        if not gaps_in_position.empty:
            # 计算跳空对该笔交易的影响
            for _, gap in gaps_in_position.iterrows():
                gap_impact = gap['gap'] * (1 if pos['direction'] == '多' else -1) * 10
                
                detail = {
                    'entry_time': str(pos['entry_time']),
                    'exit_time': str(pos['exit_time']),
                    'gap_time': str(gap['datetime']),
                    'direction': pos['direction'],
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['exit_price'],
                    'trade_pnl': pos['pnl'],
                    'gap': gap['gap'],
                    'gap_atr': gap['gap_atr'],
                    'gap_type': gap['gap_type'],
                    'is_extreme': gap['is_extreme'],
                    'gap_impact': gap_impact,
                    'time_gap_minutes': gap['time_gap'],
                }
                gap_details.append(detail)
                
                if gap_impact > 0:
                    total_gap_profit += gap_impact
                else:
                    total_gap_loss += gap_impact
    
    return {
        'contract': contract,
        'total_trades': len(positions_df),
        'gap_affected_trades': len(set([d['entry_time'] for d in gap_details])),
        'gap_profit': total_gap_profit,
        'gap_loss': total_gap_loss,
        'net_gap_impact': total_gap_profit + total_gap_loss,
        'details': gap_details
    }


def main():
    """主函数"""
    # 数据路径
    wind_dir = PROJECT_ROOT / "data" / "analyse" / "wind"
    xt_dir = PROJECT_ROOT / "data" / "analyse"
    
    # 全量 Wind 合约
    wind_contracts = [
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
        wind_contracts.append(("p2601", p2601_file))
    
    all_results = []
    all_gap_events = []
    
    for contract, data_file in wind_contracts:
        if not data_file.exists():
            print(f"Skipping {contract}: file not found")
            continue
        
        try:
            trades_df, daily, gap_events, engine = run_backtest_with_trade_log(
                str(data_file), contract
            )
            
            # 分析跳空影响
            result = analyze_gap_impact(trades_df, gap_events, contract)
            all_results.append(result)
            
            # 保存跳空事件
            gap_events['contract'] = contract
            all_gap_events.append(gap_events)
            
            # 计算总收益
            if daily:
                total_pnl = sum([d.net_pnl for d in daily.values()])
            else:
                total_pnl = 0
            
            print(f"\n  Total PnL: {total_pnl:.0f}")
            print(f"  Total trades: {result['total_trades']}")
            print(f"  Gap-affected trades: {result['gap_affected_trades']}")
            print(f"  Gap profit: {result['gap_profit']:.0f}")
            print(f"  Gap loss: {result['gap_loss']:.0f}")
            print(f"  Net gap impact: {result['net_gap_impact']:.0f}")
            
        except Exception as e:
            print(f"Error processing {contract}: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总统计
    print("\n" + "="*80)
    print("SUMMARY: Gap Impact Analysis")
    print("="*80)
    
    total_gap_profit = sum(r['gap_profit'] for r in all_results)
    total_gap_loss = sum(r['gap_loss'] for r in all_results)
    total_gap_affected = sum(r['gap_affected_trades'] for r in all_results)
    
    print(f"\nTotal gap-affected trades: {total_gap_affected}")
    print(f"Total gap profit: {total_gap_profit:.0f}")
    print(f"Total gap loss: {total_gap_loss:.0f}")
    print(f"Net gap impact: {total_gap_profit + total_gap_loss:.0f}")
    
    # 按合约输出详细跳空事件
    print("\n" + "="*80)
    print("DETAILED GAP EVENTS BY CONTRACT")
    print("="*80)
    
    for result in all_results:
        if result['details']:
            print(f"\n### {result['contract']} ###")
            for d in result['details']:
                emoji = "📈" if d['gap_impact'] > 0 else "📉"
                extreme_flag = "⚠️EXTREME" if d['is_extreme'] else ""
                print(f"  {emoji} Gap @ {d['gap_time']}")
                print(f"     Direction: {d['direction']}, Gap: {d['gap']:.0f} pts ({d['gap_atr']:.1f}×ATR) {extreme_flag}")
                print(f"     Trade PnL: {d['trade_pnl']:.0f}, Gap Impact: {d['gap_impact']:.0f}")
                print(f"     Time gap: {d['time_gap_minutes']:.0f} minutes")
    
    # 保存结果到 JSON
    output_file = PROJECT_ROOT / "experiments" / "gap_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_gap_profit': total_gap_profit,
                'total_gap_loss': total_gap_loss,
                'net_gap_impact': total_gap_profit + total_gap_loss,
                'total_gap_affected_trades': total_gap_affected,
            },
            'by_contract': all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    main()
