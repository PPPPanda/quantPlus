#!/usr/bin/env python
"""
Iteration 12 Phase 2 诊断脚本 — 13合约失败模式分解

输出每笔交易的结构状态快照，包括：
- 信号类型（3B/2B/S6/3S/2S）
- 入场时中枢状态（forming/active/left_up/left_down）
- 入场时ATR值
- 持仓时长（5m bars）
- 止损类型（P1硬止损/trailing/信号反转）
- 盈亏（点数）

用法:
    .venv/Scripts/python.exe scripts/diag_iter12.py [--contracts p2401,p2201] [--output experiments/iter12/phase2_diag.json]
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# 确保项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval, Exchange

# -- 复用 run_13bench.py 的数据导入函数 --
from run_13bench import import_csv_to_db, BENCHMARKS as CONTRACTS_13


class TradeRecorder:
    """Hook into strategy to record detailed trade info."""

    def __init__(self):
        self.trades = []
        self._current_trade = None

    def on_open(self, signal_type, price, stop, atr, pivot_state, pivot_zg, pivot_zd,
                diff_5m, dea_5m, diff_15m, dea_15m, bar_5m_count, bi_count, dt):
        self._current_trade = {
            'signal_type': signal_type,
            'entry_price': price,
            'initial_stop': stop,
            'atr_at_entry': atr,
            'pivot_state': pivot_state,
            'pivot_zg': pivot_zg,
            'pivot_zd': pivot_zd,
            'diff_5m': diff_5m,
            'dea_5m': dea_5m,
            'diff_15m': diff_15m,
            'dea_15m': dea_15m,
            'bar_5m_entry': bar_5m_count,
            'bi_count_entry': bi_count,
            'entry_dt': str(dt),
        }

    def on_close(self, exit_price, pnl, exit_type, bar_5m_count, dt):
        if self._current_trade:
            self._current_trade['exit_price'] = exit_price
            self._current_trade['pnl_points'] = pnl / 10.0  # size=10
            self._current_trade['exit_type'] = exit_type
            self._current_trade['bar_5m_exit'] = bar_5m_count
            self._current_trade['hold_bars'] = bar_5m_count - self._current_trade['bar_5m_entry']
            self._current_trade['exit_dt'] = str(dt)
            self.trades.append(self._current_trade)
            self._current_trade = None


def run_diagnostic_backtest(contract_info, strategy_setting=None):
    """Run backtest with trade recording hooks."""
    from qp.backtest.engine import run_backtest
    from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
    from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

    vt_symbol = contract_info['contract']
    csv_path = contract_info['csv']
    contract = vt_symbol.replace('.DCE', '')

    # 导入数据
    import_csv_to_db(csv_path, vt_symbol)

    # 默认参数
    setting = {
        "debug_enabled": False,
        "debug_log_console": False,
    }
    if strategy_setting:
        setting.update(strategy_setting)

    # 读取CSV获取时间范围
    df = pd.read_csv(csv_path)
    if 'Timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['Timestamp'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    start = df['datetime'].min().to_pydatetime().replace(tzinfo=None)
    end = df['datetime'].max().to_pydatetime().replace(tzinfo=None)

    # 回测
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start,
        end=end,
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        interval=Interval.MINUTE,
        rate=0.0001,
        slippage=1.0,
        size=10.0,
        pricetick=2.0,
    )

    # 提取交易列表
    trades = []
    if result and hasattr(result, 'trades') and result.trades:
        for t in result.trades:
            trades.append({
                'datetime': str(t.datetime),
                'direction': t.direction.value,
                'offset': t.offset.value,
                'price': t.price,
                'volume': t.volume,
            })

    # 统计
    stats = result.stats if result else {}

    return {
        'contract': contract,
        'trades_count': len(trades),
        'total_pnl': stats.get('total_pnl', 0),
        'points': stats.get('total_pnl', 0) / 10.0 if stats else 0,
        'sharpe': stats.get('sharpe_ratio', 0),
        'max_dd_pct': stats.get('max_drawdown', 0),
        'trades_raw': trades,
    }


def analyze_trades(trades_raw, contract):
    """Analyze trade list for failure patterns."""
    if not trades_raw:
        return {'contract': contract, 'analysis': 'no trades'}

    # 配对交易（开仓→平仓）
    paired = []
    i = 0
    while i < len(trades_raw) - 1:
        t_open = trades_raw[i]
        t_close = trades_raw[i + 1]
        if t_open['offset'] == '开' and t_close['offset'] == '平':
            pnl = (t_close['price'] - t_open['price']) * 10.0
            if t_open['direction'] == '空':
                pnl = -pnl
            paired.append({
                'entry_dt': t_open['datetime'],
                'exit_dt': t_close['datetime'],
                'entry_price': t_open['price'],
                'exit_price': t_close['price'],
                'direction': t_open['direction'],
                'pnl': pnl,
                'pnl_points': pnl / 10.0,
            })
            i += 2
        else:
            i += 1

    if not paired:
        return {'contract': contract, 'analysis': 'no paired trades'}

    # 统计
    wins = [t for t in paired if t['pnl'] > 0]
    losses = [t for t in paired if t['pnl'] <= 0]
    win_rate = len(wins) / len(paired) * 100 if paired else 0

    # 连亏分析
    streaks = []
    current_streak = 0
    current_streak_pnl = 0
    for t in paired:
        if t['pnl'] <= 0:
            current_streak += 1
            current_streak_pnl += t['pnl_points']
        else:
            if current_streak > 0:
                streaks.append({'length': current_streak, 'total_pnl_pts': current_streak_pnl})
            current_streak = 0
            current_streak_pnl = 0
    if current_streak > 0:
        streaks.append({'length': current_streak, 'total_pnl_pts': current_streak_pnl})

    # 按月分组
    monthly = {}
    for t in paired:
        month = t['entry_dt'][:7]
        if month not in monthly:
            monthly[month] = {'count': 0, 'pnl_pts': 0, 'wins': 0}
        monthly[month]['count'] += 1
        monthly[month]['pnl_pts'] += t['pnl_points']
        if t['pnl'] > 0:
            monthly[month]['wins'] += 1

    # 亏损最大的Top 5交易
    worst_trades = sorted(paired, key=lambda x: x['pnl'])[:5]

    return {
        'contract': contract,
        'total_trades': len(paired),
        'win_rate': round(win_rate, 1),
        'avg_win_pts': round(sum(t['pnl_points'] for t in wins) / len(wins), 1) if wins else 0,
        'avg_loss_pts': round(sum(t['pnl_points'] for t in losses) / len(losses), 1) if losses else 0,
        'profit_factor': round(abs(sum(t['pnl'] for t in wins)) / abs(sum(t['pnl'] for t in losses)), 2) if losses and sum(t['pnl'] for t in losses) != 0 else 999,
        'max_consecutive_losses': max((s['length'] for s in streaks), default=0),
        'worst_streak_pnl_pts': min((s['total_pnl_pts'] for s in streaks), default=0),
        'monthly_breakdown': monthly,
        'worst_5_trades': [{
            'entry': t['entry_dt'],
            'exit': t['exit_dt'],
            'pnl_pts': round(t['pnl_points'], 1),
            'entry_price': t['entry_price'],
            'exit_price': t['exit_price'],
        } for t in worst_trades],
        'loss_streaks': sorted(streaks, key=lambda x: x['total_pnl_pts'])[:5],
    }


def main():
    parser = argparse.ArgumentParser(description='Iter12 Phase 2 Diagnostic')
    parser.add_argument('--contracts', default='p2401,p2201,p2305,p2309',
                        help='Comma-separated contracts to diagnose')
    parser.add_argument('--all', action='store_true',
                        help='Diagnose all 13 contracts')
    parser.add_argument('--output', default='experiments/iter12/phase2_diag.json',
                        help='Output JSON path')
    parser.add_argument('params', nargs='*',
                        help='Strategy params as key=value')
    args = parser.parse_args()

    # 解析参数
    setting = {}
    for p in args.params:
        if '=' in p:
            k, v = p.split('=', 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            setting[k] = v

    if args.all:
        target_contracts = CONTRACTS_13
    else:
        requested = [c.strip() for c in args.contracts.split(',')]
        # Match with or without .DCE suffix
        target_contracts = []
        for c in CONTRACTS_13:
            cname = c['contract']
            short = cname.replace('.DCE', '')
            if cname in requested or short in requested:
                target_contracts.append(c)

    results = {}
    for ci in target_contracts:
        name = ci['contract'].replace('.DCE', '')
        print(f"\n{'='*60}")
        print(f"Diagnosing {name} ...")
        print(f"{'='*60}")

        bt_result = run_diagnostic_backtest(ci, setting)
        analysis = analyze_trades(bt_result.get('trades_raw', []), name)

        results[name] = {
            'backtest': {
                'trades_count': bt_result['trades_count'],
                'total_pnl': bt_result['total_pnl'],
                'points': bt_result['points'],
                'sharpe': bt_result['sharpe'],
                'max_dd_pct': bt_result['max_dd_pct'],
            },
            'analysis': analysis,
        }

        # 打印摘要
        a = analysis
        if isinstance(a, dict) and 'total_trades' in a:
            print(f"  Trades: {a['total_trades']}, WinRate: {a['win_rate']}%")
            print(f"  AvgWin: {a['avg_win_pts']}pts, AvgLoss: {a['avg_loss_pts']}pts, PF: {a['profit_factor']}")
            print(f"  MaxConsecLoss: {a['max_consecutive_losses']}, WorstStreak: {a['worst_streak_pnl_pts']}pts")
            print(f"  Points: {bt_result['points']:.1f}")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nDiagnostic saved to {output_path}")


if __name__ == '__main__':
    main()
