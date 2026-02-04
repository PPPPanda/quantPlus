#!/usr/bin/env python3
"""
iter7 Phase 1: 7合约统计特征审计脚本.

输出：缠论结构指标、信号分布、交易聚类、失败模式分析

用法:
    cd quantPlus
    .venv/Scripts/python.exe scripts/diag_iter7.py [contract1] [contract2] ...
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from vnpy.trader.constant import Interval

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from vnpy_ctastrategy.backtesting import BacktestingEngine
from qp.backtest.engine import run_backtest, BacktestResult
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

# ---- Reuse run_7bench's import logic ----
from run_7bench import import_csv_to_db, BENCHMARKS, BT_PARAMS


def run_backtest_with_strategy(vt_symbol, start, end, strategy_class, strategy_setting, **bt_params):
    """Run backtest and return (stats, strategy_instance)."""
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=bt_params.get('interval', Interval.MINUTE),
        start=start, end=end,
        rate=bt_params.get('rate', 0.0001),
        slippage=bt_params.get('slippage', 1.0),
        size=bt_params.get('size', 10.0),
        pricetick=bt_params.get('pricetick', 2.0),
        capital=bt_params.get('capital', 1_000_000.0),
    )
    engine.add_strategy(strategy_class, strategy_setting or {})
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()
    return stats, engine.strategy

SETTINGS = {
    "debug_enabled": False,
    "debug_log_console": False,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "cooldown_losses": 2,
    "cooldown_bars": 20,
    "circuit_breaker_losses": 6,
    "circuit_breaker_bars": 60,
    "atr_entry_filter": 2.0,
}


class InstrumentedStrategy(CtaChanPivotStrategy):
    """Monkey-patched strategy to capture detailed trade & chan state info."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trade_log = []
        self._bi_lengths = []
        self._bi_price_ranges = []
        self._bi_directions = []   # 'up' or 'down'
        self._pivot_log = []
        self._signal_log = []
        self._atr_samples = []
        self._bar5m_prices = []
        self._last_signal_type = ""
        self._current_trade = None

    def _on_5m_bar(self, bar):
        super()._on_5m_bar(bar)
        if self.atr > 0:
            self._atr_samples.append(self.atr)
        self._bar5m_prices.append(bar['close'])

    def _process_bi(self):
        n_before = len(self._bi_points)
        result = super()._process_bi()
        if result and len(self._bi_points) >= 2:
            bp = self._bi_points
            idx_diff = bp[-1]['idx'] - bp[-2]['idx']
            price_range = abs(bp[-1]['price'] - bp[-2]['price'])
            self._bi_lengths.append(idx_diff)
            self._bi_price_ranges.append(price_range)
            # direction: top→bottom = down, bottom→top = up
            if bp[-2]['type'] == 'top' and bp[-1]['type'] == 'bottom':
                self._bi_directions.append('down')
            else:
                self._bi_directions.append('up')
        return result

    def _update_pivots(self):
        ap_before_id = id(self._active_pivot) if self._active_pivot else None
        ap_before_start = self._active_pivot.get('start_bi_idx') if self._active_pivot else None
        super()._update_pivots()
        ap = self._active_pivot
        if ap and ap.get('start_bi_idx') != ap_before_start:
            self._pivot_log.append({
                'zg': ap['zg'], 'zd': ap['zd'],
                'width': ap['zg'] - ap['zd'],
                'start_bi': ap['start_bi_idx'],
                'end_bi': ap['end_bi_idx'],
                'state': ap['state'],
            })

    def _check_signal(self, curr_bar, new_bi):
        ps_before = id(self._pending_signal) if self._pending_signal else None
        super()._check_signal(curr_bar, new_bi)
        if self._pending_signal and id(self._pending_signal) != ps_before:
            self._last_signal_type = self._signal_type
            sig = self._pending_signal
            self._signal_log.append({
                'type': sig['type'],
                'signal_type': self._signal_type,
                'trigger_price': sig['trigger_price'],
                'stop_base': sig['stop_base'],
                'bar_5m_count': self._bar_5m_count,
                'atr': self.atr,
                'diff_15m': self._prev_diff_15m,
                'dea_15m': self._prev_dea_15m,
                'pivot_state': self._active_pivot['state'] if self._active_pivot else 'none',
            })

    def _open_position(self, direction, price, stop_base):
        self._current_trade = {
            'signal_type': self._last_signal_type,
            'entry_price': price,
            'entry_bar5m': self._bar_5m_count,
            'stop_base': stop_base,
            'atr_at_entry': self.atr,
            'diff_15m': self._prev_diff_15m,
            'dea_15m': self._prev_dea_15m,
            'pivot_state': self._active_pivot['state'] if self._active_pivot else 'none',
            'pivot_zg': self._active_pivot['zg'] if self._active_pivot else None,
            'pivot_zd': self._active_pivot['zd'] if self._active_pivot else None,
            'n_bi': len(self._bi_points),
            'n_pivots': len(self._pivots),
            'trailing_activated': False,
        }
        super()._open_position(direction, price, stop_base)

    def _check_stop_loss_1m(self, bar):
        pos_before = self._position
        trailing_before = self._trailing_active
        result = super()._check_stop_loss_1m(bar)
        if result and pos_before != 0 and self._current_trade:
            reason = 'trailing_stop' if trailing_before else 'hard_stop'
            if pos_before == 1:
                exit_price = bar['open'] if bar['open'] < self._stop_price else self._stop_price
            else:
                exit_price = bar['open'] if bar['open'] > self._stop_price else self._stop_price
            self._close_trade(exit_price, reason)
        return result

    def _check_entry_1m(self, bar):
        pos_before = self._position
        sig_type = self._pending_signal.get('type', '') if self._pending_signal else ''
        trigger_price = self._pending_signal.get('trigger_price', 0) if self._pending_signal else 0
        super()._check_entry_1m(bar)
        if sig_type == 'CloseLong' and pos_before == 1 and self._position == 0:
            fill_price = min(trigger_price, bar['open']) if trigger_price else bar['close']
            if self._current_trade:
                self._close_trade(fill_price, 'signal_close')

    def _update_trailing_stop(self, bar):
        trailing_before = self._trailing_active
        super()._update_trailing_stop(bar)
        if not trailing_before and self._trailing_active and self._current_trade:
            self._current_trade['trailing_activated'] = True

    def _close_trade(self, exit_price, reason):
        if not self._current_trade:
            return
        t = self._current_trade
        pnl = exit_price - t['entry_price']
        hold_bars = self._bar_5m_count - t['entry_bar5m']
        initial_risk = abs(t['entry_price'] - t['stop_base'])
        if initial_risk <= 0:
            initial_risk = t.get('atr_at_entry', 1)
        t.update({
            'exit_price': exit_price,
            'exit_bar5m': self._bar_5m_count,
            'exit_reason': reason,
            'pnl': pnl,
            'pnl_r': pnl / initial_risk if initial_risk > 0 else 0,
            'hold_bars': hold_bars,
        })
        self._trade_log.append(t)
        self._current_trade = None


def analyze_contract(bench, settings):
    """Full analysis for one contract."""
    vt_symbol = bench["contract"]
    csv_path = bench["csv"]
    name = vt_symbol.split('.')[0]

    print(f"\n{'='*70}")
    print(f"  {name} — 统计特征审计")
    print(f"{'='*70}")

    # Import data
    start, end, bar_count = import_csv_to_db(csv_path, vt_symbol)

    # Run instrumented backtest
    stats, strat = run_backtest_with_strategy(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=InstrumentedStrategy,
        strategy_setting=settings,
        **BT_PARAMS,
    )
    stats = stats or {}

    # === 1. 基本指标 ===
    pnl = stats.get('total_net_pnl', 0)
    print(f"\n--- 基本回测指标 ---")
    print(f"  PnL(金额): {pnl:.0f}  PnL(点数): {pnl/10:.1f}")
    print(f"  Ret%: {stats.get('total_return', 0):.2f}%  Sharpe: {stats.get('sharpe_ratio', 0):.2f}")
    print(f"  MaxDD%: {stats.get('max_ddpercent', 0):.2f}%  Trades: {stats.get('total_trade_count', 0)}")
    print(f"  Commission: {stats.get('total_commission', 0):.0f}  Slippage: {stats.get('total_slippage', 0):.0f}")

    # === 2. ATR 统计 ===
    atr_arr = np.array(strat._atr_samples) if strat._atr_samples else np.array([0])
    prices = np.array(strat._bar5m_prices) if strat._bar5m_prices else np.array([1])
    atr_pct = (atr_arr[:len(prices)] / prices[:len(atr_arr)]) * 100 if len(atr_arr) > 0 else np.array([0])
    print(f"\n--- ATR 统计 ---")
    print(f"  Mean: {atr_arr.mean():.1f}  Std: {atr_arr.std():.1f}  Min/Max: {atr_arr.min():.1f}/{atr_arr.max():.1f}")
    print(f"  P25/P50/P75: {np.percentile(atr_arr,25):.1f}/{np.percentile(atr_arr,50):.1f}/{np.percentile(atr_arr,75):.1f}")
    print(f"  ATR/Price%: mean={atr_pct.mean():.3f}%, max={atr_pct.max():.3f}%")

    # === 3. 缠论结构指标 ===
    print(f"\n--- 缠论结构 ---")
    print(f"  包含处理后K线数: {len(strat._k_lines)}")
    print(f"  笔端点数: {len(strat._bi_points)}")
    print(f"  中枢数: {len(strat._pivot_log)} (归档{len(strat._pivots)}+活跃{'1' if strat._active_pivot else '0'})")

    bi_lens = np.array(strat._bi_lengths) if strat._bi_lengths else np.array([0])
    bi_ranges = np.array(strat._bi_price_ranges) if strat._bi_price_ranges else np.array([0])
    print(f"  笔长度(K线): mean={bi_lens.mean():.1f} med={np.median(bi_lens):.0f} min={bi_lens.min()} max={bi_lens.max()}")
    print(f"  笔幅度(点): mean={bi_ranges.mean():.0f} med={np.median(bi_ranges):.0f} min={bi_ranges.min():.0f} max={bi_ranges.max():.0f}")

    # 笔方向统计
    dir_counts = Counter(strat._bi_directions)
    print(f"  笔方向: up={dir_counts.get('up',0)} down={dir_counts.get('down',0)}")

    # 中枢统计
    if strat._pivot_log:
        pwidths = [p['width'] for p in strat._pivot_log]
        pdurations = [p['end_bi'] - p['start_bi'] for p in strat._pivot_log]
        print(f"  中枢宽度(ZG-ZD): mean={np.mean(pwidths):.0f} med={np.median(pwidths):.0f} min={min(pwidths):.0f} max={max(pwidths):.0f}")
        print(f"  中枢持续(笔数): mean={np.mean(pdurations):.1f} med={np.median(pdurations):.0f}")
        # 信枢比
        bi_pivot_ratio = len(strat._bi_points) / len(strat._pivot_log) if strat._pivot_log else 0
        print(f"  信枢比(笔/中枢): {bi_pivot_ratio:.1f}")

    # === 4. 信号统计 ===
    print(f"\n--- 信号统计 ---")
    sig_types = Counter(s['signal_type'] for s in strat._signal_log)
    sig_actions = Counter(s['type'] for s in strat._signal_log)
    pivot_states = Counter(s['pivot_state'] for s in strat._signal_log)
    print(f"  总信号数: {len(strat._signal_log)}")
    print(f"  信号类型: {dict(sig_types)}")
    print(f"  动作类型: {dict(sig_actions)}")
    print(f"  中枢状态: {dict(pivot_states)}")

    # === 5. 交易详细分析 ===
    trades = strat._trade_log
    print(f"\n--- 交易详细分析 ({len(trades)} instrumented trades) ---")

    trade_summary = {}
    if trades:
        # 按信号类型
        by_sig = defaultdict(list)
        for t in trades:
            by_sig[t['signal_type']].append(t)

        print(f"\n  按信号类型:")
        for sig_t in sorted(by_sig.keys()):
            trs = by_sig[sig_t]
            pnls = [t['pnl'] for t in trs]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            wr = len(wins) / len(pnls) * 100 if pnls else 0
            avg_w = np.mean(wins) if wins else 0
            avg_l = np.mean(losses) if losses else 0
            pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
            holds = [t['hold_bars'] for t in trs]
            print(f"    {sig_t}: n={len(trs)} WR={wr:.1f}% avgW={avg_w:.0f} avgL={avg_l:.0f} PF={pf:.2f} sum={sum(pnls):.0f} avgHold={np.mean(holds):.1f}")
            trade_summary[sig_t] = {'n': len(trs), 'pnl': sum(pnls), 'wr': wr, 'pf': pf}

        # 按退出原因
        by_exit = defaultdict(list)
        for t in trades:
            by_exit[t.get('exit_reason', 'unknown')].append(t)

        print(f"\n  按退出原因:")
        for reason in sorted(by_exit.keys()):
            trs = by_exit[reason]
            pnls = [t['pnl'] for t in trs]
            print(f"    {reason}: n={len(trs)} sum={sum(pnls):.0f} avg={np.mean(pnls):.0f}")

        # 持仓时长分桶
        holds = [t['hold_bars'] for t in trades]
        print(f"\n  持仓时长(5m bars): mean={np.mean(holds):.1f} med={np.median(holds):.0f}")
        buckets = {'<=2': [], '3-5': [], '6-10': [], '11-20': [], '21-50': [], '>50': []}
        for t in trades:
            h = t['hold_bars']
            if h <= 2: k = '<=2'
            elif h <= 5: k = '3-5'
            elif h <= 10: k = '6-10'
            elif h <= 20: k = '11-20'
            elif h <= 50: k = '21-50'
            else: k = '>50'
            buckets[k].append(t['pnl'])
        for k, pnls in buckets.items():
            if pnls:
                print(f"    {k}: n={len(pnls)} sum={sum(pnls):.0f} avg={np.mean(pnls):.0f}")

        # 连亏序列
        pnl_seq = [t['pnl'] for t in trades]
        streaks = []
        curr = 0
        for p in pnl_seq:
            if p <= 0:
                curr += 1
            else:
                if curr > 0:
                    streaks.append(curr)
                curr = 0
        if curr > 0:
            streaks.append(curr)
        mean_streak = f"{np.mean(streaks):.1f}" if streaks else "0"
        print(f"\n  连亏: max={max(streaks) if streaks else 0} mean={mean_streak} count(>=4)={sum(1 for s in streaks if s >= 4)}")

        # R倍数
        r_mults = [t.get('pnl_r', 0) for t in trades]
        r_arr = np.array(r_mults)
        print(f"  R倍数: mean={r_arr.mean():.2f} med={np.median(r_arr):.2f} P10={np.percentile(r_arr,10):.2f} P90={np.percentile(r_arr,90):.2f}")
        print(f"    大赢(>3R): {sum(1 for r in r_arr if r > 3)}  大亏(<-1.5R): {sum(1 for r in r_arr if r < -1.5)}")

        # trailing stop 使用率
        trailing_used = sum(1 for t in trades if t.get('trailing_activated'))
        print(f"  Trailing激活率: {trailing_used}/{len(trades)} ({trailing_used/len(trades)*100:.1f}%)")

    return {
        'name': name,
        'pnl': pnl,
        'pts': pnl / 10,
        'sharpe': stats.get('sharpe_ratio', 0),
        'trades_count': stats.get('total_trade_count', 0),
        'commission': stats.get('total_commission', 0),
        'atr_mean': float(atr_arr.mean()),
        'atr_pct_mean': float(atr_pct.mean()),
        'bi_count': len(strat._bi_points),
        'pivot_count': len(strat._pivot_log),
        'bi_len_mean': float(bi_lens.mean()),
        'bi_range_mean': float(bi_ranges.mean()),
        'pivot_width_mean': float(np.mean([p['width'] for p in strat._pivot_log])) if strat._pivot_log else 0,
        'trade_summary': trade_summary,
        'instrumented_trades': len(trades),
    }


def cross_contract_summary(results):
    """跨合约对比总结."""
    print(f"\n\n{'='*70}")
    print(f"  跨合约对比总结")
    print(f"{'='*70}")

    print(f"\n{'合约':<8} {'PnL(pts)':>9} {'Sharpe':>7} {'Trades':>7} {'ATR均':>7} {'ATR/P%':>7} {'笔数':>5} {'中枢':>5} {'笔均长':>6} {'中枢宽':>7}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:<8} {r['pts']:>9.1f} {r['sharpe']:>7.2f} {r['trades_count']:>7} "
              f"{r['atr_mean']:>7.0f} {r['atr_pct_mean']:>6.3f}% {r['bi_count']:>5} "
              f"{r['pivot_count']:>5} {r['bi_len_mean']:>6.1f} {r['pivot_width_mean']:>7.0f}")

    # 波动率 vs 收益
    print(f"\n  波动率(ATR/Price%) vs PnL:")
    for r in sorted(results, key=lambda x: x['atr_pct_mean']):
        print(f"    {r['name']}: vol={r['atr_pct_mean']:.3f}% → pnl={r['pts']:.1f}")

    # 信号类型跨合约
    all_sig_types = set()
    for r in results:
        all_sig_types.update(r['trade_summary'].keys())
    print(f"\n  信号类型跨合约对比:")
    for sig_t in sorted(all_sig_types):
        parts = []
        for r in results:
            s = r['trade_summary'].get(sig_t, {'n': 0, 'pnl': 0, 'wr': 0})
            parts.append(f"{r['name']}:{s['n']}笔/{s['pnl']:.0f}")
        print(f"    {sig_t}: {' | '.join(parts)}")


def main():
    filter_contracts = []
    for arg in sys.argv[1:]:
        filter_contracts.append(arg)

    results = []
    for bench in BENCHMARKS:
        name = bench["contract"].split('.')[0]
        if filter_contracts and name not in filter_contracts:
            continue
        t0 = time.time()
        r = analyze_contract(bench, SETTINGS)
        elapsed = time.time() - t0
        print(f"  [{name} 完成: {elapsed:.1f}s]")
        results.append(r)

    if len(results) > 1:
        cross_contract_summary(results)

    # 保存JSON
    out_path = ROOT / "experiments" / "iter7" / "phase1_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
