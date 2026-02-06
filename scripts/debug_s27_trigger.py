#!/usr/bin/env python3
"""
Debug S27 trigger - check if strategy actually triggers S27.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from vnpy.trader.constant import Interval

import logging
# Enable strategy logging to see S27 messages
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger("vnpy").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

bench = {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"}

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

# Enable S27 with debug
SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "cooldown_losses": 2,
    "cooldown_bars": 20,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "atr_entry_filter": 2.0,
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "max_pullback_atr": 3.2,
    "gap_threshold_atr": 1.5,
    "atr_boost_factor": 1.5,
    "atr_boost_bars": 6,
}

# Patch strategy to capture S27 logs
original_write_log = CtaChanPivotStrategy.write_log
s27_logs = []

def patched_write_log(self, msg: str):
    if "S27" in msg or "gap" in msg.lower():
        s27_logs.append(msg)
        print(f"[S27] {msg}")
    original_write_log(self, msg)

CtaChanPivotStrategy.write_log = patched_write_log

def main():
    df = pd.read_csv(bench['csv'])
    df_norm = normalize_1m_bars(df, PALM_OIL_SESSIONS)

    start = pd.Timestamp(df_norm['datetime'].min()).to_pydatetime()
    end = pd.Timestamp(df_norm['datetime'].max()).to_pydatetime()

    print(f"Running backtest for p2209...")
    print(f"Data range: {start} to {end}")
    print("-" * 60)

    result = run_backtest(
        vt_symbol=bench['contract'],
        interval=BT_PARAMS['interval'],
        start=start, end=end,
        rate=BT_PARAMS['rate'],
        slippage=BT_PARAMS['slippage'],
        size=BT_PARAMS['size'],
        pricetick=BT_PARAMS['pricetick'],
        capital=BT_PARAMS['capital'],
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=SETTING,
    )

    print("-" * 60)
    print(f"\nTotal S27 related logs: {len(s27_logs)}")

    if s27_logs:
        print("\nS27 triggers:")
        for log in s27_logs:
            print(f"  {log}")

    stats = result.stats if hasattr(result, 'stats') else result
    total_net = stats.get('total_net_pnl', stats.get('total_pnl', 0))
    print(f"\nTotal PnL: {total_net / 10:.1f} pts")


if __name__ == '__main__':
    main()
