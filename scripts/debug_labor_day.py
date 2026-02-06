#!/usr/bin/env python3
"""
Debug Labor Day 2022 disaster trades in p2209.
Check if the losing trades were opened before or after the holiday gap.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from vnpy.trader.constant import Interval

import logging
logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

# p2209 config
bench = {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"}

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

SETTING = {
    "debug_enabled": True,
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


def main():
    df = pd.read_csv(bench['csv'])
    df_norm = normalize_1m_bars(df, PALM_OIL_SESSIONS)

    start = pd.Timestamp(df_norm['datetime'].min()).to_pydatetime()
    end = pd.Timestamp(df_norm['datetime'].max()).to_pydatetime()

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

    # Get trades around Labor Day (2022-05-05)
    trades = result.trades if hasattr(result, 'trades') else []

    print("=" * 60)
    print("Trades around Labor Day 2022 (May 1-7)")
    print("=" * 60)

    # Filter trades near Labor Day
    labor_day_trades = []
    for t in trades:
        t_date = t.datetime.date() if hasattr(t.datetime, 'date') else t.datetime
        if isinstance(t_date, datetime):
            t_date = t_date.date()
        if str(t_date) >= '2022-04-28' and str(t_date) <= '2022-05-10':
            labor_day_trades.append(t)

    if labor_day_trades:
        for t in labor_day_trades:
            direction = "BUY" if hasattr(t, 'direction') and "LONG" in str(t.direction).upper() else "SELL"
            if hasattr(t, 'direction'):
                direction = str(t.direction.value) if hasattr(t.direction, 'value') else str(t.direction)
            offset = str(t.offset.value) if hasattr(t.offset, 'value') else str(t.offset)
            print(f"  {t.datetime}: {direction} {offset} {t.volume}@{t.price}")
    else:
        print("  No trades found in this period")

    # Get daily PnL around Labor Day
    print("\n" + "=" * 60)
    print("Daily PnL around Labor Day")
    print("=" * 60)

    daily_results = result.daily_results if hasattr(result, 'daily_results') else {}
    for date_key in sorted(daily_results.keys()):
        date_str = str(date_key)
        if date_str >= '2022-04-28' and date_str <= '2022-05-10':
            dr = daily_results[date_key]
            pnl = dr.net_pnl if hasattr(dr, 'net_pnl') else dr.get('net_pnl', 0)
            print(f"  {date_str}: PnL = {pnl:.0f}")

    # Check round trips
    print("\n" + "=" * 60)
    print("Round trips (complete trades) around Labor Day")
    print("=" * 60)

    # Build round trips from trades
    open_trades = []
    round_trips = []

    for t in sorted(trades, key=lambda x: x.datetime):
        offset = str(t.offset.value) if hasattr(t.offset, 'value') else str(t.offset)
        if "OPEN" in offset.upper():
            open_trades.append({
                'datetime': t.datetime,
                'price': t.price,
                'volume': t.volume,
                'direction': str(t.direction.value) if hasattr(t.direction, 'value') else str(t.direction),
            })
        elif "CLOSE" in offset.upper() and open_trades:
            open_t = open_trades.pop(0)
            direction = open_t['direction']
            if "LONG" in direction.upper():
                pnl = (t.price - open_t['price']) * open_t['volume'] * 10 - 2  # rough PnL in CNY
            else:
                pnl = (open_t['price'] - t.price) * open_t['volume'] * 10 - 2

            rt = {
                'entry_time': open_t['datetime'],
                'exit_time': t.datetime,
                'direction': direction,
                'entry_price': open_t['price'],
                'exit_price': t.price,
                'volume': open_t['volume'],
                'pnl': pnl,
                'pnl_pts': pnl / 10,
            }
            round_trips.append(rt)

    # Filter around Labor Day
    for rt in round_trips:
        exit_date = str(rt['exit_time'].date()) if hasattr(rt['exit_time'], 'date') else str(rt['exit_time'])[:10]
        entry_date = str(rt['entry_time'].date()) if hasattr(rt['entry_time'], 'date') else str(rt['entry_time'])[:10]
        if entry_date >= '2022-04-25' and entry_date <= '2022-05-10':
            marker = "[DISASTER]" if rt['pnl_pts'] < -400 else ""
            print(f"  {entry_date} -> {exit_date}: {rt['direction']} @ {rt['entry_price']:.0f} -> {rt['exit_price']:.0f}, PnL={rt['pnl_pts']:.0f} pts {marker}")


if __name__ == '__main__':
    main()
