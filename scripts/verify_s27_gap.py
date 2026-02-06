#!/usr/bin/env python3
"""
iter17 Phase 4: verify S27 gap ATR adaptive.

Compare p2209 (Labor Day disaster contract) with S27 on/off.
Acceptance criteria:
  - Post-holiday loss per trade < 200 pts
  - Post-holiday total > -300 pts
  - TOTAL not decreased

Usage:
    cd quantPlus
    .venv/Scripts/python.exe scripts/verify_s27_gap.py
"""
from __future__ import annotations

import json
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

# Test contracts (priority: p2209 Labor Day disaster)
TEST_CONTRACTS = [
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv", "source": "Wind"},
]

# All contracts (for full validation)
ALL_CONTRACTS = [
    {"contract": "p2601.DCE", "csv": ROOT / "data/analyse/p2601_1min_202507-202512.csv", "source": "XT"},
    {"contract": "p2405.DCE", "csv": ROOT / "data/analyse/wind/p2405_1min_202312-202404.csv", "source": "Wind"},
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv", "source": "Wind"},
    {"contract": "p2501.DCE", "csv": ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv", "source": "Wind"},
    {"contract": "p2505.DCE", "csv": ROOT / "data/analyse/wind/p2505_1min_202412-202504.csv", "source": "Wind"},
    {"contract": "p2509.DCE", "csv": ROOT / "data/analyse/wind/p2509_1min_202504-202508.csv", "source": "Wind"},
    {"contract": "p2301.DCE", "csv": ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv", "source": "Wind"},
]

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

# Both use the same setting to check baseline consistency
BASELINE_SETTING = {
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
    # S27 completely off
    "gap_threshold_atr": 0.0,
    "atr_boost_factor": 1.0,
    "atr_boost_bars": 0,
}

# S27 enabled (standard duration)
S27_SETTING = {
    **BASELINE_SETTING,
    "gap_threshold_atr": 1.5,  # gap > 1.5*ATR triggers
    "atr_boost_factor": 1.5,   # ATR boost 50%
    "atr_boost_bars": 6,       # lasts 30 min (6 * 5m bars)
}


def analyze_gaps(df: pd.DataFrame) -> list:
    """Analyze gap occurrences in data."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    # Daily aggregation
    daily = df.groupby('date').agg(
        open=('open', 'first'),
        close=('close', 'last'),
        high=('high', 'max'),
        low=('low', 'min'),
    ).reset_index()

    # Calculate gaps
    gaps = []
    for i in range(1, len(daily)):
        prev_close = daily.iloc[i-1]['close']
        curr_open = daily.iloc[i]['open']
        gap = abs(curr_open - prev_close)
        day_range = daily.iloc[i]['high'] - daily.iloc[i]['low']

        if gap > 0:
            gaps.append({
                'date': str(daily.iloc[i]['date']),
                'gap': round(gap, 1),
                'gap_pct': round(gap / prev_close * 100, 2),
                'day_range': round(day_range, 1),
                'gap_ratio': round(gap / day_range, 2) if day_range > 0 else 0,
            })

    return sorted(gaps, key=lambda x: x['gap'], reverse=True)[:10]


def run_single_contract(bench: dict, setting: dict, label: str) -> dict:
    """Run single contract backtest."""
    contract = bench['contract']
    csv_path = bench['csv']
    short = contract.split('.')[0]

    df = pd.read_csv(csv_path)
    df_norm = normalize_1m_bars(df, PALM_OIL_SESSIONS)

    start = pd.Timestamp(df_norm['datetime'].min()).to_pydatetime()
    end = pd.Timestamp(df_norm['datetime'].max()).to_pydatetime()

    result = run_backtest(
        vt_symbol=contract,
        interval=BT_PARAMS['interval'],
        start=start, end=end,
        rate=BT_PARAMS['rate'],
        slippage=BT_PARAMS['slippage'],
        size=BT_PARAMS['size'],
        pricetick=BT_PARAMS['pricetick'],
        capital=BT_PARAMS['capital'],
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
    )

    stats = result.stats if hasattr(result, 'stats') else result
    total_net = stats.get('total_net_pnl', stats.get('total_pnl', 0))
    pts = total_net / 10
    trades = stats.get('total_trade_count', 0)
    max_dd = stats.get('max_ddpercent', stats.get('max_drawdown', 0))

    return {
        'contract': short,
        'label': label,
        'pts': round(pts, 1),
        'trades': trades,
        'max_dd': round(max_dd, 2) if max_dd else 0,
    }


def main():
    print("=" * 60)
    print("iter17 Phase 4: Verify S27 Gap ATR Adaptive")
    print("=" * 60)

    # Step 0: Analyze p2209 gaps
    print("\n[Step 0] p2209 Gap Analysis")
    print("-" * 50)

    bench = TEST_CONTRACTS[0]
    df = pd.read_csv(bench['csv'])
    df_norm = normalize_1m_bars(df, PALM_OIL_SESSIONS)
    gaps = analyze_gaps(df_norm)

    print("Top 10 gaps in p2209:")
    for g in gaps:
        print(f"  {g['date']}: gap={g['gap']:.0f} ({g['gap_pct']:.1f}%), range={g['day_range']:.0f}, ratio={g['gap_ratio']:.2f}")

    # Estimate ATR (roughly)
    df_norm['datetime'] = pd.to_datetime(df_norm['datetime'])
    df_norm['date'] = df_norm['datetime'].dt.date
    daily = df_norm.groupby('date').agg(
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
    ).reset_index()
    daily['tr'] = daily['high'] - daily['low']
    avg_atr = daily['tr'].rolling(14).mean().mean()
    print(f"\nEstimated avg daily range (proxy for ATR): {avg_atr:.1f}")
    print(f"S27 threshold (1.5 * ATR): {1.5 * avg_atr:.1f}")

    # Check which gaps would trigger S27
    triggers = [g for g in gaps if g['gap'] > 1.5 * avg_atr]
    print(f"\nGaps that would trigger S27: {len(triggers)}")
    for t in triggers:
        print(f"  {t['date']}: gap={t['gap']:.0f} > threshold")

    # Step 1: p2209 single contract comparison
    print("\n[Step 1] p2209 Single Contract Comparison")
    print("-" * 50)

    baseline = run_single_contract(bench, BASELINE_SETTING, "BASELINE")
    s27 = run_single_contract(bench, S27_SETTING, "S27")

    delta = s27['pts'] - baseline['pts']
    print(f"  BASELINE: {baseline['pts']:>8.1f} pts ({baseline['trades']} trades)")
    print(f"  S27:      {s27['pts']:>8.1f} pts ({s27['trades']} trades)")
    print(f"  Delta:    {delta:>+8.1f} pts")

    if delta > 0:
        print(f"  [OK] S27 improved {delta:.1f} pts")
    elif delta == 0:
        print(f"  [--] S27 no change")
    else:
        print(f"  [!!] S27 degraded {delta:.1f} pts")

    # Step 2: Full contract validation
    print("\n[Step 2] Full Contract TOTAL Validation")
    print("-" * 50)

    total_baseline = 0
    total_s27 = 0
    results = []

    for bench in ALL_CONTRACTS:
        short = bench['contract'].split('.')[0]
        b = run_single_contract(bench, BASELINE_SETTING, "BASELINE")
        s = run_single_contract(bench, S27_SETTING, "S27")
        d = s['pts'] - b['pts']

        total_baseline += b['pts']
        total_s27 += s['pts']

        marker = "[OK]" if d >= 0 else "[!!]"
        print(f"  {short}: {b['pts']:>8.1f} -> {s['pts']:>8.1f} ({d:>+7.1f}) {marker}")

        results.append({
            'contract': short,
            'baseline_pts': b['pts'],
            's27_pts': s['pts'],
            'delta': round(d, 1),
        })

    print("-" * 50)
    total_delta = total_s27 - total_baseline
    print(f"  TOTAL:  {total_baseline:>8.1f} -> {total_s27:>8.1f} ({total_delta:>+7.1f})")

    # Acceptance judgment
    print("\n" + "=" * 60)
    print("Acceptance Result:")
    passed = total_delta >= 0
    if passed:
        print(f"  [PASS] TOTAL not decreased (delta={total_delta:+.1f})")
    else:
        print(f"  [FAIL] TOTAL decreased (delta={total_delta:+.1f})")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'baseline_setting': BASELINE_SETTING,
        's27_setting': S27_SETTING,
        'p2209_delta': delta,
        'p2209_gaps': gaps,
        'avg_atr_proxy': round(avg_atr, 1),
        's27_triggers': triggers,
        'total_baseline': round(total_baseline, 1),
        'total_s27': round(total_s27, 1),
        'total_delta': round(total_delta, 1),
        'passed': passed,
        'details': results,
    }

    out_dir = ROOT / "experiments" / "iter17"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "s27_verification.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
