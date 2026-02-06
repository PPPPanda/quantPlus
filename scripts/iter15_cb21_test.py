#!/usr/bin/env python3
"""
iter15_cb21_test.py - Test CB2.1 (magnitude weighted + adaptive recovery)
Compare baseline vs CB2.1 on 13 contracts
"""

import sys
import os
import json
from pathlib import Path
from datetime import timedelta

# Setup path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(str(ROOT))

from run_13bench import BENCHMARKS, import_csv_to_db, BT_PARAMS, DEFAULT_SETTING
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from vnpy.trader.constant import Interval

# Known best baseline parameters
BASELINE_PARAMS = {
    **DEFAULT_SETTING,
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
}

# CB2.1 test configurations
CB21_CONFIGS = {
    "cb21_default": {
        "cb2_enabled": True,
        "cb2_window_trades": 10,
        "cb2_l1_threshold": -4.0,
        "cb2_l2_threshold": -6.0,
        "cb2_l3_threshold": -8.0,
        "cb2_recovery_win": 2,
        "cb2_magnitude_weight": True,
        "cb2_weight_floor": 0.5,
        "cb2_weight_cap": 3.0,
        "cb2_decay_per_win": 0.3,
        "cb2_atr_recovery": True,
        "cb2_atr_recovery_factor": 0.8,
        "cb2_l1_skip_pct": 0.5,
    },
    "cb21_relaxed": {
        "cb2_enabled": True,
        "cb2_window_trades": 15,  # Larger window
        "cb2_l1_threshold": -5.0,  # Relaxed thresholds
        "cb2_l2_threshold": -7.0,
        "cb2_l3_threshold": -9.0,
        "cb2_recovery_win": 2,
        "cb2_magnitude_weight": True,
        "cb2_weight_floor": 0.5,
        "cb2_weight_cap": 3.0,
        "cb2_decay_per_win": 0.4,  # Faster decay
        "cb2_atr_recovery": True,
        "cb2_atr_recovery_factor": 0.85,
        "cb2_l1_skip_pct": 0.3,  # Skip less at L1
    },
    "cb21_tight": {
        "cb2_enabled": True,
        "cb2_window_trades": 8,  # Smaller window
        "cb2_l1_threshold": -3.0,  # Tighter thresholds
        "cb2_l2_threshold": -5.0,
        "cb2_l3_threshold": -7.0,
        "cb2_recovery_win": 3,
        "cb2_magnitude_weight": True,
        "cb2_weight_floor": 0.5,
        "cb2_weight_cap": 3.0,
        "cb2_decay_per_win": 0.2,  # Slower decay
        "cb2_atr_recovery": True,
        "cb2_atr_recovery_factor": 0.75,
        "cb2_l1_skip_pct": 0.5,
    },
    "cb21_no_l1skip": {
        # Same as default but no L1 signal skipping
        "cb2_enabled": True,
        "cb2_window_trades": 10,
        "cb2_l1_threshold": -4.0,
        "cb2_l2_threshold": -6.0,
        "cb2_l3_threshold": -8.0,
        "cb2_recovery_win": 2,
        "cb2_magnitude_weight": True,
        "cb2_weight_floor": 0.5,
        "cb2_weight_cap": 3.0,
        "cb2_decay_per_win": 0.3,
        "cb2_atr_recovery": True,
        "cb2_atr_recovery_factor": 0.8,
        "cb2_l1_skip_pct": 0.0,  # No skipping at L1
    },
}


def run_config(config_name, extra_params):
    """Run a configuration across all 13 contracts"""
    results = {}
    total_pnl = 0.0
    
    for bench in BENCHMARKS:
        contract = bench["contract"]  # e.g. "p2201.DCE"
        csv_path = bench["csv"]
        symbol = contract.split(".")[0]  # e.g. "p2201"
        
        if not csv_path.exists():
            print(f"  Skipping {symbol} (file not found: {csv_path})")
            continue
        
        # Prepare database and get date range
        start, end, bar_count = import_csv_to_db(csv_path, contract)
        
        # Build parameters
        params = {**BASELINE_PARAMS, **extra_params}
        
        # Run backtest
        result = run_backtest(
            vt_symbol=contract,
            start=start - timedelta(days=1),
            end=end + timedelta(days=1),
            strategy_class=CtaChanPivotStrategy,
            strategy_setting=params,
            **BT_PARAMS,
        )
        
        stats = result.stats or {}
        pnl = stats.get("total_net_pnl", 0) or 0
        trades = stats.get("total_trade_count", 0) or 0
        pnl_pts = pnl / 10.0  # Convert to points
        
        results[symbol] = {"pnl": pnl_pts, "trades": trades}
        total_pnl += pnl_pts
        print(f"  {symbol}: {pnl_pts:.1f} pts ({trades} trades)")
    
    results["TOTAL"] = {"pnl": total_pnl}
    return results


def main():
    print("=" * 60)
    print("CB2.1 Test: 13 contracts")
    print("=" * 60)
    
    # Baseline (CB2 disabled)
    print("\n[1/5] Running baseline (cb2_enabled=False)...")
    baseline = run_config("baseline", {"cb2_enabled": False})
    baseline_total = baseline["TOTAL"]["pnl"]
    print(f"  ==> Baseline TOTAL: {baseline_total:.1f} pts")
    
    # Protected contracts
    baseline_p2209 = baseline.get("p2209", {}).get("pnl", 0)
    baseline_p2601 = baseline.get("p2601", {}).get("pnl", 0)
    baseline_p2401 = baseline.get("p2401", {}).get("pnl", 0)
    
    all_results = {"baseline": baseline}
    
    # Test each CB2.1 configuration
    for i, (cfg_name, cfg_params) in enumerate(CB21_CONFIGS.items(), 2):
        print(f"\n[{i}/5] Running {cfg_name}...")
        results = run_config(cfg_name, cfg_params)
        total = results["TOTAL"]["pnl"]
        
        # Check constraints
        p2209 = results.get("p2209", {}).get("pnl", 0)
        p2601 = results.get("p2601", {}).get("pnl", 0)
        p2401 = results.get("p2401", {}).get("pnl", 0)
        
        delta_total = total - baseline_total
        delta_p2209 = p2209 - baseline_p2209
        delta_p2601 = p2601 - baseline_p2601
        delta_p2401 = p2401 - baseline_p2401
        
        # Pass criteria: TOTAL not degraded, p2209/p2601 not hurt
        total_ok = total >= baseline_total * 0.95  # Allow 5% tolerance
        p2209_ok = p2209 >= baseline_p2209 * 0.9  # Allow 10% tolerance
        p2601_ok = p2601 >= baseline_p2601 * 0.9
        p2401_ok = p2401 >= baseline_p2401  # p2401 should not get worse
        
        status = "PASS" if (total_ok and p2209_ok and p2601_ok and p2401_ok) else "FAIL"
        
        print(f"  ==> {cfg_name}: TOTAL={total:.1f} (delta={delta_total:+.1f})")
        print(f"    p2209: {p2209:.1f} (delta={delta_p2209:+.1f}) {'OK' if p2209_ok else 'FAIL'}")
        print(f"    p2601: {p2601:.1f} (delta={delta_p2601:+.1f}) {'OK' if p2601_ok else 'FAIL'}")
        print(f"    p2401: {p2401:.1f} (delta={delta_p2401:+.1f}) {'OK' if p2401_ok else 'FAIL'}")
        print(f"  Status: {status}")
        
        all_results[cfg_name] = results
    
    # Save results
    output_path = ROOT / "experiments/iter15/cb21_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<20} {'TOTAL':>10} {'Delta':>10} {'p2401':>10} {'Status':>8}")
    print("-" * 60)
    for cfg_name, results in all_results.items():
        total = results["TOTAL"]["pnl"]
        delta = total - baseline_total if cfg_name != "baseline" else 0
        p2401 = results.get("p2401", {}).get("pnl", 0)
        p2209 = results.get("p2209", {}).get("pnl", 0)
        
        # Quick pass/fail
        status = "BASE" if cfg_name == "baseline" else (
            "PASS" if (total >= baseline_total * 0.95 and 
                       p2209 >= baseline_p2209 * 0.9 and
                       p2401 >= baseline_p2401) else "FAIL"
        )
        print(f"{cfg_name:<20} {total:>10.1f} {delta:>+10.1f} {p2401:>10.1f} {status:>8}")


if __name__ == "__main__":
    main()
