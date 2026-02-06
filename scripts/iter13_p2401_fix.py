"""
Iter13 Phase 4: p2401 修复方案搜索
测试多种参数组合，目标：p2401 > -180 且不伤其他
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from run_13bench import import_csv_to_db, BENCHMARKS

PROJECT = Path(__file__).resolve().parents[1]

def run_single(csv_path, vt_symbol, setting):
    start, end, _ = import_csv_to_db(csv_path, vt_symbol)
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start, end=end,
        strategy_class=CtaChanPivotStrategy,
        strategy_setting={
            "debug_enabled": False,
            "debug_log_console": False,
            **setting,
        },
        interval=Interval.MINUTE,
        rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0,
        capital=1_000_000,
    )
    return result.stats.get('total_net_pnl', 0) / 10  # pts

# Phase 1: Quick test on p2401 only
P2401 = {"contract": "p2401.DCE", "csv": str(PROJECT / "data/analyse/wind/p2401_1min_202308-202312.csv")}

CONFIGS = {
    "baseline": {},
    # ATR volatility gate
    "atr_008": {"min_atr_pct": 0.008},
    "atr_010": {"min_atr_pct": 0.010},
    "atr_012": {"min_atr_pct": 0.012},
    "atr_015": {"min_atr_pct": 0.015},
    # Stronger circuit breaker
    "cb4_40": {"circuit_breaker_losses": 4, "circuit_breaker_bars": 40},
    "cb3_60": {"circuit_breaker_losses": 3, "circuit_breaker_bars": 60},
    "cb4_80": {"circuit_breaker_losses": 4, "circuit_breaker_bars": 80},
    "cb3_100": {"circuit_breaker_losses": 3, "circuit_breaker_bars": 100},
    # Rolling drawdown breaker
    "dd8_6": {"dd_window_trades": 8, "dd_threshold_atr": 6.0, "dd_pause_bars": 60},
    "dd10_8": {"dd_window_trades": 10, "dd_threshold_atr": 8.0, "dd_pause_bars": 80},
    "dd6_5": {"dd_window_trades": 6, "dd_threshold_atr": 5.0, "dd_pause_bars": 80},
    # Warmup period
    "warm30": {"warmup_bars": 30},
    "warm60": {"warmup_bars": 60},
    # Entry filter
    "ef30": {"atr_entry_filter": 3.0},
    "ef15": {"atr_entry_filter": 1.5},
    # Tighter trailing
    "trail25": {"atr_trailing_mult": 2.5, "atr_activate_mult": 2.0},
    # Combinations
    "atr010_cb4": {"min_atr_pct": 0.010, "circuit_breaker_losses": 4, "circuit_breaker_bars": 60},
    "atr012_cb3_dd": {"min_atr_pct": 0.012, "circuit_breaker_losses": 3, "circuit_breaker_bars": 80, "dd_window_trades": 8, "dd_threshold_atr": 6.0, "dd_pause_bars": 60},
    "atr015_cb3_warm": {"min_atr_pct": 0.015, "circuit_breaker_losses": 3, "circuit_breaker_bars": 80, "warmup_bars": 30},
    "kitchen_sink": {"min_atr_pct": 0.012, "circuit_breaker_losses": 3, "circuit_breaker_bars": 100, "dd_window_trades": 6, "dd_threshold_atr": 5.0, "dd_pause_bars": 80, "warmup_bars": 30, "atr_entry_filter": 1.5},
}

def main():
    print("=== p2401 Fix Search ===")
    print(f"{'Config':<25} {'pts':>8} {'improve':>8}")
    print("-" * 45)
    
    base = run_single(P2401["csv"], P2401["contract"], {})
    print(f"{'baseline':<25} {base:>+8.1f} {'---':>8}")
    
    results = []
    best_name = "baseline"
    best_pts = base
    
    for name, setting in CONFIGS.items():
        if name == "baseline":
            continue
        pts = run_single(P2401["csv"], P2401["contract"], setting)
        improve = pts - base
        marker = " **" if pts > best_pts else ""
        print(f"{name:<25} {pts:>+8.1f} {improve:>+8.1f}{marker}")
        results.append({"name": name, "setting": setting, "pts": pts, "improve": improve})
        if pts > best_pts:
            best_pts = pts
            best_name = name
    
    print(f"\nBest: {best_name} = {best_pts:+.1f}pts (vs baseline {base:+.1f})")
    
    # Phase 2: Run top 3 configs on 3 key contracts
    top3 = sorted(results, key=lambda x: x['pts'], reverse=True)[:3]
    
    KEY_CONTRACTS = [
        ("p2209", "p2209.DCE", str(PROJECT / "data/analyse/wind/p2209_1min_202204-202208.csv")),
        ("p2601", "p2601.DCE", str(PROJECT / "data/analyse/p2601_1min_202507-202512.csv")),
        ("p2405", "p2405.DCE", str(PROJECT / "data/analyse/wind/p2405_1min_202312-202404.csv")),
    ]
    
    print(f"\n=== Top 3 configs on key contracts ===")
    print(f"{'Config':<25} {'p2401':>8} {'p2209':>8} {'p2601':>8} {'p2405':>8} {'verdict':>8}")
    print("-" * 75)
    
    # Baseline on key contracts
    base_vals = {}
    for cname, sym, csv in KEY_CONTRACTS:
        base_vals[cname] = run_single(csv, sym, {})
    print(f"{'baseline':<25} {base:>+8.1f} {base_vals['p2209']:>+8.1f} {base_vals['p2601']:>+8.1f} {base_vals['p2405']:>+8.1f} {'BASE':>8}")
    
    for cfg in top3:
        vals = {"p2401": cfg['pts']}
        for cname, sym, csv in KEY_CONTRACTS:
            vals[cname] = run_single(csv, sym, cfg['setting'])
        
        # Verdict
        p2209_ok = vals['p2209'] >= base_vals['p2209'] * 0.9  # max 10% drop
        p2601_ok = vals['p2601'] >= base_vals['p2601'] * 0.85
        p2405_ok = vals['p2405'] >= base_vals['p2405'] * 0.85
        verdict = "PASS" if (p2209_ok and p2601_ok and p2405_ok) else "FAIL"
        
        print(f"{cfg['name']:<25} {vals['p2401']:>+8.1f} {vals['p2209']:>+8.1f} {vals['p2601']:>+8.1f} {vals['p2405']:>+8.1f} {verdict:>8}")
    
    # Save
    with open(PROJECT / "experiments/iter13/p2401_fix_search.json", 'w') as f:
        json.dump({"baseline": base, "results": results, "top3": [t['name'] for t in top3]}, f, indent=2)
    print(f"\nResults saved")

if __name__ == "__main__":
    main()
