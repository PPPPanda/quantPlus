"""
Iter14: p2305 只差3.6pts（-183.6 → 需要 >-180）
做参数微调细网格搜索
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_13bench import import_csv_to_db, BENCHMARKS

PROJECT = Path(__file__).resolve().parents[1]

BEST_BASE = {
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.0,
}

def run_single(csv_path, vt_symbol, extra_setting):
    setting = {**BEST_BASE, **extra_setting, "debug_enabled": False, "debug_log_console": False}
    start, end, _ = import_csv_to_db(csv_path, vt_symbol)
    result = run_backtest(
        vt_symbol=vt_symbol, start=start, end=end,
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        interval=Interval.MINUTE,
        rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0,
        capital=1_000_000,
    )
    return result.stats.get('total_net_pnl', 0) / 10

# p2305 CSV
P2305 = next(b for b in BENCHMARKS if "p2305" in b["contract"])

# Fine-grid search: tweak each param slightly
CONFIGS = {
    "baseline": {},
    # div_threshold variants (current 0.39)
    "div035": {"div_threshold": 0.35},
    "div037": {"div_threshold": 0.37},
    "div040": {"div_threshold": 0.40},
    "div042": {"div_threshold": 0.42},
    "div045": {"div_threshold": 0.45},
    # max_pullback_atr variants (current 3.0)
    "mpa28": {"max_pullback_atr": 2.8},
    "mpa32": {"max_pullback_atr": 3.2},
    "mpa35": {"max_pullback_atr": 3.5},
    "mpa40": {"max_pullback_atr": 4.0},
    # cb fine tune
    "cb7_80": {"circuit_breaker_bars": 80},
    "cb7_90": {"circuit_breaker_bars": 90},
    "cb6_70": {"circuit_breaker_losses": 6},
    "cb8_70": {"circuit_breaker_losses": 8},
    # stop buffer (current 0.02)
    "sb003": {"stop_buffer_atr_pct": 0.03},
    "sb004": {"stop_buffer_atr_pct": 0.04},
    "sb005": {"stop_buffer_atr_pct": 0.05},
    # atr trailing mult (current 3.0)
    "trail32": {"atr_trailing_mult": 3.2},
    "trail35": {"atr_trailing_mult": 3.5},
    # cooldown 
    "cd3_20": {"cooldown_losses": 3, "cooldown_bars": 20},
    # min_hold_bars
    "mhb2": {"min_hold_bars": 2},
    "mhb3": {"min_hold_bars": 3},
    # Combinations
    "div042_sb003": {"div_threshold": 0.42, "stop_buffer_atr_pct": 0.03},
    "div042_mpa35": {"div_threshold": 0.42, "max_pullback_atr": 3.5},
    "sb003_mhb2": {"stop_buffer_atr_pct": 0.03, "min_hold_bars": 2},
    "div040_sb004": {"div_threshold": 0.40, "stop_buffer_atr_pct": 0.04},
}

def main():
    print("=== p2305 Fine-Grid Search (target: > -180pts) ===")
    print(f"{'Config':<20} {'pts':>8} {'vs base':>8} {'pass':>5}")
    print("-" * 45)
    
    results = []
    base_pts = None
    
    for name, setting in CONFIGS.items():
        pts = run_single(P2305["csv"], P2305["contract"], setting)
        if name == "baseline":
            base_pts = pts
        improve = pts - base_pts if base_pts is not None else 0
        passed = "PASS" if pts > -180 else "FAIL"
        print(f"{name:<20} {pts:>+8.1f} {improve:>+8.1f} {passed:>5}")
        results.append({"name": name, "pts": pts, "improve": improve, "passed": pts > -180, "setting": setting})
    
    # Find all passing configs
    passing = [r for r in results if r['passed'] and r['name'] != 'baseline']
    print(f"\nPassing configs: {[p['name'] for p in passing]}")
    
    with open(PROJECT / "experiments/iter14/p2305_search.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
