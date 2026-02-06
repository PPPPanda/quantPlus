"""
Iter13: Test p2401 fixes ON TOP OF iter10-full best params
Best params from iter10-full: cb=7/70, div_threshold=0.39, max_pullback_atr=3.0
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

BEST_BASE = {
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.0,
}

def run_single(csv_path, vt_symbol, extra_setting):
    setting = {**BEST_BASE, **extra_setting}
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
    return result.stats.get('total_net_pnl', 0) / 10

# First: p2401 only search
P2401 = {"contract": "p2401.DCE", "csv": str(PROJECT / "data/analyse/wind/p2401_1min_202308-202312.csv")}

CONFIGS = {
    "best_base": {},
    # Circuit breaker variants (on top of cb=7/70)
    "cb5_70": {"circuit_breaker_losses": 5, "circuit_breaker_bars": 70},
    "cb5_100": {"circuit_breaker_losses": 5, "circuit_breaker_bars": 100},
    "cb4_100": {"circuit_breaker_losses": 4, "circuit_breaker_bars": 100},
    "cb3_120": {"circuit_breaker_losses": 3, "circuit_breaker_bars": 120},
    # Entry filter
    "ef15": {"atr_entry_filter": 1.5},
    "ef10": {"atr_entry_filter": 1.0},
    # Rolling drawdown
    "dd8_6_60": {"dd_window_trades": 8, "dd_threshold_atr": 6.0, "dd_pause_bars": 60},
    "dd6_5_80": {"dd_window_trades": 6, "dd_threshold_atr": 5.0, "dd_pause_bars": 80},
    # Cooldown (L1)
    "cd3_30": {"cooldown_losses": 3, "cooldown_bars": 30},
    "cd2_40": {"cooldown_losses": 2, "cooldown_bars": 40},
    # Escalating CB
    "cb_esc": {"cb_escalation": True, "cb_max_level": 3},
    # Combos
    "cb5_100_ef15": {"circuit_breaker_losses": 5, "circuit_breaker_bars": 100, "atr_entry_filter": 1.5},
    "cb5_100_dd6": {"circuit_breaker_losses": 5, "circuit_breaker_bars": 100, "dd_window_trades": 6, "dd_threshold_atr": 5.0, "dd_pause_bars": 60},
}

def run_full13(extra_setting, name):
    total = 0
    neg = []
    below_180 = []
    details = {}
    for bench in BENCHMARKS:
        vt = bench["contract"]
        csv = bench["csv"]
        pts = run_single(csv, vt, extra_setting)
        total += pts
        cn = vt.split('.')[0]
        details[cn] = pts
        if pts < 0:
            neg.append(f"{cn}({pts:.1f})")
        if pts < -180:
            below_180.append(cn)
    
    t_ok = total >= 12164
    n_ok = not below_180
    status = "PASS" if t_ok and n_ok else "FAIL"
    return total, neg, below_180, status, details

def main():
    # p2401 single contract search
    print("=== p2401 search (on iter10-full best base) ===")
    print(f"{'Config':<25} {'pts':>8} {'improve':>8}")
    print("-" * 45)
    
    base = run_single(P2401["csv"], P2401["contract"], {})
    print(f"{'best_base':<25} {base:>+8.1f} {'---':>8}")
    
    p2401_results = []
    for name, setting in CONFIGS.items():
        if name == "best_base":
            continue
        pts = run_single(P2401["csv"], P2401["contract"], setting)
        improve = pts - base
        print(f"{name:<25} {pts:>+8.1f} {improve:>+8.1f}")
        p2401_results.append({"name": name, "pts": pts, "improve": improve, "setting": setting})
    
    # Sort and pick top 3
    p2401_results.sort(key=lambda x: x['pts'], reverse=True)
    top3 = p2401_results[:3]
    print(f"\nTop 3: {[t['name'] for t in top3]}")
    
    # Full 13-contract test for top 3
    print(f"\n=== Full 13-contract validation ===")
    
    # Baseline full 13
    total_b, neg_b, below_b, status_b, det_b = run_full13({}, "best_base")
    print(f"best_base: TOTAL={total_b:.1f}, neg={neg_b}, below180={below_b}, {status_b}")
    
    for cfg in top3:
        total, neg, below, status, det = run_full13(cfg['setting'], cfg['name'])
        p2401_val = det.get('p2401', 0)
        p2209_val = det.get('p2209', 0)
        p2601_val = det.get('p2601', 0)
        print(f"{cfg['name']}: TOTAL={total:.1f}, p2401={p2401_val:+.1f}, p2209={p2209_val:+.1f}, p2601={p2601_val:+.1f}, neg={neg}, {status}")
        
        with open(PROJECT / f"experiments/iter13/test_{cfg['name']}_full.json", 'w') as f:
            json.dump({"setting": {**BEST_BASE, **cfg['setting']}, "total": total, "details": det, "neg": neg, "status": status}, f, indent=2)
    
    # Save summary
    with open(PROJECT / "experiments/iter13/iter13_search_summary.json", 'w') as f:
        json.dump({
            "best_base": BEST_BASE, "baseline_total": total_b,
            "p2401_search": p2401_results, "top3": [t['name'] for t in top3]
        }, f, indent=2)

if __name__ == "__main__":
    main()
