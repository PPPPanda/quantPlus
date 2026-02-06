"""
Iter13 Phase 4: S17 条件性做空测试
先在 p2401(目标改善) 和 p2209(不能退化) 上对比测试
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
    pnl = result.stats.get('total_net_pnl', 0)
    trades_count = len(result.trades) if result.trades else 0
    return pnl / 10, trades_count  # pts

# Key test contracts
TESTS = {
    "p2401": {"contract": "p2401.DCE", "csv": str(PROJECT / "data/analyse/wind/p2401_1min_202308-202312.csv")},
    "p2209": {"contract": "p2209.DCE", "csv": str(PROJECT / "data/analyse/wind/p2209_1min_202204-202208.csv")},
    "p2601": {"contract": "p2601.DCE", "csv": str(PROJECT / "data/analyse/p2601_1min_202507-202512.csv")},
}

# Test configurations
CONFIGS = {
    "baseline": {},
    "short_p2":  {"conditional_short": True, "short_min_pivots": 2, "short_max_diff_15m": 0},
    "short_p2_diff": {"conditional_short": True, "short_min_pivots": 2, "short_max_diff_15m": -5},
    "short_p3":  {"conditional_short": True, "short_min_pivots": 3, "short_max_diff_15m": 0},
}

def main():
    print("=== S17 Conditional Short Test ===\n")
    
    results = {}
    for config_name, setting in CONFIGS.items():
        print(f"--- {config_name}: {setting or 'default'} ---")
        results[config_name] = {}
        for name, info in TESTS.items():
            pts, trades = run_single(info["csv"], info["contract"], setting)
            results[config_name][name] = {"pts": pts, "trades": trades}
            print(f"  {name}: {pts:+.1f}pts ({trades}t)")
        print()
    
    # Comparison
    print("=== Comparison ===")
    print(f"{'Config':<20} {'p2401':>10} {'p2209':>10} {'p2601':>10} {'verdict':>10}")
    print("-" * 65)
    for config_name, data in results.items():
        p2401 = data.get('p2401', {}).get('pts', 0)
        p2209 = data.get('p2209', {}).get('pts', 0)
        p2601 = data.get('p2601', {}).get('pts', 0)
        
        if config_name == 'baseline':
            verdict = 'BASE'
        else:
            # p2209 must not drop > 500pts, p2401 must improve
            p2209_ok = p2209 >= results['baseline']['p2209']['pts'] - 500
            p2401_ok = p2401 > results['baseline']['p2401']['pts']
            p2601_ok = p2601 >= results['baseline']['p2601']['pts'] - 100
            if p2209_ok and p2401_ok and p2601_ok:
                verdict = 'PASS'
            elif not p2209_ok:
                verdict = 'KILL_P2209'
            elif not p2401_ok:
                verdict = 'NO_HELP'
            else:
                verdict = 'KILL_P2601'
        
        print(f"{config_name:<20} {p2401:>+10.1f} {p2209:>+10.1f} {p2601:>+10.1f} {verdict:>10}")
    
    # Save
    with open(PROJECT / "experiments/iter13/s17_test.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to experiments/iter13/s17_test.json")

if __name__ == "__main__":
    main()
