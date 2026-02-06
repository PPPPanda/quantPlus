"""
Iter13: Test cb3_100 and combinations on full 13 contracts
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
    return result.stats.get('total_net_pnl', 0) / 10

CONFIGS = {
    "baseline": {},
    "cb3_100": {"circuit_breaker_losses": 3, "circuit_breaker_bars": 100},
    "cb3_100_ef15": {"circuit_breaker_losses": 3, "circuit_breaker_bars": 100, "atr_entry_filter": 1.5},
    "cb3_100_dd6_5": {"circuit_breaker_losses": 3, "circuit_breaker_bars": 100, "dd_window_trades": 6, "dd_threshold_atr": 5.0, "dd_pause_bars": 80},
    "cb4_60_ef15": {"circuit_breaker_losses": 4, "circuit_breaker_bars": 60, "atr_entry_filter": 1.5},
}

def main():
    for config_name, setting in CONFIGS.items():
        total = 0
        neg = []
        below_180 = []
        
        for bench in BENCHMARKS:
            vt_symbol = bench["contract"]
            csv_path = bench["csv"]
            pts = run_single(csv_path, vt_symbol, setting)
            total += pts
            name = vt_symbol.split('.')[0]
            if pts < 0:
                neg.append(f"{name}({pts:.1f})")
            if pts < -180:
                below_180.append(name)
        
        status = "PASS" if total >= 12164 and not below_180 else "FAIL"
        print(f"=== {config_name} ===")
        print(f"TOTAL: {total:.1f}pts")
        print(f"Neg: {neg}")
        print(f"Below -180: {below_180}")
        print(f"TOTAL>=12164: {'PASS' if total >= 12164 else 'FAIL'}")
        print(f"All neg>-180: {'PASS' if not below_180 else 'FAIL'}")
        print(f"STATUS: {status}")
        print()
        
        with open(PROJECT / f"experiments/iter13/test_{config_name}.json", 'w') as f:
            json.dump({"setting": setting, "total_pts": total, "neg": neg, "below_180": below_180, "status": status}, f, indent=2)

if __name__ == "__main__":
    main()
