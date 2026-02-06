"""
Iter13 Phase 4 Round 1: S14 SMA偏离度过滤 网格搜索
先在 p2401 上搜最优，再全量 13 合约验证
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
    trades = result.stats.get('total_count', 0)
    return pnl / 10, trades  # pts

def main():
    # Grid search on p2401 only first
    p2401_csv = str(PROJECT / "data/analyse/wind/p2401_1min_202308-202312.csv")
    p2401_sym = "p2401.DCE"
    
    # Search: sma_trend_period x max_sma_deviation
    # When max_sma_deviation > 0, S13 (below-SMA filter) is auto-disabled
    # Only S14 (above-SMA deviation filter) is active
    sma_periods = [60, 90, 120, 180]
    deviations = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    
    print("=== S14 Grid Search on p2401 ===")
    print(f"{'sma':>5} {'dev':>6} | {'pts':>8} {'trades':>6} {'improve':>8}")
    print("-" * 50)
    
    # Baseline (no SMA filter)
    base_pts, base_trades = run_single(p2401_csv, p2401_sym, {})
    print(f"{'base':>5} {'0':>6} | {base_pts:>8.1f} {base_trades:>6} {'---':>8}")
    
    results = []
    best = (base_pts, 0, 0.0)
    
    for sma in sma_periods:
        for dev in deviations:
            setting = {"sma_trend_period": sma, "max_sma_deviation": dev}
            pts, trades = run_single(p2401_csv, p2401_sym, setting)
            improve = pts - base_pts
            results.append({
                'sma_period': sma, 'deviation': dev,
                'pts': pts, 'trades': trades, 'improve': improve
            })
            marker = " *" if pts > best[0] else ""
            print(f"{sma:>5} {dev:>6.3f} | {pts:>8.1f} {trades:>6} {improve:>+8.1f}{marker}")
            if pts > best[0]:
                best = (pts, sma, dev)
    
    print(f"\nBest on p2401: sma={best[1]}, dev={best[2]:.3f} -> {best[0]:.1f}pts (vs baseline {base_pts:.1f})")
    
    # Save results
    out = {"baseline_pts": base_pts, "grid": results, "best": {"sma": best[1], "dev": best[2], "pts": best[0]}}
    with open(PROJECT / "experiments/iter13/s14_grid_p2401.json", 'w') as f:
        json.dump(out, f, indent=2)
    
    # Now run full 13-contract with best params
    if best[0] > base_pts:
        print(f"\n=== Full 13-contract test with sma={best[1]}, dev={best[2]} ===")
        best_setting = {"sma_trend_period": best[1], "max_sma_deviation": best[2]}
        total = 0
        neg_contracts = []
        below_180 = []
        for bench in BENCHMARKS:
            vt_symbol = bench["contract"]
            csv_path = bench["csv"]
            pts, trades = run_single(csv_path, vt_symbol, best_setting)
            total += pts
            name = vt_symbol.split('.')[0]
            status = "OK" if pts >= 0 else ("WARN" if pts > -180 else "FAIL")
            print(f"  {name}: {pts:+.1f}pts ({trades}t) [{status}]")
            if pts < 0:
                neg_contracts.append(f"{name}({pts:.1f})")
            if pts < -180:
                below_180.append(name)
        
        print(f"\nTOTAL: {total:.1f}pts")
        print(f"Negative: {neg_contracts}")
        print(f"Below -180: {below_180}")
        print(f"TOTAL >= 12164: {'PASS' if total >= 12164 else 'FAIL'}")
        print(f"All neg > -180: {'PASS' if not below_180 else 'FAIL'}")
        
        full_result = {
            "setting": best_setting,
            "total_pts": total,
            "neg_contracts": neg_contracts,
            "below_180": below_180,
            "pass": total >= 12164 and not below_180
        }
        with open(PROJECT / "experiments/iter13/s14_full13.json", 'w') as f:
            json.dump(full_result, f, indent=2)

if __name__ == "__main__":
    main()
