"""
Iter14 S18: 价格通道位置过滤器测试
先在 p2401 上搜索最优参数，再全量 13 合约验证
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from run_13bench import import_csv_to_db, BENCHMARKS

PROJECT = Path(__file__).resolve().parents[1]

BASE = {
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
}

def run_single(csv, sym, extra):
    s = {**BASE, **extra, "debug_enabled": False, "debug_log_console": False}
    start, end, _ = import_csv_to_db(csv, sym)
    r = run_backtest(vt_symbol=sym, start=start, end=end,
        strategy_class=CtaChanPivotStrategy, strategy_setting=s,
        interval=Interval.MINUTE, rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000)
    return r.stats.get('total_net_pnl', 0) / 10

P2401 = next(b for b in BENCHMARKS if "p2401" in b["contract"])

# Grid: channel_period x channel_max_pct
PERIODS = [60, 120, 180, 240]  # 5h, 10h, 15h, 20h
MAX_PCTS = [0.50, 0.60, 0.70, 0.80]

def main():
    print("=== S18 Channel Filter: p2401 Grid Search ===")
    print(f"{'period':>6} x {'max_pct':>7} = {'p2401':>8} {'improve':>8}")
    print("-" * 40)
    
    base_pts = run_single(P2401["csv"], P2401["contract"], {})
    print(f"{'base':>6}   {'':>7}   {base_pts:>+8.1f}   {'---':>8}")
    
    results = []
    for period in PERIODS:
        for max_pct in MAX_PCTS:
            pts = run_single(P2401["csv"], P2401["contract"], 
                {"channel_period": period, "channel_max_pct": max_pct})
            improve = pts - base_pts
            results.append({"period": period, "max_pct": max_pct, "pts": pts, "improve": improve})
            print(f"{period:>6}   {max_pct:>7.2f}   {pts:>+8.1f}   {improve:>+8.1f}")
    
    # Top 5
    results.sort(key=lambda x: x['pts'], reverse=True)
    top5 = results[:5]
    print(f"\nTop 5:")
    for r in top5:
        print(f"  period={r['period']}, max_pct={r['max_pct']}: {r['pts']:+.1f} ({r['improve']:+.1f})")
    
    # Also test on guardian contracts (p2209, p2601)
    P2209 = next(b for b in BENCHMARKS if "p2209" in b["contract"])
    P2601 = next(b for b in BENCHMARKS if "p2601" in b["contract"])
    
    print(f"\n=== Guardian Contract Check (top 3) ===")
    for r in top5[:3]:
        extra = {"channel_period": r['period'], "channel_max_pct": r['max_pct']}
        p2209 = run_single(P2209["csv"], P2209["contract"], extra)
        p2601 = run_single(P2601["csv"], P2601["contract"], extra)
        p2209_base = run_single(P2209["csv"], P2209["contract"], {})
        p2601_base = run_single(P2601["csv"], P2601["contract"], {})
        print(f"  ch={r['period']}/{r['max_pct']}: p2209={p2209:+.1f}(base={p2209_base:+.1f}), p2601={p2601:+.1f}(base={p2601_base:+.1f})")
    
    with open(PROJECT / "experiments/iter14/s18_grid.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
