"""
Iter13 Phase 2 深度诊断：做空在每个合约上到底做了什么？
逐合约分析做空带来的具体交易变化
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

def run_with_trades(csv_path, vt_symbol, extra_setting):
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
    pts = result.stats.get('total_net_pnl', 0) / 10
    trades = result.trades or []
    
    # Pair trades
    longs, shorts = 0, 0
    long_pnl, short_pnl = 0, 0
    opens = []
    for t in trades:
        if t.offset.value in ("Open", "\u5f00"):
            opens.append(t)
        elif t.offset.value in ("Close", "\u5e73"):
            if opens:
                o = opens.pop(0)
                if o.direction.value in ("Long", "\u591a"):
                    pnl = (t.price - o.price)
                    longs += 1
                    long_pnl += pnl
                else:
                    pnl = (o.price - t.price)
                    shorts += 1
                    short_pnl += pnl
    
    return pts, longs, long_pnl, shorts, short_pnl

# Focus on contracts that worsened most with shorting
FOCUS = [
    ("p2201", "p2201.DCE", str(PROJECT / "data/analyse/wind/p2201_1min_202108-202112.csv")),
    ("p2601", "p2601.DCE", str(PROJECT / "data/analyse/p2601_1min_202507-202512.csv")),
    ("p2501", "p2501.DCE", str(PROJECT / "data/analyse/wind/p2501_1min_202410-202501.csv")),
    ("p2401", "p2401.DCE", str(PROJECT / "data/analyse/wind/p2401_1min_202308-202312.csv")),
    ("p2405", "p2405.DCE", str(PROJECT / "data/analyse/wind/p2405_1min_202312-202404.csv")),
    ("p2205", "p2205.DCE", str(PROJECT / "data/analyse/wind/p2205_1min_202112-202204.csv")),
]

def main():
    print("=== WHY SHORTING HURTS: Per-Contract Long/Short Breakdown ===\n")
    
    print(f"{'Contract':<8} | {'--- Baseline ---':^32} | {'--- With Short (p2) ---':^38} |")
    print(f"{'':8} | {'PnL':>8} {'Longs':>6} {'L_PnL':>8} {'Shorts':>6} {'S_PnL':>8} | {'PnL':>8} {'Longs':>6} {'L_PnL':>8} {'Shorts':>6} {'S_PnL':>8} |")
    print("-" * 100)
    
    results = {}
    for name, sym, csv in FOCUS:
        # Baseline
        b_pts, b_l, b_lpnl, b_s, b_spnl = run_with_trades(csv, sym, {})
        # With shorting
        s_pts, s_l, s_lpnl, s_s, s_spnl = run_with_trades(csv, sym, 
            {"conditional_short": True, "short_min_pivots": 2, "short_max_diff_15m": 0})
        
        print(f"{name:<8} | {b_pts:>+8.1f} {b_l:>6} {b_lpnl:>+8.0f} {b_s:>6} {b_spnl:>+8.0f} | {s_pts:>+8.1f} {s_l:>6} {s_lpnl:>+8.0f} {s_s:>6} {s_spnl:>+8.0f} |")
        
        results[name] = {
            "baseline": {"pts": b_pts, "longs": b_l, "long_pnl": b_lpnl, "shorts": b_s, "short_pnl": b_spnl},
            "short_p2": {"pts": s_pts, "longs": s_l, "long_pnl": s_lpnl, "shorts": s_s, "short_pnl": s_spnl},
        }
    
    print("\n=== ANALYSIS ===")
    for name, data in results.items():
        b = data['baseline']
        s = data['short_p2']
        long_delta = s['long_pnl'] - b['long_pnl']
        print(f"\n{name}:")
        print(f"  Short trades added: {s['shorts']} (raw PnL: {s['short_pnl']:+.0f}pts)")
        print(f"  Long trade change: {b['longs']}->{s['longs']} trades, PnL delta: {long_delta:+.0f}pts")
        print(f"  Net effect: {s['pts'] - b['pts']:+.1f}pts")
        if s['short_pnl'] < 0:
            print(f"  >>> SHORT TRADES ARE LOSING MONEY!")
        if long_delta < 0:
            print(f"  >>> SHORTING ALSO HURTS LONG TRADES (interference)!")
    
    with open(PROJECT / "experiments/iter13/short_why_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/iter13/short_why_analysis.json")

if __name__ == "__main__":
    main()
