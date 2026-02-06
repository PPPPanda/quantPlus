"""
Iter13: 验证"做空干扰多头"假设
对比有/无做空时的多头交易时间线，找出被做空"挤占"的多头入场
"""
import sys, json
from pathlib import Path
from datetime import datetime

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

def get_trade_pairs(result):
    """Extract trade pairs with direction info"""
    trades = result.trades or []
    pairs = []
    opens = []
    for t in trades:
        if t.offset.value in ("Open", "\u5f00"):
            opens.append(t)
        elif t.offset.value in ("Close", "\u5e73"):
            if opens:
                o = opens.pop(0)
                direction = "LONG" if o.direction.value in ("Long", "\u591a") else "SHORT"
                if direction == "LONG":
                    pnl = t.price - o.price
                else:
                    pnl = o.price - t.price
                pairs.append({
                    "dir": direction,
                    "open_time": str(o.datetime),
                    "close_time": str(t.datetime),
                    "open_price": o.price,
                    "close_price": t.price,
                    "pts": pnl,
                })
    return pairs

def run_and_get_pairs(csv_path, vt_symbol, extra_setting):
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
    pairs = get_trade_pairs(result)
    return pts, pairs

# Test on 3 contracts where shorting hurts
TESTS = [
    ("p2201", "p2201.DCE", str(PROJECT / "data/analyse/wind/p2201_1min_202108-202112.csv")),
    ("p2601", "p2601.DCE", str(PROJECT / "data/analyse/p2601_1min_202507-202512.csv")),
    ("p2401", "p2401.DCE", str(PROJECT / "data/analyse/wind/p2401_1min_202308-202312.csv")),
]

SHORT_SETTING = {"conditional_short": True, "short_min_pivots": 2, "short_max_diff_15m": 0}

def main():
    print("=== SHORTING INTERFERENCE ANALYSIS ===\n")
    
    all_analysis = {}
    for name, sym, csv in TESTS:
        b_pts, b_pairs = run_and_get_pairs(csv, sym, {})
        s_pts, s_pairs = run_and_get_pairs(csv, sym, SHORT_SETTING)
        
        b_longs = [p for p in b_pairs if p['dir'] == 'LONG']
        s_longs = [p for p in s_pairs if p['dir'] == 'LONG']
        s_shorts = [p for p in s_pairs if p['dir'] == 'SHORT']
        
        b_long_total = sum(p['pts'] for p in b_longs)
        s_long_total = sum(p['pts'] for p in s_longs)
        s_short_total = sum(p['pts'] for p in s_shorts)
        
        print(f"=== {name}: baseline={b_pts:+.1f} → short={s_pts:+.1f} (delta={s_pts-b_pts:+.1f}) ===")
        print(f"  Baseline: {len(b_longs)} longs, raw={b_long_total:+.0f}pts")
        print(f"  Short:    {len(s_longs)} longs (raw={s_long_total:+.0f}), {len(s_shorts)} shorts (raw={s_short_total:+.0f})")
        
        # Find which baseline longs were LOST due to shorting
        # (longs that existed in baseline but don't exist in short version)
        b_long_opens = set(p['open_time'][:16] for p in b_longs)
        s_long_opens = set(p['open_time'][:16] for p in s_longs)
        
        lost_longs = b_long_opens - s_long_opens
        new_longs = s_long_opens - b_long_opens
        
        # Find profitable longs that were lost
        lost_profitable = [p for p in b_longs if p['open_time'][:16] in lost_longs and p['pts'] > 0]
        lost_unprofitable = [p for p in b_longs if p['open_time'][:16] in lost_longs and p['pts'] <= 0]
        
        lost_profit_pts = sum(p['pts'] for p in lost_profitable)
        lost_loss_pts = sum(p['pts'] for p in lost_unprofitable)
        
        print(f"\n  Lost longs (existed in baseline, missing with shorts): {len(lost_longs)}")
        print(f"    Profitable ones lost: {len(lost_profitable)}, missed profit: {lost_profit_pts:+.0f}pts")
        print(f"    Unprofitable ones lost: {len(lost_unprofitable)}, avoided loss: {lost_loss_pts:+.0f}pts")
        print(f"    Net from lost longs: {lost_profit_pts + lost_loss_pts:+.0f}pts (positive=net loss)")
        
        print(f"  New longs (only with shorts): {len(new_longs)}")
        
        # Short trade details
        print(f"\n  Short trades detail:")
        short_wins = sum(1 for p in s_shorts if p['pts'] > 0)
        short_losses = sum(1 for p in s_shorts if p['pts'] <= 0)
        if s_shorts:
            print(f"    Win/Loss: {short_wins}/{short_losses}")
            print(f"    Avg win: {sum(p['pts'] for p in s_shorts if p['pts'] > 0) / max(short_wins, 1):+.0f}")
            print(f"    Avg loss: {sum(p['pts'] for p in s_shorts if p['pts'] <= 0) / max(short_losses, 1):+.0f}")
            for p in s_shorts:
                print(f"    {p['open_time'][5:16]}->{p['close_time'][5:16]}: {p['open_price']:.0f}->{p['close_price']:.0f} = {p['pts']:+.0f}pts")
        
        print()
        
        all_analysis[name] = {
            "baseline_pts": b_pts, "short_pts": s_pts, "delta": s_pts - b_pts,
            "baseline_longs": len(b_longs), "short_longs": len(s_longs), "short_shorts": len(s_shorts),
            "lost_profitable_longs": len(lost_profitable), "missed_profit": lost_profit_pts,
            "short_raw_pnl": s_short_total,
        }
    
    with open(PROJECT / "experiments/iter13/short_interference.json", 'w') as f:
        json.dump(all_analysis, f, indent=2)
    print("Saved to experiments/iter13/short_interference.json")

if __name__ == "__main__":
    main()
