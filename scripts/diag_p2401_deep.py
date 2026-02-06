"""
p2401 深度诊断脚本 — 逐笔交易分析
目标：找出为什么即使在上涨月份(8月+640pts)策略也亏钱
"""
import sys, json, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS
from run_13bench import import_csv_to_db
import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

def run_p2401_diag():
    csv_path = PROJECT / "data/analyse/wind/p2401_1min_202308-202312.csv"
    vt_symbol = "p2401.DCE"
    
    # Import data
    start, end, _ = import_csv_to_db(str(csv_path), vt_symbol)
    
    # Run backtest with default params
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start, end=end,
        strategy_class=CtaChanPivotStrategy,
        strategy_setting={
            "debug_enabled": False,
            "debug_log_console": False,
        },
        interval=Interval.MINUTE,
        rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0,
        capital=1_000_000,
    )
    
    stats = result.stats
    trades = result.trades
    print(f"=== p2401 Backtest Summary ===")
    print(f"Total PnL: {stats.get('total_net_pnl', 0):.0f} ({stats.get('total_net_pnl', 0)/10:.1f} pts)")
    print(f"Trades: {stats.get('total_count', 0)}")
    print(f"Win rate: {stats.get('winning_rate', 0):.1f}%")
    print(f"Max DD: {stats.get('max_ddpercent', 0):.2f}%")
    print(f"Sharpe: {stats.get('sharpe_ratio', 0):.2f}")
    
    # Analyze individual trades
    if not trades:
        print("No trades!")
        return
    
    print(f"\n=== Trade-by-Trade Analysis ({len(trades)} trades) ===")
    
    trade_pairs = []
    opens = []
    for t in trades:
        if t.offset.value == "Open" or t.offset.value == "开":
            opens.append(t)
        elif t.offset.value == "Close" or t.offset.value == "平":
            if opens:
                o = opens.pop(0)
                if o.direction.value in ("Long", "多"):
                    pnl = (t.price - o.price) * 10  # size=10
                else:
                    pnl = (o.price - t.price) * 10
                trade_pairs.append({
                    'open_time': o.datetime,
                    'close_time': t.datetime,
                    'direction': 'LONG' if o.direction.value in ("Long", "多") else 'SHORT',
                    'open_price': o.price,
                    'close_price': t.price,
                    'pnl_raw': pnl,
                    'pts': (t.price - o.price) if o.direction.value in ("Long", "多") else (o.price - t.price),
                    'hold_minutes': (t.datetime - o.datetime).total_seconds() / 60,
                })
    
    print(f"Paired trades: {len(trade_pairs)}")
    
    # Monthly breakdown
    monthly = {}
    for tp in trade_pairs:
        m = tp['open_time'].strftime('%Y-%m')
        if m not in monthly:
            monthly[m] = {'trades': 0, 'wins': 0, 'pnl': 0, 'details': []}
        monthly[m]['trades'] += 1
        monthly[m]['pnl'] += tp['pts']
        if tp['pts'] > 0:
            monthly[m]['wins'] += 1
        monthly[m]['details'].append(tp)
    
    print(f"\n=== Monthly Breakdown ===")
    for m in sorted(monthly.keys()):
        d = monthly[m]
        wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        wins = [t['pts'] for t in d['details'] if t['pts'] > 0]
        losses = [t['pts'] for t in d['details'] if t['pts'] <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_hold = np.mean([t['hold_minutes'] for t in d['details']])
        print(f"{m}: {d['trades']}t, wr={wr:.0f}%, pnl={d['pnl']:+.0f}pts, avgWin={avg_win:+.0f}, avgLoss={avg_loss:+.0f}, avgHold={avg_hold:.0f}min")
    
    # Direction analysis
    longs = [t for t in trade_pairs if t['direction'] == 'LONG']
    shorts = [t for t in trade_pairs if t['direction'] == 'SHORT']
    print(f"\n=== Direction Analysis ===")
    print(f"LONG: {len(longs)} trades, pnl={sum(t['pts'] for t in longs):+.0f}pts")
    if shorts:
        print(f"SHORT: {len(shorts)} trades, pnl={sum(t['pts'] for t in shorts):+.0f}pts")
    else:
        print(f"SHORT: 0 trades (strategy is long-only)")
    
    # Holding time analysis
    print(f"\n=== Holding Time Distribution ===")
    holds = [t['hold_minutes'] for t in trade_pairs]
    for bucket, label in [(10, '<=10min'), (30, '<=30min'), (60, '<=1h'), (180, '<=3h'), (float('inf'), '>3h')]:
        subset = [t for t in trade_pairs if t['hold_minutes'] <= bucket and t['hold_minutes'] > (bucket/3 if bucket > 10 else 0)]
        # Better: use ranges
        pass
    
    bins = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 300), (300, float('inf'))]
    prev_upper = 0
    for lo, hi in bins:
        subset = [t for t in trade_pairs if lo <= t['hold_minutes'] < hi]
        if subset:
            pnl = sum(t['pts'] for t in subset)
            wr = sum(1 for t in subset if t['pts'] > 0) / len(subset) * 100
            print(f"  {lo}-{hi if hi != float('inf') else 'inf'}min: {len(subset)}t, pnl={pnl:+.0f}pts, wr={wr:.0f}%")
    
    # Consecutive loss analysis
    print(f"\n=== Consecutive Loss Streaks ===")
    streaks = []
    curr_streak = 0
    curr_streak_pnl = 0
    for t in trade_pairs:
        if t['pts'] <= 0:
            curr_streak += 1
            curr_streak_pnl += t['pts']
        else:
            if curr_streak > 0:
                streaks.append((curr_streak, curr_streak_pnl))
            curr_streak = 0
            curr_streak_pnl = 0
    if curr_streak > 0:
        streaks.append((curr_streak, curr_streak_pnl))
    
    streaks.sort(key=lambda x: x[0], reverse=True)
    for i, (n, pnl) in enumerate(streaks[:5]):
        print(f"  Streak {i+1}: {n} consecutive losses, total={pnl:+.0f}pts")
    
    # Top 10 worst trades
    print(f"\n=== Top 10 Worst Trades ===")
    worst = sorted(trade_pairs, key=lambda t: t['pts'])[:10]
    for t in worst:
        print(f"  {t['open_time'].strftime('%m-%d %H:%M')} -> {t['close_time'].strftime('%m-%d %H:%M')}: "
              f"{t['direction']} {t['open_price']:.0f}->{t['close_price']:.0f} = {t['pts']:+.0f}pts, hold={t['hold_minutes']:.0f}min")
    
    # Top 5 best trades
    print(f"\n=== Top 5 Best Trades ===")
    best = sorted(trade_pairs, key=lambda t: t['pts'], reverse=True)[:5]
    for t in best:
        print(f"  {t['open_time'].strftime('%m-%d %H:%M')} -> {t['close_time'].strftime('%m-%d %H:%M')}: "
              f"{t['direction']} {t['open_price']:.0f}->{t['close_price']:.0f} = {t['pts']:+.0f}pts, hold={t['hold_minutes']:.0f}min")
    
    # PnL by price level
    print(f"\n=== PnL by Price Level (entry price) ===")
    for lo, hi in [(6700, 7200), (7200, 7500), (7500, 7800), (7800, 8000)]:
        subset = [t for t in trade_pairs if lo <= t['open_price'] < hi]
        if subset:
            pnl = sum(t['pts'] for t in subset)
            wr = sum(1 for t in subset if t['pts'] > 0) / len(subset) * 100
            print(f"  {lo}-{hi}: {len(subset)}t, pnl={pnl:+.0f}pts, wr={wr:.0f}%")
    
    # Save detailed results
    output = {
        'summary': {
            'total_pnl_pts': sum(t['pts'] for t in trade_pairs),
            'trades': len(trade_pairs),
            'win_rate': sum(1 for t in trade_pairs if t['pts'] > 0) / len(trade_pairs) * 100,
            'avg_win': float(np.mean([t['pts'] for t in trade_pairs if t['pts'] > 0])) if any(t['pts'] > 0 for t in trade_pairs) else 0,
            'avg_loss': float(np.mean([t['pts'] for t in trade_pairs if t['pts'] <= 0])) if any(t['pts'] <= 0 for t in trade_pairs) else 0,
        },
        'monthly': {m: {'trades': d['trades'], 'wins': d['wins'], 'pnl_pts': d['pnl']} for m, d in monthly.items()},
        'trades': [{
            'open_time': str(t['open_time']),
            'close_time': str(t['close_time']),
            'direction': t['direction'],
            'open_price': t['open_price'],
            'close_price': t['close_price'],
            'pts': t['pts'],
            'hold_minutes': t['hold_minutes'],
        } for t in trade_pairs],
    }
    
    out_path = PROJECT / "experiments/iter13/p2401_trades_detail.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to {out_path}")

if __name__ == "__main__":
    run_p2401_diag()
