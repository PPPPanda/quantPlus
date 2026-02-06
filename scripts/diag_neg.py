"""
诊断负收益合约: p2401, p2201 的失败模式分析.
"""
from __future__ import annotations
import json, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from vnpy.trader.constant import Interval
import logging
logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

TARGETS = [
    {"contract": "p2401.DCE", "csv": ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv"},
    {"contract": "p2201.DCE", "csv": ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv"},
    {"contract": "p2305.DCE", "csv": ROOT / "data/analyse/wind/p2305_1min_202212-202304.csv"},
    {"contract": "p2309.DCE", "csv": ROOT / "data/analyse/wind/p2309_1min_202304-202308.csv"},
]

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

SETTING = {
    "debug_enabled": False, "debug_log_console": False,
    "cooldown_losses": 2, "cooldown_bars": 20,
    "circuit_breaker_losses": 7, "circuit_breaker_bars": 70,
    "atr_activate_mult": 2.5, "atr_trailing_mult": 3.0, "atr_entry_filter": 2.0,
    "div_threshold": 0.39,
}


def import_csv_to_db(csv_path, vt_symbol):
    from vnpy.trader.database import get_database
    from vnpy.trader.object import BarData
    from vnpy.trader.constant import Exchange
    from zoneinfo import ZoneInfo
    CN_TZ = ZoneInfo("Asia/Shanghai")
    db = get_database()
    symbol, exchange_str = vt_symbol.split(".")
    exchange = Exchange(exchange_str)
    db.delete_bar_data(symbol, exchange, Interval.MINUTE)
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
    df.sort_values("datetime", inplace=True)
    df.drop_duplicates(subset=["datetime"], keep="first", inplace=True)
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(CN_TZ)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(CN_TZ)
    bars = []
    for _, row in df.iterrows():
        dt = row["datetime"]
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        bar = BarData(
            symbol=symbol, exchange=Exchange(exchange_str), datetime=dt,
            interval=Interval.MINUTE,
            volume=float(row.get("volume", 0)),
            turnover=float(row.get("turnover", 0)),
            open_interest=float(row.get("open_interest", 0)),
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            gateway_name="DB",
        )
        bars.append(bar)
    db.save_bar_data(bars)
    start = df["datetime"].min().to_pydatetime()
    end = df["datetime"].max().to_pydatetime()
    
    # Also compute price stats
    prices = df["close"].values
    price_range = prices.max() - prices.min()
    daily_returns = df.groupby(df["datetime"].dt.date)["close"].last().pct_change().dropna()
    
    return start, end, len(bars), {
        "price_min": float(prices.min()),
        "price_max": float(prices.max()),
        "price_range": float(price_range),
        "price_range_pct": float(price_range / prices.mean() * 100),
        "daily_vol": float(daily_returns.std() * 100),
        "mean_price": float(prices.mean()),
    }


def analyze(bench):
    vt_symbol = bench["contract"]
    name = vt_symbol.split(".")[0]
    csv_path = bench["csv"]
    
    start, end, bar_count, price_stats = import_csv_to_db(csv_path, vt_symbol)
    
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=SETTING,
        **BT_PARAMS,
    )
    
    trades = result.trades if result.trades else []
    
    # Pair trades
    paired = []
    open_trade = None
    for t in trades:
        if t.offset.value == '开':
            open_trade = t
        elif open_trade is not None:
            pnl = (t.price - open_trade.price) * 10 if open_trade.direction.value == '多' else (open_trade.price - t.price) * 10
            entry_dt = open_trade.datetime.replace(tzinfo=None) if open_trade.datetime.tzinfo else open_trade.datetime
            exit_dt = t.datetime.replace(tzinfo=None) if t.datetime.tzinfo else t.datetime
            hold_mins = (exit_dt - entry_dt).total_seconds() / 60
            
            paired.append({
                'entry_dt': entry_dt,
                'exit_dt': exit_dt,
                'entry_price': open_trade.price,
                'exit_price': t.price,
                'pnl': pnl,
                'pnl_pts': pnl / 10,
                'hold_mins': hold_mins,
                'direction': open_trade.direction.value,
            })
            open_trade = None
    
    # Analysis
    total_pnl = sum(t['pnl'] for t in paired)
    winners = [t for t in paired if t['pnl'] > 0]
    losers = [t for t in paired if t['pnl'] < 0]
    
    print(f"\n{'='*70}")
    print(f"=== {name} ===")
    print(f"Price stats: mean={price_stats['mean_price']:.0f}, range={price_stats['price_range']:.0f} ({price_stats['price_range_pct']:.1f}%), daily_vol={price_stats['daily_vol']:.2f}%")
    print(f"Total trades: {len(paired)}, PnL: {total_pnl:.0f} ({total_pnl/10:.1f} pts)")
    print(f"Winners: {len(winners)}, Losers: {len(losers)}, WR: {len(winners)/max(len(paired),1)*100:.0f}%")
    
    if winners:
        print(f"Avg win: {np.mean([t['pnl'] for t in winners]):.0f}, Max win: {max(t['pnl'] for t in winners):.0f}")
    if losers:
        print(f"Avg loss: {np.mean([t['pnl'] for t in losers]):.0f}, Max loss: {min(t['pnl'] for t in losers):.0f}")
    
    # Hold time
    if paired:
        hold_times = [t['hold_mins'] for t in paired]
        print(f"Hold: median={np.median(hold_times):.0f}min, mean={np.mean(hold_times):.0f}min")
    
    # Cumulative PnL trajectory
    cum_pnl = 0
    print(f"\nCumulative PnL trajectory (first/last 10 trades):")
    for i, t in enumerate(paired):
        cum_pnl += t['pnl']
        if i < 10 or i >= len(paired) - 10:
            marker = "***" if t['pnl'] < -300 else ""
            print(f"  #{i+1:3d} {str(t['entry_dt'])[:16]} -> {str(t['exit_dt'])[:16]}: pnl={t['pnl']:>6.0f} cum={cum_pnl:>7.0f} hold={t['hold_mins']:>5.0f}m {marker}")
        elif i == 10:
            print(f"  ... ({len(paired)-20} more trades) ...")
    
    # Loss cluster analysis
    loss_clusters = []
    current_cluster = []
    for t in paired:
        if t['pnl'] < 0:
            current_cluster.append(t)
        else:
            if len(current_cluster) >= 2:
                loss_clusters.append(current_cluster)
            current_cluster = []
    if len(current_cluster) >= 2:
        loss_clusters.append(current_cluster)
    
    print(f"\nLoss clusters (>=2 consecutive): {len(loss_clusters)}")
    for i, cluster in enumerate(sorted(loss_clusters, key=lambda c: sum(t['pnl'] for t in c))):
        cl_pnl = sum(t['pnl'] for t in cluster)
        print(f"  Cluster: {len(cluster)} trades, PnL={cl_pnl:.0f}")
        for t in cluster:
            print(f"    {str(t['entry_dt'])[:16]}: pnl={t['pnl']:.0f} hold={t['hold_mins']:.0f}m")
        if i >= 4:  # top 5 worst clusters
            break
    
    # Win/loss pattern by time of day
    morning = [t for t in paired if t['entry_dt'].hour < 12]
    afternoon = [t for t in paired if 12 <= t['entry_dt'].hour < 16]
    night = [t for t in paired if t['entry_dt'].hour >= 21]
    print(f"\nBy session: AM({len(morning)})={sum(t['pnl'] for t in morning):.0f}, PM({len(afternoon)})={sum(t['pnl'] for t in afternoon):.0f}, Night({len(night)})={sum(t['pnl'] for t in night):.0f}")
    
    # Short hold trades (<30 min)
    short_trades = [t for t in paired if t['hold_mins'] < 30]
    if short_trades:
        short_pnl = sum(t['pnl'] for t in short_trades)
        print(f"Short-hold (<30m): {len(short_trades)} trades, PnL={short_pnl:.0f}")
    
    return {
        'contract': name,
        'total_pnl': total_pnl,
        'total_pts': total_pnl / 10,
        'trades': len(paired),
        'winners': len(winners),
        'losers': len(losers),
        'loss_clusters': len(loss_clusters),
        'price_stats': price_stats,
    }


def main():
    all_results = {}
    for bench in TARGETS:
        name = bench["contract"].split(".")[0]
        print(f"\nAnalyzing {name}...", flush=True)
        r = analyze(bench)
        all_results[name] = r
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    for name, r in all_results.items():
        print(f"  {name}: pts={r['total_pts']:.1f}, trades={r['trades']}, WR={r['winners']/max(r['trades'],1)*100:.0f}%, clusters={r['loss_clusters']}")
        print(f"    price: range={r['price_stats']['price_range_pct']:.1f}%, vol={r['price_stats']['daily_vol']:.2f}%")


if __name__ == "__main__":
    main()
