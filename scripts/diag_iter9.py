"""
iter9 诊断：分析交易级别的失败模式.

输出：
1. 每个合约的亏损交易详情
2. 连亏簇分析
3. 跳空/session gap 相关亏损
4. 持仓时间分布
5. 止盈机会分析（最大浮盈 vs 实际盈亏）
"""
from __future__ import annotations

import json
import sys
import time
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

BENCHMARKS = [
    {"contract": "p2601.DCE", "csv": ROOT / "data/analyse/p2601_1min_202507-202512.csv", "source": "XT"},
    {"contract": "p2405.DCE", "csv": ROOT / "data/analyse/wind/p2405_1min_202312-202404.csv", "source": "Wind"},
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv", "source": "Wind"},
    {"contract": "p2501.DCE", "csv": ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv", "source": "Wind"},
    {"contract": "p2505.DCE", "csv": ROOT / "data/analyse/wind/p2505_1min_202412-202504.csv", "source": "Wind"},
    {"contract": "p2509.DCE", "csv": ROOT / "data/analyse/wind/p2509_1min_202504-202508.csv", "source": "Wind"},
    {"contract": "p2301.DCE", "csv": ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv", "source": "Wind"},
]

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

DEFAULT_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "cooldown_losses": 2,
    "cooldown_bars": 20,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "atr_entry_filter": 2.0,
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
    return start, end, len(bars)


def analyze_contract(bench):
    vt_symbol = bench["contract"]
    name = vt_symbol.split(".")[0]
    csv_path = bench["csv"]
    
    start, end, bar_count = import_csv_to_db(csv_path, vt_symbol)
    
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=DEFAULT_SETTING,
        **BT_PARAMS,
    )
    
    trades = result.trades if result.trades else []
    daily_results = result.daily_results if hasattr(result, 'daily_results') else {}
    
    # Pair trades (open/close pairs)
    paired = []
    open_trade = None
    for t in trades:
        if t.offset.value == '开':
            open_trade = t
        elif open_trade is not None:
            pnl = 0
            if open_trade.direction.value == '多':
                pnl = (t.price - open_trade.price) * 10  # size=10
            else:
                pnl = (open_trade.price - t.price) * 10
            
            # Calculate hold duration
            entry_dt = open_trade.datetime
            exit_dt = t.datetime
            if hasattr(entry_dt, 'tzinfo') and entry_dt.tzinfo:
                entry_dt = entry_dt.replace(tzinfo=None)
            if hasattr(exit_dt, 'tzinfo') and exit_dt.tzinfo:
                exit_dt = exit_dt.replace(tzinfo=None)
            
            hold_mins = (exit_dt - entry_dt).total_seconds() / 60
            
            # Check if entry/exit cross session boundaries
            entry_date = entry_dt.date()
            exit_date = exit_dt.date()
            crosses_day = entry_date != exit_date
            
            # Check if entry is in night session
            entry_time = entry_dt.time()
            is_night_entry = entry_time.hour >= 21 or entry_time.hour < 3
            
            paired.append({
                'entry_dt': str(entry_dt),
                'exit_dt': str(exit_dt),
                'entry_price': open_trade.price,
                'exit_price': t.price,
                'direction': open_trade.direction.value,
                'pnl': pnl,
                'pnl_pts': pnl / 10,
                'hold_mins': hold_mins,
                'crosses_day': crosses_day,
                'is_night_entry': is_night_entry,
                'entry_date': str(entry_date),
                'exit_date': str(exit_date),
            })
            open_trade = None
    
    # Analysis
    total_pnl = sum(t['pnl'] for t in paired)
    winners = [t for t in paired if t['pnl'] > 0]
    losers = [t for t in paired if t['pnl'] < 0]
    
    # Loss clusters (consecutive losses)
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
    
    # Overnight holdings (cross-day)
    overnight = [t for t in paired if t['crosses_day']]
    
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"Total trades: {len(paired)}, PnL: {total_pnl:.0f} ({total_pnl/10:.1f} pts)")
    print(f"Winners: {len(winners)}, Losers: {len(losers)}, WR: {len(winners)/max(len(paired),1)*100:.0f}%")
    
    if winners:
        avg_win = np.mean([t['pnl'] for t in winners])
        max_win = max(t['pnl'] for t in winners)
        print(f"Avg win: {avg_win:.0f}, Max win: {max_win:.0f}")
    if losers:
        avg_loss = np.mean([t['pnl'] for t in losers])
        max_loss = min(t['pnl'] for t in losers)
        print(f"Avg loss: {avg_loss:.0f}, Max loss: {max_loss:.0f}")
    
    # Hold time distribution
    if paired:
        hold_times = [t['hold_mins'] for t in paired]
        print(f"\nHold time: median={np.median(hold_times):.0f}min, mean={np.mean(hold_times):.0f}min, max={max(hold_times):.0f}min")
    
    # Overnight analysis
    if overnight:
        on_pnl = sum(t['pnl'] for t in overnight)
        on_losses = [t for t in overnight if t['pnl'] < 0]
        print(f"\nOvernight trades: {len(overnight)}, PnL: {on_pnl:.0f}, Losses: {len(on_losses)}")
        for t in overnight:
            print(f"  {t['entry_dt']} -> {t['exit_dt']}: pnl={t['pnl']:.0f}")
    
    # Loss clusters
    if loss_clusters:
        print(f"\nLoss clusters (>=2 consecutive): {len(loss_clusters)}")
        for i, cluster in enumerate(loss_clusters):
            cl_pnl = sum(t['pnl'] for t in cluster)
            print(f"  Cluster {i+1}: {len(cluster)} trades, PnL={cl_pnl:.0f}")
            for t in cluster:
                print(f"    {t['entry_dt']}: pnl={t['pnl']:.0f} hold={t['hold_mins']:.0f}min")
    
    # Top 5 worst trades
    print(f"\nTop 5 worst trades:")
    sorted_by_pnl = sorted(paired, key=lambda x: x['pnl'])[:5]
    for t in sorted_by_pnl:
        print(f"  {t['entry_dt']} -> {t['exit_dt']}: pnl={t['pnl']:.0f} hold={t['hold_mins']:.0f}min cross_day={t['crosses_day']}")
    
    # Top 5 best trades
    print(f"\nTop 5 best trades:")
    sorted_by_pnl = sorted(paired, key=lambda x: x['pnl'], reverse=True)[:5]
    for t in sorted_by_pnl:
        print(f"  {t['entry_dt']} -> {t['exit_dt']}: pnl={t['pnl']:.0f} hold={t['hold_mins']:.0f}min")
    
    return {
        'contract': name,
        'total_pnl': total_pnl,
        'total_pts': total_pnl / 10,
        'trades': len(paired),
        'winners': len(winners),
        'losers': len(losers),
        'loss_clusters': len(loss_clusters),
        'overnight_trades': len(overnight),
        'overnight_pnl': sum(t['pnl'] for t in overnight) if overnight else 0,
        'paired': paired,
    }


def main():
    all_results = {}
    for bench in BENCHMARKS:
        name = bench["contract"].split(".")[0]
        print(f"\nAnalyzing {name}...", flush=True)
        r = analyze_contract(bench)
        all_results[name] = r
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_overnight_pnl = 0
    for name, r in all_results.items():
        print(f"  {name}: pts={r['total_pts']:.1f}, trades={r['trades']}, WR={r['winners']/max(r['trades'],1)*100:.0f}%, overnight_pnl={r['overnight_pnl']:.0f}")
        total_overnight_pnl += r['overnight_pnl']
    print(f"\n  Total overnight PnL: {total_overnight_pnl:.0f} ({total_overnight_pnl/10:.1f} pts)")
    
    # Save
    output = {k: {kk: vv for kk, vv in v.items() if kk != 'paired'} for k, v in all_results.items()}
    output['_overnight_total_pnl'] = total_overnight_pnl
    Path("experiments/iter9/diag_trades.json").parent.mkdir(parents=True, exist_ok=True)
    with open("experiments/iter9/diag_trades.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: experiments/iter9/diag_trades.json")


if __name__ == "__main__":
    main()
