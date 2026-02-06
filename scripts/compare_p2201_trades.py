#!/usr/bin/env python
"""对比 p2201 在 S26 开启/关闭时的具体 trades 差异"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from vnpy.trader.constant import Interval
import logging

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

csv_path = ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv"
vt_symbol = "p2201.DCE"

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

BASELINE_SETTING = {
    "debug_enabled": False,
    "cooldown_losses": 2,
    "cooldown_bars": 20,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "atr_entry_filter": 2.0,
    "gap_extreme_atr": 0.0,  # 禁用 S26
    "gap_cooldown_bars": 0,
}

S26_SETTING = {
    **BASELINE_SETTING,
    "gap_extreme_atr": 3.0,  # 启用 S26
    "gap_cooldown_bars": 3,
}


def import_csv_to_db(csv_path: Path, vt_symbol: str):
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
            symbol=symbol, exchange=exchange, datetime=dt,
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


def run_and_get_trades(setting, label):
    start, end, bar_count = import_csv_to_db(csv_path, vt_symbol)
    
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        **BT_PARAMS,
    )
    
    trades = []
    for t in result.trades:
        trades.append({
            "datetime": t.datetime.isoformat() if hasattr(t.datetime, 'isoformat') else str(t.datetime),
            "direction": str(t.direction),
            "offset": str(t.offset),
            "price": t.price,
            "volume": t.volume,
        })
    
    return trades, result.stats


def main():
    print("=" * 80)
    print("p2201 Trades 对比分析")
    print("=" * 80)
    
    print("\n运行基线 (S26 disabled)...")
    baseline_trades, baseline_stats = run_and_get_trades(BASELINE_SETTING, "baseline")
    
    print("\n运行 S26 (enabled)...")
    s26_trades, s26_stats = run_and_get_trades(S26_SETTING, "s26")
    
    print("\n" + "=" * 80)
    print("统计对比")
    print("=" * 80)
    print(f"{'指标':<20} {'基线':>15} {'S26':>15} {'差异':>15}")
    print("-" * 80)
    
    bp = baseline_stats.get("total_net_pnl", 0)
    sp = s26_stats.get("total_net_pnl", 0)
    print(f"{'净盈亏 (RMB)':<20} {bp:>15.2f} {sp:>15.2f} {sp-bp:>+15.2f}")
    
    bt = baseline_stats.get("total_trade_count", 0)
    st = s26_stats.get("total_trade_count", 0)
    print(f"{'交易次数':<20} {bt:>15} {st:>15} {st-bt:>+15}")
    
    bw = baseline_stats.get("winning_rate", 0)
    sw = s26_stats.get("winning_rate", 0)
    print(f"{'胜率 (%)':<20} {bw:>15.2f} {sw:>15.2f} {sw-bw:>+15.2f}")
    
    bd = baseline_stats.get("max_ddpercent", 0)
    sd = s26_stats.get("max_ddpercent", 0)
    print(f"{'最大回撤 (%)':<20} {bd:>15.2f} {sd:>15.2f} {sd-bd:>+15.2f}")
    
    # 找出差异的 trades
    print("\n" + "=" * 80)
    print("Trades 差异分析")
    print("=" * 80)
    
    # 按时间排序
    baseline_set = set(t['datetime'] for t in baseline_trades)
    s26_set = set(t['datetime'] for t in s26_trades)
    
    only_baseline = baseline_set - s26_set
    only_s26 = s26_set - baseline_set
    
    print(f"\n基线独有的 trades (被 S26 过滤): {len(only_baseline)}")
    for dt in sorted(list(only_baseline))[:10]:  # 只显示前10个
        trade = next(t for t in baseline_trades if t['datetime'] == dt)
        print(f"  {dt} | {trade['direction']} {trade['offset']} @ {trade['price']}")
    
    print(f"\nS26 独有的 trades (基线没有): {len(only_s26)}")
    for dt in sorted(list(only_s26))[:10]:
        trade = next(t for t in s26_trades if t['datetime'] == dt)
        print(f"  {dt} | {trade['direction']} {trade['offset']} @ {trade['price']}")
    
    # 分析被过滤的 trades 的盈亏
    if only_baseline:
        print("\n" + "-" * 40)
        print("被 S26 过滤的 trades 详情 (开仓信号):")
        
        # 尝试配对开仓和平仓
        filtered_opens = []
        for dt in sorted(only_baseline):
            trade = next(t for t in baseline_trades if t['datetime'] == dt)
            if 'OPEN' in trade['offset'].upper():
                filtered_opens.append(trade)
        
        print(f"  被过滤的开仓次数: {len(filtered_opens)}")
        for t in filtered_opens[:5]:
            print(f"  {t['datetime']} | {t['direction']} @ {t['price']}")


if __name__ == "__main__":
    main()
