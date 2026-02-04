"""
诊断脚本：导出指定合约的逐笔交易明细.
用法：cd quantPlus && .venv/Scripts/python.exe scripts/diag_trades.py
"""
from __future__ import annotations
import sys, logging, time, json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from vnpy.trader.constant import Interval

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

BT_PARAMS = dict(
    interval=Interval.MINUTE, rate=0.0001, slippage=1.0,
    size=10.0, pricetick=2.0, capital=1_000_000.0,
)

SETTING = {
    "debug_enabled": True, "debug_log_console": False,
    "atr_activate_mult": 2.5, "atr_trailing_mult": 3.0,
    "cooldown_losses": 4, "cooldown_bars": 30,
}

CONTRACTS = [
    ("p2501.DCE", ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv"),
    ("p2505.DCE", ROOT / "data/analyse/wind/p2505_1min_202412-202504.csv"),
    ("p2509.DCE", ROOT / "data/analyse/wind/p2509_1min_202504-202508.csv"),
    # 加上亏损的非25年合约做对比
    ("p2301.DCE", ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv"),
    ("p2401.DCE", ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv"),
]


def import_csv_to_db(csv_path, vt_symbol):
    from vnpy.trader.database import get_database
    from vnpy.trader.object import BarData
    from vnpy.trader.constant import Exchange
    from zoneinfo import ZoneInfo
    db = get_database()
    CN_TZ = ZoneInfo("Asia/Shanghai")
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


for vt_symbol, csv_path in CONTRACTS:
    start, end, bar_count = import_csv_to_db(csv_path, vt_symbol)
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=SETTING,
        **BT_PARAMS,
    )
    stats = result.stats or {}
    trades = result.trades or []
    
    print(f"\n{'='*70}")
    print(f"{vt_symbol}: PnL={stats.get('total_net_pnl',0):.0f}  trades={stats.get('total_trade_count',0)}  "
          f"commission={stats.get('total_commission',0):.0f}  slippage={stats.get('total_slippage',0):.0f}")
    print(f"  win_rate={stats.get('winning_rate',0):.1f}%  avg_pnl={stats.get('average_trade_pnl',0):.0f}")
    
    # 分析交易
    wins = losses = 0
    win_pnl = loss_pnl = 0
    trade_pairs = []
    open_trade = None
    
    for t in trades:
        if t.offset.value == "开":
            open_trade = t
        elif open_trade:
            if t.direction.value == "空":  # 卖平 = 平多
                pnl = (t.price - open_trade.price) * BT_PARAMS['size'] * open_trade.volume
            else:
                pnl = (open_trade.price - t.price) * BT_PARAMS['size'] * open_trade.volume
            commission = (open_trade.price + t.price) * BT_PARAMS['size'] * open_trade.volume * BT_PARAMS['rate']
            net_pnl = pnl - commission
            trade_pairs.append({
                'open_dt': str(open_trade.datetime),
                'close_dt': str(t.datetime),
                'open_px': open_trade.price,
                'close_px': t.price,
                'gross_pnl': round(pnl, 0),
                'commission': round(commission, 0),
                'net_pnl': round(net_pnl, 0),
            })
            if net_pnl > 0:
                wins += 1
                win_pnl += net_pnl
            else:
                losses += 1
                loss_pnl += net_pnl
            open_trade = None
    
    total = wins + losses
    print(f"  配对交易: {total}笔, 盈{wins}笔(+{win_pnl:.0f}), 亏{losses}笔({loss_pnl:.0f})")
    if wins > 0:
        print(f"  平均盈利: +{win_pnl/wins:.0f}  平均亏损: {loss_pnl/losses:.0f}" if losses > 0 else f"  平均盈利: +{win_pnl/wins:.0f}")
    
    # 显示最大亏损交易
    if trade_pairs:
        worst = sorted(trade_pairs, key=lambda x: x['net_pnl'])[:5]
        print(f"  --- 最大亏损 5 笔 ---")
        for tp in worst:
            print(f"    {tp['open_dt'][:16]} -> {tp['close_dt'][:16]}  "
                  f"open={tp['open_px']:.0f} close={tp['close_px']:.0f}  "
                  f"gross={tp['gross_pnl']:.0f} fee={tp['commission']:.0f} net={tp['net_pnl']:.0f}")
        
        best = sorted(trade_pairs, key=lambda x: x['net_pnl'], reverse=True)[:3]
        print(f"  --- 最大盈利 3 笔 ---")
        for tp in best:
            print(f"    {tp['open_dt'][:16]} -> {tp['close_dt'][:16]}  "
                  f"open={tp['open_px']:.0f} close={tp['close_px']:.0f}  "
                  f"gross={tp['gross_pnl']:.0f} fee={tp['commission']:.0f} net={tp['net_pnl']:.0f}")
