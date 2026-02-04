"""诊断 p2501 交易类型分布 + 同区域入场频率."""
from __future__ import annotations
import sys, logging
from pathlib import Path
from datetime import timedelta
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from vnpy.trader.constant import Interval
logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange
from zoneinfo import ZoneInfo

CN_TZ = ZoneInfo("Asia/Shanghai")

csv_path = ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv"
vt_symbol = "p2501.DCE"

# Import
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

# Monkey-patch to capture trade details
trade_log = []
orig_open = CtaChanPivotStrategy._open_position

def patched_open(self, direction, price, stop_base):
    trade_log.append({
        'bar_count': self.bar_count,
        'price': price,
        'signal_type': self._signal_type,
        'direction': direction,
        'pivot_zg': self._active_pivot['zg'] if self._active_pivot else None,
        'pivot_zd': self._active_pivot['zd'] if self._active_pivot else None,
        'pivot_state': self._active_pivot['state'] if self._active_pivot else None,
        'pivot_entry_count': self._active_pivot.get('entry_count', 0) if self._active_pivot else None,
    })
    return orig_open(self, direction, price, stop_base)

CtaChanPivotStrategy._open_position = patched_open

setting = {
    "debug_enabled": False, "debug_log_console": False,
    "cooldown_losses": 2, "cooldown_bars": 20,
    "atr_activate_mult": 2.5, "atr_trailing_mult": 3.0,
    "atr_entry_filter": 2.0, "max_pivot_entries": 2,
}

result = run_backtest(
    vt_symbol=vt_symbol,
    start=start - timedelta(days=1),
    end=end + timedelta(days=1),
    strategy_class=CtaChanPivotStrategy,
    strategy_setting=setting,
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

# Analyze
print(f"Total trades: {len(trade_log)}")
type_counts = Counter(t['signal_type'] for t in trade_log)
print(f"Signal type distribution: {dict(type_counts)}")

state_counts = Counter(t['pivot_state'] for t in trade_log)
print(f"Pivot state at entry: {dict(state_counts)}")

entry_counts = Counter(t['pivot_entry_count'] for t in trade_log)
print(f"Pivot entry_count at entry: {dict(entry_counts)}")

# Check price clustering
if trade_log:
    prices = [t['price'] for t in trade_log]
    # Check how many trades within ATR of each other
    close_pairs = 0
    for i in range(1, len(trade_log)):
        dist = abs(trade_log[i]['price'] - trade_log[i-1]['price'])
        bar_gap = trade_log[i]['bar_count'] - trade_log[i-1]['bar_count']
        if dist < 100 and bar_gap < 50:  # within 100 pts and 50 5m bars
            close_pairs += 1
    print(f"Close-proximity pairs (price<100pts, gap<50 bars): {close_pairs}/{len(trade_log)-1}")

    # Show first 20 trades
    print("\nFirst 20 trades:")
    for t in trade_log[:20]:
        print(f"  bar={t['bar_count']:5d} price={t['price']:8.0f} sig={t['signal_type']} "
              f"pv_state={t['pivot_state']} pv_ec={t['pivot_entry_count']} "
              f"ZG={t['pivot_zg']:.0f if t['pivot_zg'] else 'N/A'} ZD={t['pivot_zd']:.0f if t['pivot_zd'] else 'N/A'}")
