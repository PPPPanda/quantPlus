"""Diagnostic v3: Start from FULL BASELINE, add ONE change at a time.

The BaselineLikeStrategy from diag_live_safety.py gives correct results.
Now add each live-safety change individually to find the culprit.
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
from datetime import timedelta
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from vnpy.trader.constant import Interval
from vnpy.trader.object import BarData, TradeData, OrderData
from vnpy.trader.constant import Offset, Status, Direction
from vnpy_ctastrategy.base import StopOrder

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

# Import the verified baseline
from diag_live_safety import BaselineLikeStrategy


# === Change 1: ONLY add position reconciliation block ===
class Change1_Recon(BaselineLikeStrategy):
    """Baseline + position reconciliation block."""
    def on_bar(self, bar):
        bar_dict = {
            'datetime': bar.datetime, 'open': bar.open_price,
            'high': bar.high_price, 'low': bar.low_price,
            'close': bar.close_price, 'volume': bar.volume,
        }
        if self._debugger and self.trading:
            self._debugger.log_kline_1m(bar_dict)

        # ADDED: position reconciliation
        if self.trading and not self._pending_open and not self._pending_close:
            if self.pos == 0 and self._position != 0:
                self.write_log(f"⚠️ 仓位对账: pos=0, _position={self._position}")
                self._position = 0
                self._trailing_active = False
                self._pending_signal = None
                self.signal = "仓位同步"

        if self.trading and self._position != 0:
            if self._check_stop_loss_1m(bar_dict):
                self.put_event()
                return
        if self.trading and self._pending_signal:
            sig_type = self._pending_signal.get('type', '')
            if (sig_type == 'Buy' and self._position == 0) or \
               (sig_type == 'CloseLong' and self._position == 1) or \
               (sig_type == 'Sell' and self._position == 0):
                self._check_entry_1m(bar_dict)
        bar_15m = self._update_15m_bar(bar_dict)
        if bar_15m: self._on_15m_bar(bar_15m)
        bar_5m = self._update_5m_bar(bar_dict)
        if bar_5m: self._on_5m_bar(bar_5m)
        self.put_event()


# === Change 2: ONLY add _pending_close=True + guard in _check_stop_loss_1m ===
class Change2_PendingClose(BaselineLikeStrategy):
    """Baseline + _pending_close flag in stop loss."""
    def _check_stop_loss_1m(self, bar):
        # ADDED: guard
        if self._pending_close:
            return False
        sl_hit = False
        exit_price = 0.0
        effective_stop = self._stop_price
        if self.min_hold_bars > 0 and self._bars_since_entry <= self.min_hold_bars:
            stop_dist = abs(self._entry_price - self._stop_price)
            if self._position == 1: effective_stop = self._entry_price - stop_dist * 2
            elif self._position == -1: effective_stop = self._entry_price + stop_dist * 2
        if self._position == 1:
            if bar['low'] <= effective_stop:
                sl_hit = True
                exit_price = bar['open'] if bar['open'] < effective_stop else effective_stop
        elif self._position == -1:
            if bar['high'] >= effective_stop:
                sl_hit = True
                exit_price = bar['open'] if bar['open'] > effective_stop else effective_stop
        if sl_hit:
            if self._position == 1:
                pnl = exit_price - self._entry_price
                self.sell(exit_price, abs(self.pos))
                action = "CLOSE_LONG"
            else:
                pnl = self._entry_price - exit_price
                self.cover(exit_price, abs(self.pos))
                action = "CLOSE_SHORT"
            if self._debugger and self.trading:
                self._debugger.log_trade(action=action, price=exit_price, volume=self.fixed_volume, position=0, pnl=pnl, signal_type=self._signal_type)
            self._position = 0
            self._trailing_active = False
            self._pending_close = True  # ADDED
            self.signal = "止损"
            self._update_loss_streak(pnl)
            return True
        return False

    def on_trade(self, trade: TradeData):
        """Need to clear _pending_close on fill."""
        self.write_log(f"成交: {trade.direction.value} {trade.offset.value} {trade.volume}手 @ {trade.price:.0f}")
        if trade.offset in (Offset.CLOSE, Offset.CLOSETODAY, Offset.CLOSEYESTERDAY):
            if self._pending_close:
                self._pending_close = False
        self.sync_data()
        self.put_event()


# === Change 3: ONLY add _pending_open guard + flag in _open_position ===
class Change3_PendingOpen(BaselineLikeStrategy):
    """Baseline + _pending_open flag in open position."""
    def _open_position(self, direction, price, stop_base):
        # ADDED: guard
        if self._pending_open:
            self.write_log("⚠️ 已有待成交开仓单，跳过新开仓")
            return
        buffer = self._calc_stop_buffer()
        self._pending_open = True  # ADDED
        if direction == 1:
            self.buy(price, self.fixed_volume)
            self._stop_price = stop_base - buffer
            self.signal = "3B/2B买入"
            action = "OPEN_LONG"
        else:
            self.short(price, self.fixed_volume)
            self._stop_price = stop_base + buffer
            self.signal = "3S/2S卖出"
            action = "OPEN_SHORT"
        self._position = direction
        self._entry_price = price
        self._initial_stop = self._stop_price
        self._trailing_active = False
        self._bars_since_entry = 0
        self._pending_signal = None
        if direction == 1:
            if self._active_pivot is not None:
                self._active_pivot['entry_count'] = self._active_pivot.get('entry_count', 0) + 1
            self._recent_entries.append({'price': price, 'bar_5m_count': self._bar_5m_count})
        if self._debugger and self.trading:
            self._debugger.log_trade(action=action, price=price, volume=self.fixed_volume, position=direction, pnl=0, signal_type=self._signal_type)

    def on_trade(self, trade: TradeData):
        """Need to clear _pending_open on fill."""
        self.write_log(f"成交: {trade.direction.value} {trade.offset.value} {trade.volume}手 @ {trade.price:.0f}")
        if trade.offset == Offset.OPEN:
            if self._pending_open:
                self._pending_open = False
        self.sync_data()
        self.put_event()


# === Change 4: ONLY add on_trade manual open detection (entry_price overwrite) ===
class Change4_OnTradeOpen(BaselineLikeStrategy):
    """Baseline + on_trade detection of non-strategy opens."""
    def on_trade(self, trade: TradeData):
        self.write_log(f"成交: {trade.direction.value} {trade.offset.value} {trade.volume}手 @ {trade.price:.0f}")
        if trade.offset == Offset.OPEN:
            # "Non-strategy" open detection — in backtest, _pending_open is always False
            if not self._pending_open:
                self.write_log(f"⚠️ 检测到非策略开仓, 同步状态")
                self._position = 1 if trade.direction == Direction.LONG else -1
                self._entry_price = trade.price  # OVERWRITES entry_price!
        self.sync_data()
        self.put_event()


# === Change 5: ONLY add on_trade manual close detection (kills signal) ===
class Change5_OnTradeClose(BaselineLikeStrategy):
    """Baseline + on_trade detection of non-strategy closes."""
    def on_trade(self, trade: TradeData):
        self.write_log(f"成交: {trade.direction.value} {trade.offset.value} {trade.volume}手 @ {trade.price:.0f}")
        if trade.offset in (Offset.CLOSE, Offset.CLOSETODAY, Offset.CLOSEYESTERDAY):
            # "Non-strategy" close detection — in backtest, _pending_close is always False
            if not self._pending_close and self._position != 0:
                pnl = 0.0
                if self._position == 1: pnl = trade.price - self._entry_price
                elif self._position == -1: pnl = self._entry_price - trade.price
                self._position = 0
                self._trailing_active = False
                self._pending_signal = None  # KILLS SIGNAL!
                self.signal = "手动平仓"
                self._update_loss_streak(pnl)
        self.sync_data()
        self.put_event()


# === Change 6: FULL on_trade rewrite (from current code, no flag setting in other methods) ===
class Change6_FullOnTrade(BaselineLikeStrategy):
    """Baseline + full on_trade rewrite from current code."""
    def on_trade(self, trade: TradeData):
        self.write_log(f"成交: {trade.direction.value} {trade.offset.value} {trade.volume}手 @ {trade.price:.0f}")
        if trade.offset == Offset.OPEN:
            if self._pending_open:
                self._pending_open = False
                self._pre_open_snapshot = None
            else:
                self.write_log(f"⚠️ 检测到非策略开仓, 同步状态")
                self._position = 1 if trade.direction == Direction.LONG else -1
                self._entry_price = trade.price
        elif trade.offset in (Offset.CLOSE, Offset.CLOSETODAY, Offset.CLOSEYESTERDAY):
            if self._pending_close:
                self._pending_close = False
                self._pre_close_snapshot = None
            elif self._position != 0:
                pnl = 0.0
                if self._position == 1: pnl = trade.price - self._entry_price
                elif self._position == -1: pnl = self._entry_price - trade.price
                self._position = 0
                self._trailing_active = False
                self._pending_signal = None
                self.signal = "手动平仓"
                self._update_loss_streak(pnl)
        self.sync_data()
        self.put_event()


CONTRACTS = [
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"},
    {"contract": "p2401.DCE", "csv": ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv"},
]
BT_PARAMS = dict(interval=Interval.MINUTE, rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=100_000.0)
ITER14_SETTING = {
    "debug": False, "debug_enabled": False, "debug_log_console": False,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "atr_window": 14, "atr_trailing_mult": 3.0, "atr_activate_mult": 2.5, "atr_entry_filter": 2.0,
    "min_bi_gap": 4, "pivot_valid_range": 6, "fixed_volume": 1,
    "cooldown_losses": 2, "cooldown_bars": 20, "circuit_breaker_losses": 7, "circuit_breaker_bars": 70,
    "lock_profit_atr": 0.0, "min_hold_bars": 2, "max_pullback_atr": 3.2, "use_bi_trailing": True,
    "stop_buffer_atr_pct": 0.02, "max_pivot_entries": 2, "pivot_reentry_atr": 0.6,
    "dedup_bars": 0, "dedup_atr_mult": 1.5, "div_mode": 1, "div_threshold": 0.39,
    "seg_enabled": False, "hist_gate": 0, "gap_extreme_atr": 0.0,
    "gap_reset_inclusion": False, "bridge_bar_enabled": False,
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
    if df["datetime"].dt.tz is None: df["datetime"] = df["datetime"].dt.tz_localize(CN_TZ)
    else: df["datetime"] = df["datetime"].dt.tz_convert(CN_TZ)
    bars = []
    for _, row in df.iterrows():
        dt = row["datetime"]
        if hasattr(dt, 'to_pydatetime'): dt = dt.to_pydatetime()
        bar = BarData(symbol=symbol, exchange=exchange, datetime=dt, interval=Interval.MINUTE,
            volume=float(row.get("volume", 0)), turnover=float(row.get("turnover", 0)),
            open_interest=float(row.get("open_interest", 0)),
            open_price=float(row["open"]), high_price=float(row["high"]),
            low_price=float(row["low"]), close_price=float(row["close"]), gateway_name="DB")
        bars.append(bar)
    db.save_bar_data(bars)
    return df["datetime"].min().to_pydatetime(), df["datetime"].max().to_pydatetime(), len(bars)


VARIANTS = [
    ("1: recon only", Change1_Recon),
    ("2: pending_close", Change2_PendingClose),
    ("3: pending_open", Change3_PendingOpen),
    ("4: on_trade open", Change4_OnTradeOpen),
    ("5: on_trade close", Change5_OnTradeClose),
    ("6: full on_trade", Change6_FullOnTrade),
]

if __name__ == "__main__":
    print("=" * 80)
    print("ISOLATE v3: Baseline + ONE change at a time")
    print("Baseline: p2209=108, p2401=111")
    print("=" * 80)

    for item in CONTRACTS:
        vt = item["contract"]
        name = vt.split(".")[0]
        csv_path = item["csv"]
        if not csv_path.exists(): continue
        start, end, n = import_csv_to_db(csv_path, vt)
        print(f"\n--- {name} ---")
        for label, cls in VARIANTS:
            r = run_backtest(vt_symbol=vt, start=start - timedelta(days=1), end=end + timedelta(days=1),
                strategy_class=cls, strategy_setting=ITER14_SETTING, **BT_PARAMS)
            s = r.stats or {}
            pnl = round(s.get("total_net_pnl", 0), 1)
            pts = round(pnl / 10, 1)
            trades = s.get("total_trade_count", 0)
            expected = 108 if name == "p2209" else 111
            status = "✅" if trades == expected else f"❌ ({expected - trades:+d})"
            print(f"  {label:>20}: trades={trades:>4}, pts={pts:>+8.1f} {status}")
