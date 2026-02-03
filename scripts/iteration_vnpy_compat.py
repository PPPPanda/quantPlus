"""iteration_vnpy_compat.py — vnpy-compatible delayed execution model.

Core difference from iteration_v6.py:
  - Orders placed on bar_N fill on bar_N+1 (matching vnpy's new_bar() flow)
  - Buy limit fills at min(order_price, next_bar.open)
  - Sell limit fills at max(order_price, next_bar.open)
  - 3S exits also deferred to next bar

Usage (Windows):
  cd E:\\clawdbot_bridge\\clawdbot_workspace\\work\\quant\\quantPlus
  .venv\\Scripts\\python.exe scripts/iteration_vnpy_compat.py
"""

from __future__ import annotations
from pathlib import Path
from collections import deque
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Import shared utilities from iteration_v6
sys.path.insert(0, str(Path(__file__).parent))
from iteration_v6 import StrategyParams, load_csv, calc_stats, signal_breakdown


class ChanPivotTesterVnpyCompat:
    """vnpy-compatible backtest tester with delayed order execution.

    Key difference from ChanPivotTesterV7:
      - Entry/exit orders fill on the NEXT 1m bar (not the same bar)
      - Fill prices match vnpy's cross_limit_order logic:
        * Buy limit: crosses if order_price >= next_bar.low, fill at min(order_price, next_bar.open)
        * Sell limit: crosses if order_price <= next_bar.high, fill at max(order_price, next_bar.open)
    """

    def __init__(self, df_1m: pd.DataFrame, p: StrategyParams):
        self.p = p
        self.df_1m = df_1m.reset_index(drop=True)
        self.trades: list[dict] = []

        # Position state
        self.position: int = 0
        self.entry_price: float = 0.0
        self.entry_time = None
        self.stop_price: float = 0.0
        self.trailing_active: bool = False
        self.entry_signal_type: str = ""

        # Chan theory state
        self.k_lines: list[dict] = []
        self.inclusion_dir: int = 0
        self.bi_points: list[dict] = []
        self.pivots: list[dict] = []
        self.pending_signal: dict | None = None
        self._recent_diffs: list[float] = []

        # vnpy-compat: pending orders (fill on next bar)
        self._pending_entry_order: dict | None = None   # {'type','price','stop_base','signal_type'}
        self._pending_exit_order: dict | None = None     # {'price','direction','signal_type'}
        self._sig_exit_request: dict | None = None       # 3S exit from _sig()

        # Pre-compute 5m and 15m data
        df_idx = self.df_1m.set_index("datetime")
        df_idx.index = pd.to_datetime(df_idx.index)
        self.df_5m = (
            df_idx.resample("5min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        self._calc()

    def _calc(self):
        """Pre-compute indicators on 5m data."""
        df = self.df_5m
        e1 = df["close"].ewm(span=12, adjust=False).mean()
        e2 = df["close"].ewm(span=26, adjust=False).mean()
        df["diff"] = e1 - e2
        df["dea"] = df["diff"].ewm(span=9, adjust=False).mean()

        df_15m = df.resample("15min", closed="right", label="right").agg({"close": "last"}).dropna()
        m1 = df_15m["close"].ewm(span=12, adjust=False).mean()
        m2 = df_15m["close"].ewm(span=26, adjust=False).mean()
        m = m1 - m2
        s = m.ewm(span=9, adjust=False).mean()
        al = pd.DataFrame({"diff": m, "dea": s}).shift(1).reindex(df.index, method="ffill")
        df["diff_15m"] = al["diff"]
        df["dea_15m"] = al["dea"]

        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()

        bb_std = df["close"].rolling(20).std()
        df["bb_width"] = (4 * bb_std) / df["close"] * 100

        plus_dm = df["high"].diff().clip(lower=0)
        minus_dm = (-df["low"].diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * plus_dm.rolling(14).sum() / tr14
        minus_di = 100 * minus_dm.rolling(14).sum() / tr14
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        df["adx"] = dx.rolling(14).mean()

    # ================================================================
    # Main loop — vnpy-compatible timing
    # ================================================================
    def run(self) -> pd.DataFrame:
        for _, row in self.df_1m.iterrows():
            ct = pd.to_datetime(row["datetime"])

            # === PHASE 1: Cross pending orders (like vnpy's cross_limit_order at start of new_bar) ===
            if self._pending_exit_order is not None:
                if self._try_cross_exit(row, ct):
                    self._pending_exit_order = None

            if self._pending_entry_order is not None:
                if self._try_cross_entry(row, ct):
                    self._pending_entry_order = None

            # === PHASE 2: Strategy logic (like vnpy's strategy.on_bar) ===
            # 2a. Stop loss check → creates pending exit order
            if self.position != 0 and self._pending_exit_order is None:
                self._check_exit_create_order(row, ct)

            # 2b. Entry check → creates pending entry order
            if self.position == 0 and self.pending_signal and self._pending_entry_order is None:
                self._check_entry_create_order(row)

            # 2c. 5m bar logic (signal generation, trailing stop)
            if ct.minute % 5 == 0 and ct in self.df_5m.index:
                bar = self.df_5m.loc[ct]
                self._on_bar(bar)

                # Handle 3S exit request from _sig()
                if self._sig_exit_request is not None and self._pending_exit_order is None:
                    self._pending_exit_order = self._sig_exit_request
                    self._sig_exit_request = None

                if self.position != 0 and self._pending_exit_order is None:
                    self._trailing(bar)

        return pd.DataFrame(self.trades)

    # ================================================================
    # Order crossing (vnpy-compatible)
    # ================================================================
    def _try_cross_exit(self, row, ct) -> bool:
        """Try to fill pending exit order using current bar's OHLC."""
        order = self._pending_exit_order
        if order is None:
            return False

        if order["direction"] == 1:
            # Closing a long position → sell limit
            # vnpy: sell crosses if order.price <= bar.high
            if order["price"] <= row["high"]:
                fill_price = max(order["price"], float(row["open"]))
                pnl = fill_price - self.entry_price
                self.trades.append({
                    "entry_time": self.entry_time, "exit_time": ct,
                    "direction": 1, "entry": self.entry_price, "exit": fill_price,
                    "pnl": pnl, "signal_type": order["signal_type"]
                })
                self.position = 0
                self.trailing_active = False
                return True
        elif order["direction"] == -1:
            # Closing a short position → buy limit (cover)
            # vnpy: buy crosses if order.price >= bar.low
            if order["price"] >= row["low"]:
                fill_price = min(order["price"], float(row["open"]))
                pnl = self.entry_price - fill_price
                self.trades.append({
                    "entry_time": self.entry_time, "exit_time": ct,
                    "direction": -1, "entry": self.entry_price, "exit": fill_price,
                    "pnl": pnl, "signal_type": order["signal_type"]
                })
                self.position = 0
                self.trailing_active = False
                return True
        return False

    def _try_cross_entry(self, row, ct) -> bool:
        """Try to fill pending entry order using current bar's OHLC."""
        order = self._pending_entry_order
        if order is None:
            return False

        # Check invalidation first: if stop_base is breached, cancel order
        if order["type"] == "buy":
            if float(row["low"]) < order["stop_base"]:
                return True  # cancel (return True to clear)
            # Buy limit: crosses if order.price >= bar.low
            if order["price"] >= float(row["low"]):
                fill_price = min(order["price"], float(row["open"]))
                self._open(1, fill_price, ct, order["stop_base"], order["signal_type"])
                return True
        elif order["type"] == "sell":
            if float(row["high"]) > order["stop_base"]:
                return True  # cancel
            # Sell limit: crosses if order.price <= bar.high
            if order["price"] <= float(row["high"]):
                fill_price = max(order["price"], float(row["open"]))
                self._open(-1, fill_price, ct, order["stop_base"], order["signal_type"])
                return True
        return False

    # ================================================================
    # Order creation (deferred, not immediate fill)
    # ================================================================
    def _check_exit_create_order(self, row, ct):
        """Check stop loss → create pending exit order (NOT immediate fill)."""
        if self.position == 1 and float(row["low"]) <= self.stop_price:
            exit_px = float(row["open"]) if float(row["open"]) < self.stop_price else self.stop_price
            self._pending_exit_order = {
                "direction": 1, "price": exit_px,
                "signal_type": self.entry_signal_type + "_SL"
            }
        elif self.position == -1 and float(row["high"]) >= self.stop_price:
            exit_px = float(row["open"]) if float(row["open"]) > self.stop_price else self.stop_price
            self._pending_exit_order = {
                "direction": -1, "price": exit_px,
                "signal_type": self.entry_signal_type + "_SL"
            }

    def _check_entry_create_order(self, row):
        """Check entry trigger → create pending entry order (NOT immediate fill)."""
        s = self.pending_signal
        if not s:
            return

        if s["type"] == "Buy":
            if float(row["low"]) < s["stop_base"]:
                self.pending_signal = None
                return
            if float(row["high"]) > s["trigger_price"]:
                fill = max(s["trigger_price"], float(row["open"]))
                if fill > float(row["high"]):
                    fill = float(row["close"])
                self._pending_entry_order = {
                    "type": "buy", "price": fill,
                    "stop_base": s["stop_base"], "signal_type": s["signal_type"]
                }
                self.pending_signal = None
        elif s["type"] == "Sell":
            if float(row["high"]) > s["stop_base"]:
                self.pending_signal = None
                return
            if float(row["low"]) < s["trigger_price"]:
                fill = min(s["trigger_price"], float(row["open"]))
                if fill < float(row["low"]):
                    fill = float(row["close"])
                self._pending_entry_order = {
                    "type": "sell", "price": fill,
                    "stop_base": s["stop_base"], "signal_type": s["signal_type"]
                }
                self.pending_signal = None

    def _open(self, d, px, t, sb, st):
        """Open position (called when entry order fills)."""
        self.position = d
        self.entry_price = float(px)
        self.entry_time = t
        self.entry_signal_type = st
        self.stop_price = float(sb - 1 if d == 1 else sb + 1)
        self.trailing_active = False

    # ================================================================
    # Trailing stop (same logic, runs on 5m bars)
    # ================================================================
    def _trailing(self, bar):
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0
        if atr <= 0:
            return
        pnl = (float(bar["close"]) - self.entry_price) * self.position
        if not self.trailing_active and pnl > self.p.activate_atr * atr:
            self.trailing_active = True
        if self.trailing_active:
            if self.position == 1:
                n = float(bar["high"]) - self.p.trail_atr * atr
                if n > self.stop_price:
                    self.stop_price = n
            else:
                n = float(bar["low"]) + self.p.trail_atr * atr
                if n < self.stop_price:
                    self.stop_price = n

    # ================================================================
    # Chan theory: inclusion, bi, pivot, signal (same logic as V7)
    # ================================================================
    def _on_bar(self, bar):
        b = {
            "high": float(bar["high"]), "low": float(bar["low"]), "time": bar.name,
            "diff": float(bar["diff"]) if not np.isnan(bar["diff"]) else 0,
            "dea": float(bar["dea"]) if not np.isnan(bar["dea"]) else 0,
            "atr": float(bar["atr"]) if not np.isnan(bar["atr"]) else 0,
            "diff_15m": float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0,
            "dea_15m": float(bar["dea_15m"]) if not np.isnan(bar["dea_15m"]) else 0,
            "close": float(bar["close"]),
            "ma5": float(bar["ma5"]) if not np.isnan(bar["ma5"]) else 0,
            "ma20": float(bar["ma20"]) if not np.isnan(bar["ma20"]) else 0,
            "ma60": float(bar["ma60"]) if not np.isnan(bar["ma60"]) else 0,
            "bb_width": float(bar["bb_width"]) if not np.isnan(bar["bb_width"]) else 0,
            "adx": float(bar["adx"]) if not np.isnan(bar["adx"]) else 0,
        }
        self._recent_diffs.append(b["diff"])
        if len(self._recent_diffs) > 10:
            self._recent_diffs.pop(0)

        self._incl(b)
        if self._bi(b):
            self._sig(bar, b)

    def _incl(self, nb):
        if not self.k_lines:
            self.k_lines.append(nb)
            return
        last = self.k_lines[-1]
        il = nb["high"] <= last["high"] and nb["low"] >= last["low"]
        inw = last["high"] <= nb["high"] and last["low"] >= nb["low"]
        if il or inw:
            if self.inclusion_dir == 0:
                self.inclusion_dir = 1
            m = last.copy()
            for k in ["time", "diff", "dea", "atr", "diff_15m", "dea_15m",
                       "close", "ma5", "ma20", "ma60", "bb_width", "adx"]:
                if k in nb:
                    m[k] = nb[k]
            if self.inclusion_dir == 1:
                m["high"] = max(last["high"], nb["high"])
                m["low"] = max(last["low"], nb["low"])
            else:
                m["high"] = min(last["high"], nb["high"])
                m["low"] = min(last["low"], nb["low"])
            self.k_lines[-1] = m
        else:
            if nb["high"] > last["high"] and nb["low"] > last["low"]:
                self.inclusion_dir = 1
            elif nb["high"] < last["high"] and nb["low"] < last["low"]:
                self.inclusion_dir = -1
            self.k_lines.append(nb)

    def _bi(self, current_bar):
        if len(self.k_lines) < 3:
            return None
        c, m2, l = self.k_lines[-1], self.k_lines[-2], self.k_lines[-3]
        cand = None
        if m2["high"] > l["high"] and m2["high"] > c["high"]:
            cand = {"type": "top", "price": m2["high"], "idx": len(self.k_lines) - 2, "data": m2}
        elif m2["low"] < l["low"] and m2["low"] < c["low"]:
            cand = {"type": "bottom", "price": m2["low"], "idx": len(self.k_lines) - 2, "data": m2}
        if not cand:
            return None
        if not self.bi_points:
            self.bi_points.append(cand)
            return None
        last = self.bi_points[-1]
        if last["type"] == cand["type"]:
            if last["type"] == "top" and cand["price"] > last["price"]:
                self.bi_points[-1] = cand
            elif last["type"] == "bottom" and cand["price"] < last["price"]:
                self.bi_points[-1] = cand
        else:
            if self.p.bi_amp_filter:
                amp = abs(cand["price"] - last["price"])
                atr = current_bar.get("atr", 0)
                if atr > 0 and amp < self.p.bi_amp_min_atr * atr:
                    return None
            if cand["idx"] - last["idx"] >= self.p.min_bi_gap:
                self.bi_points.append(cand)
                return cand
        return None

    def _upd_piv(self):
        if len(self.bi_points) < 4:
            return
        pts = self.bi_points[-4:]
        ranges = [(min(pts[i]["price"], pts[i + 1]["price"]),
                    max(pts[i]["price"], pts[i + 1]["price"])) for i in range(3)]
        zg = min(r[1] for r in ranges)
        zd = max(r[0] for r in ranges)
        if zg > zd:
            self.pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points) - 1})

    def _check_v6_filter(self, direction, signal_type, bar_data):
        p = self.p
        is_buy = direction == "Buy"
        is_2b2s = signal_type in ("2B", "2S")
        if p.filter_2b_only and not is_2b2s:
            return True
        if p.filter_buy_only and not is_buy:
            return True
        close = bar_data["close"]
        ma20 = bar_data["ma20"]
        ma60 = bar_data["ma60"]
        ma5 = bar_data["ma5"]
        if p.ma20_filter:
            if is_buy and close < ma20:
                return False
            if not is_buy and close > ma20:
                return False
        if p.ma60_filter:
            if is_buy and close < ma60:
                return False
            if not is_buy and close > ma60:
                return False
        if p.ma_align_filter:
            if is_buy and not (ma5 > ma20 > ma60 > 0):
                return False
            if not is_buy and not (0 < ma5 < ma20 < ma60):
                return False
        if p.bb_width_cap > 0:
            if bar_data["bb_width"] > p.bb_width_cap:
                return False
        if p.adx_cap > 0:
            if bar_data["adx"] > p.adx_cap:
                return False
        if p.macd15_mag_cap_atr > 0:
            atr = bar_data["atr"]
            if atr > 0:
                mag = abs(bar_data["diff_15m"]) / atr
                if mag > p.macd15_mag_cap_atr:
                    return False
        return True

    def _sig(self, bar, b):
        """Signal generation — same logic as V7, but 3S exit is deferred."""
        self._upd_piv()
        if len(self.bi_points) < 5:
            return
        atr = b["atr"]
        if atr <= 0:
            return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]

        bull = b["diff_15m"] > b["dea_15m"]
        bear = b["diff_15m"] < b["dea_15m"]

        if self.p.trend_filter:
            if b["diff"] <= 0:
                bull = False
            if b["diff"] >= 0:
                bear = False

        if self.p.macd_consistency > 0 and len(self._recent_diffs) >= self.p.macd_consistency:
            recent = self._recent_diffs[-self.p.macd_consistency:]
            if bull and not all(d > 0 for d in recent):
                bull = False
            if bear and not all(d < 0 for d in recent):
                bear = False

        lp = self.pivots[-1] if self.pivots else None
        sig = st = None
        trig = sb = 0.0

        if lp:
            # 3B
            if self.p.enable_3b and pn["type"] == "bottom" and pn["price"] > lp["zg"] and pl["price"] > lp["zg"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bull:
                    if self._check_v6_filter("Buy", "3B", b):
                        sig = "Buy"
                        st = "3B"
                        trig = float(pn["data"]["high"])
                        sb = float(pn["price"])

            # 3S
            if not sig and self.p.enable_3s and pn["type"] == "top" and pn["price"] < lp["zd"] and pl["price"] < lp["zd"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bear:
                    if self.p.disable_3s_short:
                        if self.position == 1:
                            # DEFERRED: create exit request instead of immediate close
                            exit_px = float(pn["data"]["low"])
                            self._sig_exit_request = {
                                "direction": 1, "price": exit_px,
                                "signal_type": self.entry_signal_type + "_3S_exit"
                            }
                    else:
                        if self._check_v6_filter("Sell", "3S", b):
                            sig = "Sell"
                            st = "3S"
                            trig = float(pn["data"]["low"])
                            sb = float(pn["price"])

        # 2B/2S
        if not sig and self.p.enable_2b2s:
            if pn["type"] == "bottom":
                if pn["price"] > pp["price"] and float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bull:
                    if self._check_v6_filter("Buy", "2B", b):
                        sig = "Buy"
                        st = "2B"
                        trig = float(pn["data"]["high"])
                        sb = float(pn["price"])
            elif pn["type"] == "top":
                if pn["price"] < pp["price"] and float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bear:
                    if self._check_v6_filter("Sell", "2S", b):
                        sig = "Sell"
                        st = "2S"
                        trig = float(pn["data"]["low"])
                        sb = float(pn["price"])

        if not sig:
            return
        if abs(trig - sb) >= self.p.entry_filter_atr * atr:
            return
        self.pending_signal = {"type": sig, "trigger_price": trig, "stop_base": sb, "signal_type": st}


# ================================================================
# Runner utilities
# ================================================================
def run_one(cfg_name, cfg, data_dir, contracts):
    """Run one configuration across all contracts."""
    p = StrategyParams(cfg_name, **cfg)
    cr = {}
    total = 0
    mn = float("inf")
    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches:
            continue
        df = load_csv(matches[0])
        tester = ChanPivotTesterVnpyCompat(df, p)
        trades = tester.run()
        s = calc_stats(trades)
        sb = signal_breakdown(trades)
        cr[c.upper()] = {**s, "signals": sb}
        total += s["pnl"]
        if s["pnl"] < mn:
            mn = s["pnl"]
    a800 = sum(1 for _, r in cr.items() if r["pnl"] >= 800)
    all_pos = all(r["pnl"] > 0 for r in cr.values())
    return {
        "name": cfg_name, "cfg": cfg, "contracts": cr, "total_pnl": total,
        "min_pnl": mn, "above_800": a800, "all_positive": all_pos,
        "score": mn + total / 7
    }


def build_configs():
    """Build full config grid (same as iteration_v6.py)."""
    v4 = {
        "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
        "min_bi_gap": 5, "trend_filter": True, "disable_3s_short": True,
        "bi_amp_filter": True, "bi_amp_min_atr": 1.5, "macd_consistency": 3
    }
    configs = {}

    # === Reference (baseline) ===
    # Original baseline (no iter3+ filters)
    configs["REF_baseline"] = {
        "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
        "min_bi_gap": 4
    }
    # v4best (iter3-5 best)
    configs["REF_v4best"] = v4.copy()

    # === S1: MA20 Position Filter ===
    configs["S1_ma20_all"] = {**v4, "ma20_filter": True}
    configs["S1_ma20_2b_only"] = {**v4, "ma20_filter": True, "filter_2b_only": True}
    configs["S1_ma20_buy_only"] = {**v4, "ma20_filter": True, "filter_buy_only": True}

    # === S2: MA60 Position Filter ===
    configs["S2_ma60_all"] = {**v4, "ma60_filter": True}
    configs["S2_ma60_2b_only"] = {**v4, "ma60_filter": True, "filter_2b_only": True}
    configs["S2_ma60_buy_only"] = {**v4, "ma60_filter": True, "filter_buy_only": True}

    # === S3: MA Alignment ===
    configs["S3_align_all"] = {**v4, "ma_align_filter": True}
    configs["S3_align_2b_only"] = {**v4, "ma_align_filter": True, "filter_2b_only": True}

    # === S4: BB Width Cap ===
    for cap in [1.0, 1.2, 1.5, 2.0]:
        configs[f"S4_bbw_{cap}"] = {**v4, "bb_width_cap": cap}
    configs["S4_bbw_1.2_2b"] = {**v4, "bb_width_cap": 1.2, "filter_2b_only": True}
    configs["S4_bbw_1.5_2b"] = {**v4, "bb_width_cap": 1.5, "filter_2b_only": True}

    # === S5: ADX Cap ===
    for cap in [35, 40, 45, 50]:
        configs[f"S5_adx_{cap}"] = {**v4, "adx_cap": cap}
    configs["S5_adx_45_2b"] = {**v4, "adx_cap": 45, "filter_2b_only": True}

    # === S6: MACD15 Magnitude Cap ===
    for cap in [1.5, 2.0, 2.5, 3.0]:
        configs[f"S6_macd15_{cap}"] = {**v4, "macd15_mag_cap_atr": cap}
    configs["S6_macd15_2.0_2b"] = {**v4, "macd15_mag_cap_atr": 2.0, "filter_2b_only": True}

    # === S7: Combinations ===
    configs["S7_ma20_bbw1.5"] = {**v4, "ma20_filter": True, "bb_width_cap": 1.5}
    configs["S7_ma20_bbw1.2"] = {**v4, "ma20_filter": True, "bb_width_cap": 1.2}
    configs["S7_ma20_adx45"] = {**v4, "ma20_filter": True, "adx_cap": 45}
    configs["S7_ma20_adx50"] = {**v4, "ma20_filter": True, "adx_cap": 50}
    configs["S7_ma20_macd15_2.0"] = {**v4, "ma20_filter": True, "macd15_mag_cap_atr": 2.0}
    configs["S7_ma20_macd15_2.5"] = {**v4, "ma20_filter": True, "macd15_mag_cap_atr": 2.5}
    configs["S7_ma60buy_ma20sell"] = {**v4, "ma60_filter": True}
    configs["S7_ma20_2b_adx45"] = {**v4, "ma20_filter": True, "filter_2b_only": True, "adx_cap": 45}
    configs["S7_ma20_bbw1.5_adx50"] = {**v4, "ma20_filter": True, "bb_width_cap": 1.5, "adx_cap": 50}
    configs["S7_ma20_bbw1.5_adx45"] = {**v4, "ma20_filter": True, "bb_width_cap": 1.5, "adx_cap": 45}
    configs["S7_ma20_macd15_2.0_bbw1.5"] = {**v4, "ma20_filter": True, "macd15_mag_cap_atr": 2.0, "bb_width_cap": 1.5}

    # === Grid around best ===
    for act in [1.0, 1.5, 2.0]:
        for trail in [2.0, 2.5, 3.0]:
            for ent in [1.0, 1.5, 2.0]:
                name = f"G_ma20_a{act}_t{trail}_e{ent}"
                configs[name] = {
                    **v4,
                    "ma20_filter": True,
                    "activate_atr": act, "trail_atr": trail, "entry_filter_atr": ent
                }

    return configs


def _print(*args, **kwargs):
    """Print with flush for Windows compatibility."""
    print(*args, **kwargs, flush=True)


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]

    configs = build_configs()
    _print(f"Running {len(configs)} configurations across {len(contracts)} contracts...")
    _print("(vnpy-compatible delayed execution model)")
    _print()

    results = []
    for i, (cfg_name, cfg) in enumerate(configs.items()):
        r = run_one(cfg_name, cfg, data_dir, contracts)
        results.append(r)
        if (i + 1) % 20 == 0:
            _print(f"  ... {i + 1}/{len(configs)} done")

    results.sort(key=lambda x: x["score"], reverse=True)

    # Print top results
    _print(f"\n{'=' * 110}")
    _print(f"TOP 20 RESULTS (sorted by score = min_pnl + total/7)")
    _print(f"{'=' * 110}")
    for r in results[:20]:
        ap = "Y" if r["all_positive"] else "N"
        _print(f"\n{'=' * 90}")
        _print(f"{r['name']} | Score: {r['score']:.0f} | Total: {r['total_pnl']:.0f} | "
              f"Min: {r['min_pnl']:.0f} | >=800: {r['above_800']}/7 | AllPos: {ap}")
        _print(f"{'=' * 90}")
        for c in ["P2201", "P2205", "P2401", "P2405", "P2505", "P2509", "P2601"]:
            cr = r["contracts"].get(c, {})
            pnl = cr.get("pnl", 0)
            t = cr.get("trades", 0)
            w = cr.get("win%", 0)
            d = cr.get("maxdd", 0)
            pf = cr.get("pf", 0)
            m = "OK" if pnl >= 800 else ("+" if pnl > 0 else "-")
            sigs = cr.get("signals", {})
            ss = " | ".join(f"{k}:{v['pnl']:.0f}({v['n']})" for k, v in sorted(sigs.items()))
            _print(f"  {m} {c}: pnl={pnl:>7.0f} t={t:>3} w={w:>5.1f}% dd={d:>6.0f} pf={pf:.2f} | {ss}")

    # Full ranking
    _print(f"\n{'=' * 150}")
    _print("FULL RANKING (vnpy-compat model)")
    _print(f"{'=' * 150}")
    for i, r in enumerate(results):
        ap = "Y" if r["all_positive"] else "N"
        cs = " ".join(f"{c[-4:]}:{r['contracts'].get(c, {}).get('pnl', 0):>6.0f}"
                      for c in ["P2201", "P2205", "P2401", "P2405", "P2505", "P2509", "P2601"])
        _print(f"#{i + 1:>2} {ap} {r['name']:<35} sc={r['score']:>7.0f} tot={r['total_pnl']:>7.0f} "
              f"min={r['min_pnl']:>7.0f} p={r['above_800']}/7 | {cs}")

    # Save results
    out_dir = Path("experiments") / (datetime.now().strftime("%Y%m%d_%H%M") + "_vnpy_compat")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        for r in results:
            for c, cr in r["contracts"].items():
                for k, v in list(cr.items()):
                    if isinstance(v, (np.floating, np.integer)):
                        cr[k] = float(v)
        json.dump(results, f, indent=2, default=str)
    _print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
