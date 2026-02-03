"""iteration_v5.py — Phase 4: Implementing GPT-5.2 + Claude combined proposals

Top strategies from combined analysis:
1. Anchored Pivot (merge overlapping pivots into stable anchor)
2. Pivot Overlap Regime (detect choppy vs trending from internal state)  
3. Bi Density adaptive (auto-adjust trigger sensitivity)
4. MACD slope filter (fix lag issue for P2201)
5. Volume confirmation (filter fake breakouts)
"""

from __future__ import annotations
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import json
from datetime import datetime


class StrategyParams:
    def __init__(self, name="TEST", **kwargs):
        self.name = name
        self.activate_atr = kwargs.get("activate_atr", 1.5)
        self.trail_atr = kwargs.get("trail_atr", 3.0)
        self.entry_filter_atr = kwargs.get("entry_filter_atr", 2.0)
        self.pivot_valid_range = kwargs.get("pivot_valid_range", 6)
        self.min_bi_gap = kwargs.get("min_bi_gap", 4)
        self.enable_2b2s = kwargs.get("enable_2b2s", True)
        self.enable_3b = kwargs.get("enable_3b", True)
        self.enable_3s = kwargs.get("enable_3s", True)
        self.disable_3s_short = kwargs.get("disable_3s_short", False)
        self.trend_filter = kwargs.get("trend_filter", False)
        self.bi_amp_filter = kwargs.get("bi_amp_filter", False)
        self.bi_amp_min_atr = kwargs.get("bi_amp_min_atr", 1.0)
        self.macd_consistency = kwargs.get("macd_consistency", 0)
        # v5 new strategies
        self.anchor_pivot = kwargs.get("anchor_pivot", False)           # Strategy 1
        self.anchor_mode = kwargs.get("anchor_mode", "intersect")       # "intersect" or "union"
        self.regime_switch = kwargs.get("regime_switch", False)         # Strategy 2
        self.regime_overlap_thresh = kwargs.get("regime_overlap_thresh", 0.6)
        self.regime_window = kwargs.get("regime_window", 8)
        self.bi_density_adaptive = kwargs.get("bi_density_adaptive", False)  # Strategy 3
        self.density_window_k = kwargs.get("density_window_k", 200)
        self.density_high_thresh = kwargs.get("density_high_thresh", 0.15)  # >15% = noisy
        self.macd_slope_filter = kwargs.get("macd_slope_filter", False)  # Strategy 4
        self.vol_confirm = kwargs.get("vol_confirm", False)             # Strategy 5
        self.vol_mult = kwargs.get("vol_mult", 1.2)
        self.vol_window = kwargs.get("vol_window", 20)
        self.time_stop = kwargs.get("time_stop", 0)  # bars before forced exit if no progress


class ChanPivotTesterV6:
    def __init__(self, df_1m, p):
        self.p = p
        self.df_1m = df_1m.reset_index(drop=True)
        self.trades = []
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_price = 0.0
        self.trailing_active = False
        self.entry_signal_type = ""
        self.hold_bars = 0
        self.k_lines = []
        self.inclusion_dir = 0
        self.bi_points = []
        self.pivots = []
        self.pending_signal = None
        self._recent_diffs = []
        self._recent_vols = deque(maxlen=max(p.vol_window, 20))
        self._bi_timestamps = []  # track when bis happen for density calc

        df_idx = self.df_1m.set_index("datetime")
        df_idx.index = pd.to_datetime(df_idx.index)
        self.df_5m = (
            df_idx.resample("5min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        self._calc()

    def _calc(self):
        df = self.df_5m
        e1 = df["close"].ewm(span=12, adjust=False).mean()
        e2 = df["close"].ewm(span=26, adjust=False).mean()
        df["diff"] = e1 - e2
        df["dea"] = df["diff"].ewm(span=9, adjust=False).mean()

        df_15m = df.resample("15min", closed="right", label="right").agg({"close": "last"}).dropna()
        m1 = df_15m["close"].ewm(span=12, adjust=False).mean()
        m2 = df_15m["close"].ewm(span=26, adjust=False).mean()
        m = m1 - m2; s = m.ewm(span=9, adjust=False).mean()
        al = pd.DataFrame({"diff": m, "dea": s}).shift(1).reindex(df.index, method="ffill")
        df["diff_15m"] = al["diff"]; df["dea_15m"] = al["dea"]

        # MACD slope (for strategy 4)
        if self.p.macd_slope_filter:
            df["diff_15m_prev"] = al["diff"].shift(1)

        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

    def run(self):
        for _, row in self.df_1m.iterrows():
            ct = pd.to_datetime(row["datetime"])
            if self.position != 0:
                self._check_exit(row)
            if self.position == 0 and self.pending_signal:
                self._check_entry(row)
            if ct.minute % 5 == 0 and ct in self.df_5m.index:
                bar = self.df_5m.loc[ct]
                self._on_bar(bar)
                if self.position != 0:
                    self._trailing(bar)
                    self.hold_bars += 1
                    # Time stop
                    if self.p.time_stop > 0 and self.hold_bars > self.p.time_stop:
                        pnl_now = (float(bar["close"]) - self.entry_price) * self.position
                        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0
                        if atr > 0 and pnl_now < 0.5 * atr:
                            # No meaningful progress, exit
                            self._force_exit(float(bar["close"]), bar.name)
        return pd.DataFrame(self.trades)

    def _check_entry(self, row):
        s = self.pending_signal
        if not s: return
        if s["type"] == "Buy":
            if row["low"] < s["stop_base"]: self.pending_signal = None; return
            if row["high"] > s["trigger_price"]:
                fill = max(s["trigger_price"], row["open"])
                if fill > row["high"]: fill = row["close"]
                self._open(1, fill, pd.to_datetime(row["datetime"]), s["stop_base"], s["signal_type"])
        elif s["type"] == "Sell":
            if row["high"] > s["stop_base"]: self.pending_signal = None; return
            if row["low"] < s["trigger_price"]:
                fill = min(s["trigger_price"], row["open"])
                if fill < row["low"]: fill = row["close"]
                self._open(-1, fill, pd.to_datetime(row["datetime"]), s["stop_base"], s["signal_type"])

    def _open(self, d, px, t, sb, st):
        self.position = d; self.entry_price = float(px); self.entry_time = t
        self.entry_signal_type = st
        self.stop_price = float(sb - 1 if d == 1 else sb + 1)
        self.pending_signal = None; self.trailing_active = False; self.hold_bars = 0

    def _check_exit(self, row):
        hit = False; exit_px = 0.0; t = pd.to_datetime(row["datetime"])
        if self.position == 1 and row["low"] <= self.stop_price:
            hit = True; exit_px = float(row["open"]) if row["open"] < self.stop_price else float(self.stop_price)
        elif self.position == -1 and row["high"] >= self.stop_price:
            hit = True; exit_px = float(row["open"]) if row["open"] > self.stop_price else float(self.stop_price)
        if hit:
            pnl = (exit_px - self.entry_price) * self.position
            self.trades.append({"entry_time": self.entry_time, "exit_time": t, "direction": self.position,
                                "entry": self.entry_price, "exit": exit_px, "pnl": pnl, "signal_type": self.entry_signal_type})
            self.position = 0

    def _force_exit(self, px, t):
        pnl = (px - self.entry_price) * self.position
        self.trades.append({"entry_time": self.entry_time, "exit_time": t, "direction": self.position,
                            "entry": self.entry_price, "exit": px, "pnl": pnl, "signal_type": self.entry_signal_type + "_TS"})
        self.position = 0

    def _trailing(self, bar):
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0
        if atr <= 0: return
        pnl = (float(bar["close"]) - self.entry_price) * self.position
        if not self.trailing_active and pnl > self.p.activate_atr * atr:
            self.trailing_active = True
        if self.trailing_active:
            if self.position == 1:
                n = float(bar["high"]) - self.p.trail_atr * atr
                if n > self.stop_price: self.stop_price = n
            else:
                n = float(bar["low"]) + self.p.trail_atr * atr
                if n < self.stop_price: self.stop_price = n

    def _on_bar(self, bar):
        b = {"high": float(bar["high"]), "low": float(bar["low"]), "time": bar.name,
             "diff": float(bar["diff"]), "dea": float(bar["dea"]),
             "atr": float(bar["atr"]) if not np.isnan(bar["atr"]) else 0,
             "diff_15m": float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0,
             "dea_15m": float(bar["dea_15m"]) if not np.isnan(bar["dea_15m"]) else 0,
             "vol": float(bar["volume"]) if not np.isnan(bar["volume"]) else 0}
        if self.p.macd_slope_filter and "diff_15m_prev" in self.df_5m.columns:
            v = bar.get("diff_15m_prev", np.nan) if hasattr(bar, "get") else getattr(bar, "diff_15m_prev", np.nan)
            b["diff_15m_prev"] = float(v) if not np.isnan(v) else 0
        self._recent_diffs.append(b["diff"])
        if len(self._recent_diffs) > 10: self._recent_diffs.pop(0)
        self._recent_vols.append(b["vol"])
        self._incl(b)
        if self._bi(b):
            self._bi_timestamps.append(len(self.k_lines))
            self._sig(bar)

    def _incl(self, nb):
        if not self.k_lines: self.k_lines.append(nb); return
        last = self.k_lines[-1]
        il = nb["high"] <= last["high"] and nb["low"] >= last["low"]
        inw = last["high"] <= nb["high"] and last["low"] >= nb["low"]
        if il or inw:
            if self.inclusion_dir == 0: self.inclusion_dir = 1
            m = last.copy()
            for k in ["time", "diff", "atr", "diff_15m", "dea_15m", "vol"]:
                if k in nb: m[k] = nb[k]
            if "diff_15m_prev" in nb: m["diff_15m_prev"] = nb["diff_15m_prev"]
            if self.inclusion_dir == 1:
                m["high"] = max(last["high"], nb["high"]); m["low"] = max(last["low"], nb["low"])
            else:
                m["high"] = min(last["high"], nb["high"]); m["low"] = min(last["low"], nb["low"])
            self.k_lines[-1] = m
        else:
            if nb["high"] > last["high"] and nb["low"] > last["low"]: self.inclusion_dir = 1
            elif nb["high"] < last["high"] and nb["low"] < last["low"]: self.inclusion_dir = -1
            self.k_lines.append(nb)

    def _bi(self, current_bar):
        if len(self.k_lines) < 3: return None
        c, m2, l = self.k_lines[-1], self.k_lines[-2], self.k_lines[-3]
        cand = None
        if m2["high"] > l["high"] and m2["high"] > c["high"]:
            cand = {"type": "top", "price": m2["high"], "idx": len(self.k_lines)-2, "data": m2}
        elif m2["low"] < l["low"] and m2["low"] < c["low"]:
            cand = {"type": "bottom", "price": m2["low"], "idx": len(self.k_lines)-2, "data": m2}
        if not cand: return None
        if not self.bi_points: self.bi_points.append(cand); return None
        last = self.bi_points[-1]
        if last["type"] == cand["type"]:
            if last["type"] == "top" and cand["price"] > last["price"]: self.bi_points[-1] = cand
            elif last["type"] == "bottom" and cand["price"] < last["price"]: self.bi_points[-1] = cand
        else:
            # Bi amplitude filter
            if self.p.bi_amp_filter:
                amp = abs(cand["price"] - last["price"])
                atr = current_bar.get("atr", 0)
                if atr > 0 and amp < self.p.bi_amp_min_atr * atr:
                    return None
            if cand["idx"] - last["idx"] >= self.p.min_bi_gap:
                self.bi_points.append(cand); return cand
        return None

    def _upd_piv(self):
        if len(self.bi_points) < 4: return
        pts = self.bi_points[-4:]
        ranges = [(min(pts[i]["price"], pts[i+1]["price"]), max(pts[i]["price"], pts[i+1]["price"])) for i in range(3)]
        zg = min(r[1] for r in ranges); zd = max(r[0] for r in ranges)
        if zg > zd:
            self.pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points)-1})

    def _get_anchor_pivot(self):
        """Strategy 1: Merge overlapping pivots into a stable anchor."""
        if not self.pivots: return None
        if not self.p.anchor_pivot: return self.pivots[-1]
        
        # Start from the latest pivot and merge backwards
        anchor = self.pivots[-1].copy()
        for i in range(len(self.pivots)-2, max(-1, len(self.pivots)-self.p.regime_window-1), -1):
            pv = self.pivots[i]
            # Check overlap
            overlap_zg = min(anchor["zg"], pv["zg"])
            overlap_zd = max(anchor["zd"], pv["zd"])
            if overlap_zg > overlap_zd:
                # They overlap — merge
                if self.p.anchor_mode == "intersect":
                    anchor["zg"] = overlap_zg
                    anchor["zd"] = overlap_zd
                else:  # union
                    anchor["zg"] = max(anchor["zg"], pv["zg"])
                    anchor["zd"] = min(anchor["zd"], pv["zd"])
                anchor["end_bi_idx"] = max(anchor["end_bi_idx"], pv["end_bi_idx"])
            else:
                break  # no more overlap
        return anchor

    def _get_overlap_score(self):
        """Strategy 2: Calculate how much recent pivots overlap (choppy indicator)."""
        if len(self.pivots) < 3: return 0.0
        window = min(self.p.regime_window, len(self.pivots))
        recent = self.pivots[-window:]
        overlaps = 0
        for i in range(len(recent)-1):
            a, b = recent[i], recent[i+1]
            if min(a["zg"], b["zg"]) > max(a["zd"], b["zd"]):
                overlaps += 1
        return overlaps / (len(recent) - 1) if len(recent) > 1 else 0.0

    def _get_bi_density(self):
        """Strategy 3: Calculate bi density (bis per K-line in recent window)."""
        if len(self.k_lines) < 50: return 0.1
        window = min(self.p.density_window_k, len(self.k_lines))
        cutoff = len(self.k_lines) - window
        bis_in_window = sum(1 for t in self._bi_timestamps if t >= cutoff)
        return bis_in_window / window

    def _sig(self, bar):
        self._upd_piv()
        if len(self.bi_points) < 5: return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0
        if atr <= 0: return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        
        # Base direction
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])

        # Strategy 4: MACD slope filter
        if self.p.macd_slope_filter:
            diff_15m = float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0
            prev = bar.get("diff_15m_prev", np.nan) if hasattr(bar, "get") else getattr(bar, "diff_15m_prev", np.nan)
            diff_15m_prev = float(prev) if not np.isnan(prev) else 0
            # Bull only if MACD is positive AND not declining
            if bull and diff_15m < diff_15m_prev:
                bull = False  # MACD positive but declining = potential top
            if bear and diff_15m > diff_15m_prev:
                bear = False  # MACD negative but rising = potential bottom

        if self.p.trend_filter:
            diff_5m = float(bar["diff"]) if not np.isnan(bar["diff"]) else 0
            if diff_5m <= 0: bull = False
            if diff_5m >= 0: bear = False

        if self.p.macd_consistency > 0 and len(self._recent_diffs) >= self.p.macd_consistency:
            recent = self._recent_diffs[-self.p.macd_consistency:]
            if bull and not all(d > 0 for d in recent): bull = False
            if bear and not all(d < 0 for d in recent): bear = False

        # Strategy 2: Regime switch
        regime = "trend"
        if self.p.regime_switch:
            overlap = self._get_overlap_score()
            if overlap > self.p.regime_overlap_thresh:
                regime = "range"

        # Strategy 3: Bi density adaptive
        density_ok = True
        if self.p.bi_density_adaptive:
            density = self._get_bi_density()
            if density > self.p.density_high_thresh:
                # High density (noisy) — require stronger confirmation
                # Only allow signals if 5m close confirms (not just wick)
                density_ok = False  # will be checked per signal type below

        # Strategy 5: Volume confirmation
        vol_ok = True
        if self.p.vol_confirm and len(self._recent_vols) >= self.p.vol_window:
            vol_avg = sum(self._recent_vols) / len(self._recent_vols)
            curr_vol = self._recent_vols[-1] if self._recent_vols else 0
            if vol_avg > 0 and curr_vol < vol_avg * self.p.vol_mult:
                vol_ok = False

        # Strategy 1: Use anchor pivot instead of raw last pivot
        lp = self._get_anchor_pivot()

        sig = st = None; trig = sb = 0.0

        if lp:
            # In range regime, restrict 2B/2S but allow pivot-boundary signals
            if self.p.enable_3b and pn["type"] == "bottom" and pn["price"] > lp["zg"] and pl["price"] > lp["zg"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bull:
                    if vol_ok or not self.p.vol_confirm:
                        sig = "Buy"; st = "3B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            
            if not sig and self.p.enable_3s and pn["type"] == "top" and pn["price"] < lp["zd"] and pl["price"] < lp["zd"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bear:
                    if self.p.disable_3s_short:
                        if self.position == 1:
                            t = pn["data"].get("time", None)
                            exit_px = float(pn["data"]["low"])
                            pnl = exit_px - self.entry_price
                            self.trades.append({"entry_time": self.entry_time, "exit_time": t,
                                                "direction": 1, "entry": self.entry_price, "exit": exit_px,
                                                "pnl": pnl, "signal_type": self.entry_signal_type + "_3S_exit"})
                            self.position = 0; self.entry_time = None
                    else:
                        if vol_ok or not self.p.vol_confirm:
                            sig = "Sell"; st = "3S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        # 2B/2S — restricted in range regime
        if not sig and self.p.enable_2b2s:
            allow_2b2s = True
            if regime == "range" and self.p.regime_switch:
                allow_2b2s = False  # suppress momentum signals in choppy regime

            if allow_2b2s:
                if pn["type"] == "bottom":
                    if pn["price"] > pp["price"] and float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bull:
                        sig = "Buy"; st = "2B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
                elif pn["type"] == "top":
                    if pn["price"] < pp["price"] and float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bear:
                        sig = "Sell"; st = "2S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        if not sig: return
        
        # Density check: in high density, require signal bi to be at least 1 ATR
        if not density_ok and self.p.bi_density_adaptive:
            sig_amp = abs(pn["price"] - pl["price"])
            if sig_amp < 1.0 * atr:
                return  # too small in noisy regime
        
        if abs(trig - sb) >= self.p.entry_filter_atr * atr: return
        self.pending_signal = {"type": sig, "trigger_price": trig, "stop_base": sb, "signal_type": st}


def load_csv(fp):
    df = pd.read_csv(fp); df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"]); df = df.sort_values("datetime")
    if "volume" not in df.columns: df["volume"] = 0
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def calc_stats(trades):
    if trades.empty:
        return {"pnl": 0.0, "trades": 0, "win%": 0.0, "maxdd": 0.0, "avg": 0.0, "pf": 0.0}
    pnl = float(trades["pnl"].sum()); n = len(trades)
    w = float((trades["pnl"] > 0).mean() * 100)
    eq = trades.sort_values("exit_time")["pnl"].cumsum()
    dd = float(abs((eq - eq.cummax()).min()))
    gp = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gl = float(abs(trades.loc[trades["pnl"] < 0, "pnl"].sum()))
    pf = gp / gl if gl > 0 else float("inf")
    return {"pnl": pnl, "trades": n, "win%": w, "maxdd": dd, "avg": pnl/n, "pf": pf}


def signal_breakdown(trades):
    if trades.empty: return {}
    g = trades.groupby("signal_type").agg(pnl=("pnl", "sum"), n=("pnl", "size"))
    return {st: {"pnl": float(r["pnl"]), "n": int(r["n"])} for st, r in g.iterrows()}


def run_one(cfg_name, cfg, data_dir, contracts):
    p = StrategyParams(cfg_name, **cfg)
    cr = {}; total = 0; mn = float("inf"); mt = float("inf")
    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches: continue
        df = load_csv(matches[0])
        tester = ChanPivotTesterV6(df, p)
        trades = tester.run()
        s = calc_stats(trades); sb = signal_breakdown(trades)
        cr[c.upper()] = {**s, "signals": sb}
        total += s["pnl"]
        if s["pnl"] < mn: mn = s["pnl"]
        if s["trades"] < mt: mt = s["trades"]
    a800 = sum(1 for _, r in cr.items() if r["pnl"] >= 800)
    return {"name": cfg_name, "cfg": cfg, "contracts": cr, "total_pnl": total,
            "min_pnl": mn, "min_trades": mt, "above_800": a800, "score": mn + total/7}


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]
    
    # Best v4 references
    v4_best = {"activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
               "min_bi_gap": 5, "trend_filter": True, "disable_3s_short": True,
               "bi_amp_filter": True, "bi_amp_min_atr": 1.5, "macd_consistency": 3}
    v1_best = {"activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
               "min_bi_gap": 7, "trend_filter": True, "disable_3s_short": True}
    baseline = {"activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
                "min_bi_gap": 4, "disable_3s_short": True}
    
    configs = {}
    
    # References
    configs["REF_v4best"] = v4_best.copy()
    configs["REF_v1best"] = v1_best.copy()
    configs["REF_baseline"] = {"activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5, "min_bi_gap": 4}
    
    # === Strategy 1: Anchored Pivot ===
    for bg in [4, 5, 7]:
        base = v1_best.copy() if bg == 7 else (v4_best.copy() if bg == 5 else baseline.copy())
        base["min_bi_gap"] = bg
        configs[f"S1_anchor_int_bg{bg}"] = {**base, "anchor_pivot": True, "anchor_mode": "intersect"}
        configs[f"S1_anchor_uni_bg{bg}"] = {**base, "anchor_pivot": True, "anchor_mode": "union"}
    
    # === Strategy 2: Regime Switch ===
    for thresh in [0.5, 0.6, 0.7, 0.8]:
        configs[f"S2_regime_{thresh}_bg4"] = {**baseline, "trend_filter": True, "regime_switch": True, "regime_overlap_thresh": thresh}
        configs[f"S2_regime_{thresh}_bg5"] = {**v4_best, "regime_switch": True, "regime_overlap_thresh": thresh}
    
    # === Strategy 4: MACD slope ===
    configs["S4_slope_bg7"] = {**v1_best, "macd_slope_filter": True}
    configs["S4_slope_bg5"] = {**v4_best, "macd_slope_filter": True}
    configs["S4_slope_bg4"] = {**baseline, "trend_filter": True, "macd_slope_filter": True}
    
    # === Strategy 3: Bi Density Adaptive ===
    for dt in [0.10, 0.12, 0.15, 0.18]:
        configs[f"S3_density_{dt}_bg5"] = {**v4_best, "bi_density_adaptive": True, "density_high_thresh": dt}
    configs["S3_density_0.12_bg4"] = {**baseline, "trend_filter": True, "disable_3s_short": True,
                                       "bi_density_adaptive": True, "density_high_thresh": 0.12}
    
    # === Strategy 5: Volume confirmation ===
    for vm in [1.0, 1.2, 1.5]:
        configs[f"S5_vol_{vm}_bg5"] = {**v4_best, "vol_confirm": True, "vol_mult": vm}
    configs["S5_vol_1.2_bg7"] = {**v1_best, "vol_confirm": True, "vol_mult": 1.2}
    
    # === Time stop ===
    configs["TS_30_bg5"] = {**v4_best, "time_stop": 30}
    configs["TS_50_bg5"] = {**v4_best, "time_stop": 50}
    
    # === Combos (top strategies) ===
    # Anchor + regime
    configs["C_anchor_regime_bg4"] = {**baseline, "trend_filter": True, "disable_3s_short": True,
                                       "anchor_pivot": True, "regime_switch": True, "regime_overlap_thresh": 0.6}
    configs["C_anchor_regime_bg5"] = {**v4_best, "anchor_pivot": True, "regime_switch": True, "regime_overlap_thresh": 0.6}
    
    # Anchor + MACD slope
    configs["C_anchor_slope_bg7"] = {**v1_best, "anchor_pivot": True, "macd_slope_filter": True}
    configs["C_anchor_slope_bg5"] = {**v4_best, "anchor_pivot": True, "macd_slope_filter": True}
    
    # Regime + density
    configs["C_regime_density_bg4"] = {**baseline, "trend_filter": True, "disable_3s_short": True,
                                        "regime_switch": True, "regime_overlap_thresh": 0.6,
                                        "bi_density_adaptive": True, "density_high_thresh": 0.12}
    
    # Full combo: anchor + regime + slope + amp
    configs["FULL_bg5"] = {**v4_best, "anchor_pivot": True, "regime_switch": True, 
                            "regime_overlap_thresh": 0.6, "macd_slope_filter": True}
    configs["FULL_bg4"] = {**baseline, "trend_filter": True, "disable_3s_short": True,
                            "bi_amp_filter": True, "bi_amp_min_atr": 1.5,
                            "anchor_pivot": True, "regime_switch": True,
                            "regime_overlap_thresh": 0.6, "macd_slope_filter": True}
    
    print(f"Running {len(configs)} configurations across {len(contracts)} contracts...")
    
    results = []
    for i, (cfg_name, cfg) in enumerate(configs.items()):
        r = run_one(cfg_name, cfg, data_dir, contracts)
        results.append(r)
        if (i+1) % 10 == 0: print(f"  ... {i+1}/{len(configs)} done")
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    for r in results[:15]:
        print(f"\n{'='*70}")
        print(f"{r['name']} | Score: {r['score']:.0f} | Total: {r['total_pnl']:.0f} | Min: {r['min_pnl']:.0f} | Pass>=800: {r['above_800']}/7")
        print(f"{'='*70}")
        for c in ["P2201", "P2205", "P2401", "P2405", "P2505", "P2509", "P2601"]:
            cr = r["contracts"].get(c, {})
            pnl=cr.get("pnl",0); t=cr.get("trades",0); w=cr.get("win%",0); d=cr.get("maxdd",0); pf=cr.get("pf",0)
            m = "OK" if pnl >= 800 else "XX"
            sigs = cr.get("signals", {})
            ss = " | ".join(f"{k}:{v['pnl']:.0f}({v['n']})" for k,v in sorted(sigs.items()))
            print(f"  {m} {c}: pnl={pnl:>7.0f} t={t:>3} w={w:>5.1f}% dd={d:>6.0f} pf={pf:.2f} | {ss}")
    
    print(f"\n{'='*110}")
    print("FULL RANKING")
    print(f"{'='*110}")
    for i, r in enumerate(results):
        cs = " ".join(f"{c[-4:]}:{r['contracts'].get(c,{}).get('pnl',0):.0f}" for c in ["P2201","P2205","P2401","P2405","P2505","P2509","P2601"])
        print(f"#{i+1:>2} {r['name']:<30} sc={r['score']:>7.0f} tot={r['total_pnl']:>7.0f} min={r['min_pnl']:>7.0f} p={r['above_800']}/7 | {cs}")
    
    out_dir = Path("experiments") / (datetime.now().strftime("%Y%m%d_%H%M") + "_iter5")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        for r in results:
            for c, cr in r["contracts"].items():
                for k, v in list(cr.items()):
                    if isinstance(v, (np.floating, np.integer)): cr[k] = float(v)
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
