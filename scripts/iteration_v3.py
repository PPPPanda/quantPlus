"""iteration_v3.py — Phase 1-3 Iteration #3

Key insight from v1/v2:
- bi_gap (bar count) creates irreconcilable tradeoff: P2201 vs P2601
- P2201/P2401 (choppy) need strict filtering → high bi_gap helps
- P2601 (trending) profits from more signals → high bi_gap hurts
- No single bar-count threshold works for both

Iteration 3 core idea: **Bi Amplitude Filter**
- Instead of filtering by bar count, filter by price amplitude
- Require each bi to cover at least N * ATR in price movement
- In choppy markets: small bis get filtered → fewer false signals (helps P2201)
- In trending markets: large bis pass easily → signals preserved (helps P2601)
- This is market-adaptive without needing different parameters per contract

Additional ideas:
N) Bi amplitude filter: abs(bi_end - bi_start) >= N * ATR
O) Signal amplitude: require signal bi (current) to have meaningful size
P) Pivot relative position: signal must be meaningfully outside pivot (not just barely)
Q) Combined: amplitude + distance from pivot
R) Momentum alignment: require 5m MACD direction consistent across last 3 bars (not just snapshot)
"""

from __future__ import annotations
from pathlib import Path
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
        # v3 params
        self.bi_amp_filter = kwargs.get("bi_amp_filter", False)          # N: filter small bis
        self.bi_amp_min_atr = kwargs.get("bi_amp_min_atr", 1.0)         # N: min amplitude in ATR
        self.signal_amp_filter = kwargs.get("signal_amp_filter", False)  # O: filter small signal bis
        self.signal_amp_min_atr = kwargs.get("signal_amp_min_atr", 0.5) # O
        self.pivot_distance_filter = kwargs.get("pivot_distance_filter", False)  # P
        self.pivot_dist_min_atr = kwargs.get("pivot_dist_min_atr", 0.3)          # P: how far outside pivot
        self.macd_consistency = kwargs.get("macd_consistency", 0)        # R: require N consecutive bars
        self.pivot_width_filter = kwargs.get("pivot_width_filter", False)
        self.pivot_width_min_atr = kwargs.get("pivot_width_min_atr", 0.5)
        self.pivot_width_max_atr = kwargs.get("pivot_width_max_atr", 4.0)


class ChanPivotTesterV4:
    def __init__(self, df_1m: pd.DataFrame, p: StrategyParams):
        self.p = p
        self.df_1m = df_1m.reset_index(drop=True)
        self.trades: List[Dict] = []
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_price = 0.0
        self.trailing_active = False
        self.entry_signal_type = ""
        self.k_lines = []
        self.inclusion_dir = 0
        self.bi_points = []
        self.pivots = []
        self.pending_signal = None
        self._recent_diffs = []  # for MACD consistency check

        df_idx = self.df_1m.set_index("datetime")
        df_idx.index = pd.to_datetime(df_idx.index)
        self.df_5m = (
            df_idx.resample("5min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        self._calc_indicators()

    def _calc_indicators(self):
        df = self.df_5m
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        sig = macd.ewm(span=9, adjust=False).mean()
        df["diff"] = macd
        df["dea"] = sig

        df_15m = df.resample("15min", closed="right", label="right").agg({"close": "last"}).dropna()
        e1 = df_15m["close"].ewm(span=12, adjust=False).mean()
        e2 = df_15m["close"].ewm(span=26, adjust=False).mean()
        m = e1 - e2
        s = m.ewm(span=9, adjust=False).mean()
        aligned = pd.DataFrame({"diff": m, "dea": s}).shift(1).reindex(df.index, method="ffill")
        df["diff_15m"] = aligned["diff"]
        df["dea_15m"] = aligned["dea"]

        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

    def run(self) -> pd.DataFrame:
        for _, row in self.df_1m.iterrows():
            ct = pd.to_datetime(row["datetime"])
            if self.position != 0:
                self._check_exit(row)
            if self.position == 0 and self.pending_signal:
                self._check_entry(row)
            if ct.minute % 5 == 0 and ct in self.df_5m.index:
                bar = self.df_5m.loc[ct]
                self._on_bar_close(bar)
                if self.position != 0:
                    self._update_trailing(bar)
        return pd.DataFrame(self.trades)

    def _check_entry(self, row):
        s = self.pending_signal
        if not s: return
        if s["type"] == "Buy":
            if row["low"] < s["stop_base"]:
                self.pending_signal = None; return
            if row["high"] > s["trigger_price"]:
                fill = max(s["trigger_price"], row["open"])
                if fill > row["high"]: fill = row["close"]
                self._open(1, fill, pd.to_datetime(row["datetime"]), s["stop_base"], s["signal_type"])
        elif s["type"] == "Sell":
            if row["high"] > s["stop_base"]:
                self.pending_signal = None; return
            if row["low"] < s["trigger_price"]:
                fill = min(s["trigger_price"], row["open"])
                if fill < row["low"]: fill = row["close"]
                self._open(-1, fill, pd.to_datetime(row["datetime"]), s["stop_base"], s["signal_type"])

    def _open(self, d, px, t, sb, st):
        self.position = d; self.entry_price = float(px); self.entry_time = t
        self.entry_signal_type = st
        self.stop_price = float(sb - 1 if d == 1 else sb + 1)
        self.pending_signal = None; self.trailing_active = False

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
            self.position = 0; self.entry_time = None; self.entry_signal_type = ""

    def _update_trailing(self, bar):
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
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

    def _on_bar_close(self, bar):
        b = {"high": float(bar["high"]), "low": float(bar["low"]), "time": bar.name,
             "diff": float(bar["diff"]), "dea": float(bar["dea"]),
             "atr": float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0,
             "diff_15m": float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0.0,
             "dea_15m": float(bar["dea_15m"]) if not np.isnan(bar["dea_15m"]) else 0.0}
        
        # Track recent diffs for MACD consistency
        self._recent_diffs.append(b["diff"])
        if len(self._recent_diffs) > 10:
            self._recent_diffs.pop(0)
        
        self._inclusion(b)
        if self._bi(b):
            self._signal(bar)

    def _inclusion(self, nb):
        if not self.k_lines:
            self.k_lines.append(nb); return
        last = self.k_lines[-1]
        il = nb["high"] <= last["high"] and nb["low"] >= last["low"]
        inw = last["high"] <= nb["high"] and last["low"] >= nb["low"]
        if il or inw:
            if self.inclusion_dir == 0: self.inclusion_dir = 1
            m = last.copy(); m["time"] = nb["time"]; m["diff"] = nb["diff"]; m["atr"] = nb["atr"]
            m["diff_15m"] = nb["diff_15m"]; m["dea_15m"] = nb["dea_15m"]
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
        if not self.bi_points:
            self.bi_points.append(cand); return None
        last = self.bi_points[-1]
        if last["type"] == cand["type"]:
            if last["type"] == "top" and cand["price"] > last["price"]: self.bi_points[-1] = cand
            elif last["type"] == "bottom" and cand["price"] < last["price"]: self.bi_points[-1] = cand
        else:
            if cand["idx"] - last["idx"] >= self.p.min_bi_gap:
                # Bi amplitude filter: check if this bi has meaningful price movement
                if self.p.bi_amp_filter:
                    amp = abs(cand["price"] - last["price"])
                    atr = current_bar.get("atr", 0)
                    if atr > 0 and amp < self.p.bi_amp_min_atr * atr:
                        return None  # too small, skip this bi
                self.bi_points.append(cand); return cand
        return None

    def _update_pivots(self):
        if len(self.bi_points) < 4: return
        pts = self.bi_points[-4:]
        ranges = [(min(pts[i]["price"], pts[i+1]["price"]), max(pts[i]["price"], pts[i+1]["price"])) for i in range(3)]
        zg = min(r[1] for r in ranges); zd = max(r[0] for r in ranges)
        if zg > zd:
            self.pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points)-1})

    def _signal(self, bar):
        self._update_pivots()
        if len(self.bi_points) < 5: return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0: return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])

        if self.p.trend_filter:
            diff_5m = float(bar["diff"]) if not np.isnan(bar["diff"]) else 0
            if diff_5m <= 0: bull = False
            if diff_5m >= 0: bear = False

        # MACD consistency check
        if self.p.macd_consistency > 0 and len(self._recent_diffs) >= self.p.macd_consistency:
            recent = self._recent_diffs[-self.p.macd_consistency:]
            if bull and not all(d > 0 for d in recent):
                bull = False  # not consistently bullish
            if bear and not all(d < 0 for d in recent):
                bear = False

        # Signal amplitude filter: require the current bi to be meaningful
        if self.p.signal_amp_filter and atr > 0:
            sig_amp = abs(pn["price"] - pl["price"])
            if sig_amp < self.p.signal_amp_min_atr * atr:
                return  # signal bi too small

        sig = st = None; trig = sb = 0.0
        lp = self.pivots[-1] if self.pivots else None

        # Pivot width filter
        if lp and self.p.pivot_width_filter:
            pw = lp["zg"] - lp["zd"]
            if pw < self.p.pivot_width_min_atr * atr or pw > self.p.pivot_width_max_atr * atr:
                lp = None

        if lp:
            # Pivot distance filter
            if self.p.pivot_distance_filter and atr > 0:
                min_dist = self.p.pivot_dist_min_atr * atr
            else:
                min_dist = 0

            # 3B
            if self.p.enable_3b and pn["type"] == "bottom":
                if pn["price"] > lp["zg"] + min_dist and pl["price"] > lp["zg"]:
                    if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bull:
                        sig = "Buy"; st = "3B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            # 3S
            if not sig and self.p.enable_3s and pn["type"] == "top":
                if pn["price"] < lp["zd"] - min_dist and pl["price"] < lp["zd"]:
                    if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bear:
                        if self.p.disable_3s_short:
                            if self.position == 1:
                                t = pn["data"].get("time", None)
                                exit_px = float(pn["data"]["low"])
                                pnl = exit_px - self.entry_price
                                self.trades.append({"entry_time": self.entry_time, "exit_time": t,
                                                    "direction": 1, "entry": self.entry_price, "exit": exit_px,
                                                    "pnl": pnl, "signal_type": self.entry_signal_type + "_3S_exit"})
                                self.position = 0; self.entry_time = None; self.entry_signal_type = ""
                        else:
                            sig = "Sell"; st = "3S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        # 2B/2S
        if not sig and self.p.enable_2b2s:
            if pn["type"] == "bottom":
                if pn["price"] > pp["price"] and float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bull:
                    sig = "Buy"; st = "2B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            elif pn["type"] == "top":
                if pn["price"] < pp["price"] and float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bear:
                    sig = "Sell"; st = "2S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        if not sig: return
        if abs(trig - sb) >= self.p.entry_filter_atr * atr: return
        self.pending_signal = {"type": sig, "trigger_price": trig, "stop_base": sb, "signal_type": st}


def load_csv(fp):
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    if "volume" not in df.columns: df["volume"] = 0
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def calc_stats(trades):
    if trades.empty:
        return {"pnl": 0.0, "trades": 0, "win%": 0.0, "maxdd": 0.0, "avg": 0.0, "pf": 0.0}
    pnl = float(trades["pnl"].sum())
    n = len(trades)
    w = float((trades["pnl"] > 0).mean() * 100)
    eq = trades.sort_values("exit_time")["pnl"].cumsum()
    dd = float(abs((eq - eq.cummax()).min()))
    gross_p = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_l = float(abs(trades.loc[trades["pnl"] < 0, "pnl"].sum()))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")
    return {"pnl": pnl, "trades": n, "win%": w, "maxdd": dd, "avg": pnl/n, "pf": pf}


def signal_breakdown(trades):
    if trades.empty: return {}
    g = trades.groupby("signal_type").agg(pnl=("pnl", "sum"), n=("pnl", "size"))
    return {st: {"pnl": float(r["pnl"]), "n": int(r["n"])} for st, r in g.iterrows()}


def run_one(cfg_name, cfg, data_dir, contracts):
    p = StrategyParams(cfg_name, **cfg)
    contract_results = {}
    total_pnl = 0; min_pnl = float("inf"); min_trades = float("inf")
    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches: continue
        df = load_csv(matches[0])
        tester = ChanPivotTesterV4(df, p)
        trades = tester.run()
        s = calc_stats(trades)
        sb = signal_breakdown(trades)
        contract_results[c.upper()] = {**s, "signals": sb}
        total_pnl += s["pnl"]
        if s["pnl"] < min_pnl: min_pnl = s["pnl"]
        if s["trades"] < min_trades: min_trades = s["trades"]
    above_800 = sum(1 for c, r in contract_results.items() if r["pnl"] >= 800)
    score = min_pnl + total_pnl / 7
    return {"name": cfg_name, "cfg": cfg, "contracts": contract_results,
            "total_pnl": total_pnl, "min_pnl": min_pnl, "min_trades": min_trades,
            "above_800": above_800, "score": score}


def print_ranking(results):
    print(f"\n{'='*110}")
    print("RANKING (by score = min_pnl + avg_pnl)")
    print(f"{'='*110}")
    for i, r in enumerate(results):
        c_status = []
        for c in ["P2201", "P2205", "P2401", "P2405", "P2505", "P2509", "P2601"]:
            p = r["contracts"].get(c, {}).get("pnl", 0)
            c_status.append(f"{c[-4:]}:{p:.0f}")
        print(f"#{i+1:>2} {r['name']:<25} score={r['score']:>7.0f} total={r['total_pnl']:>7.0f} min={r['min_pnl']:>7.0f} pass={r['above_800']}/7 | {' '.join(c_status)}")


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]
    
    # Best v1 reference
    v1_best = {"activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
               "min_bi_gap": 7, "trend_filter": True, "disable_3s_short": True}
    
    # Baseline reference
    baseline = {"activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5, "min_bi_gap": 4}
    
    configs = {}
    
    # === References ===
    configs["REF_v1best"] = v1_best.copy()
    configs["REF_baseline"] = baseline.copy()
    
    # === N: Bi amplitude filter with various bi_gap + amplitude combos ===
    # With low bi_gap (4) + amplitude filter (the key innovation)
    for amp in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        configs[f"N_bg4_amp{amp}"] = {
            **baseline, "disable_3s_short": True, "trend_filter": True,
            "bi_amp_filter": True, "bi_amp_min_atr": amp,
            "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
        }
    
    # With bi_gap=5 + amplitude
    for amp in [0.5, 0.8, 1.0, 1.5]:
        configs[f"N_bg5_amp{amp}"] = {
            "min_bi_gap": 5, "disable_3s_short": True, "trend_filter": True,
            "bi_amp_filter": True, "bi_amp_min_atr": amp,
            "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
        }
    
    # With bi_gap=6 + amplitude
    for amp in [0.5, 0.8, 1.0]:
        configs[f"N_bg6_amp{amp}"] = {
            "min_bi_gap": 6, "disable_3s_short": True, "trend_filter": True,
            "bi_amp_filter": True, "bi_amp_min_atr": amp,
            "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
        }
    
    # With bi_gap=7 + amplitude (to see if it helps P2601)
    for amp in [0.5, 0.8]:
        configs[f"N_bg7_amp{amp}"] = {
            **v1_best, "bi_amp_filter": True, "bi_amp_min_atr": amp,
        }
    
    # === O: Signal amplitude filter ===
    for sa in [0.3, 0.5, 0.8, 1.0]:
        configs[f"O_sigamp{sa}"] = {
            **v1_best, "signal_amp_filter": True, "signal_amp_min_atr": sa,
        }
    
    # === P: Pivot distance filter ===
    for pd_val in [0.2, 0.3, 0.5, 0.8]:
        configs[f"P_pvdist{pd_val}"] = {
            **v1_best, "pivot_distance_filter": True, "pivot_dist_min_atr": pd_val,
        }
    
    # === R: MACD consistency ===
    for mc in [2, 3, 4]:
        configs[f"R_macdcon{mc}"] = {
            **v1_best, "macd_consistency": mc,
        }
    
    # === Combos: bi_amp + other filters ===
    # Best amplitude candidates with signal amp
    configs["NP_bg4_amp1_pd0.3"] = {
        **baseline, "disable_3s_short": True, "trend_filter": True,
        "bi_amp_filter": True, "bi_amp_min_atr": 1.0,
        "pivot_distance_filter": True, "pivot_dist_min_atr": 0.3,
        "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
    }
    configs["NO_bg4_amp1_sa0.5"] = {
        **baseline, "disable_3s_short": True, "trend_filter": True,
        "bi_amp_filter": True, "bi_amp_min_atr": 1.0,
        "signal_amp_filter": True, "signal_amp_min_atr": 0.5,
        "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
    }
    configs["NOR_bg4_amp1_sa0.5_mc2"] = {
        **baseline, "disable_3s_short": True, "trend_filter": True,
        "bi_amp_filter": True, "bi_amp_min_atr": 1.0,
        "signal_amp_filter": True, "signal_amp_min_atr": 0.5,
        "macd_consistency": 2,
        "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
    }
    
    # Try bi_gap=4 amp with BASE risk params (higher trail/activate)
    configs["N_bg4_amp1_base_risk"] = {
        "min_bi_gap": 4, "disable_3s_short": True, "trend_filter": True,
        "bi_amp_filter": True, "bi_amp_min_atr": 1.0,
        "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
    }
    configs["N_bg4_amp1.2_base_risk"] = {
        "min_bi_gap": 4, "disable_3s_short": True, "trend_filter": True,
        "bi_amp_filter": True, "bi_amp_min_atr": 1.2,
        "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
    }
    
    # With 3S enabled (not disabled) + amplitude filter — maybe 3S works when bis are clean?
    configs["N_bg4_amp1_3s_on"] = {
        "min_bi_gap": 4, "trend_filter": True,
        "bi_amp_filter": True, "bi_amp_min_atr": 1.0,
        "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
    }
    configs["N_bg5_amp0.8_3s_on"] = {
        "min_bi_gap": 5, "trend_filter": True,
        "bi_amp_filter": True, "bi_amp_min_atr": 0.8,
        "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
    }
    
    print(f"Running {len(configs)} configurations across {len(contracts)} contracts...")
    print(f"Target: all 7 contracts PnL >= 800 points")
    print()
    
    results = []
    for i, (cfg_name, cfg) in enumerate(configs.items()):
        r = run_one(cfg_name, cfg, data_dir, contracts)
        results.append(r)
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(configs)} done")
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Print top 20 details
    for r in results[:20]:
        print(f"\n{'='*70}")
        print(f"Config: {r['name']} | Score: {r['score']:.0f} | Total: {r['total_pnl']:.0f} | Min: {r['min_pnl']:.0f} | Pass>=800: {r['above_800']}/7")
        print(f"{'='*70}")
        for c in ["P2201", "P2205", "P2401", "P2405", "P2505", "P2509", "P2601"]:
            cr = r["contracts"].get(c, {})
            pnl = cr.get("pnl", 0); trades = cr.get("trades", 0)
            win = cr.get("win%", 0); dd = cr.get("maxdd", 0); pf = cr.get("pf", 0)
            marker = "OK" if pnl >= 800 else "XX"
            signals = cr.get("signals", {})
            sig_str = " | ".join(f"{k}:{v['pnl']:.0f}({v['n']})" for k, v in sorted(signals.items()))
            print(f"  {marker} {c}: pnl={pnl:>7.0f} trades={trades:>3} win={win:>5.1f}% dd={dd:>6.0f} pf={pf:.2f} | {sig_str}")
    
    print_ranking(results)
    
    # Save
    out_dir = Path("experiments") / (datetime.now().strftime("%Y%m%d_%H%M") + "_iter3")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        for r in results:
            for c, cr in r["contracts"].items():
                for k, v in list(cr.items()):
                    if isinstance(v, (np.floating, np.integer)):
                        cr[k] = float(v)
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
