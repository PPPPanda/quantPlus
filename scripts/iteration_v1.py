"""iteration_v1.py — Phase 1-3 Iteration #1

Strategy: Instead of fixing chan theory to be textbook correct (which degraded results),
explore a **hybrid approach**:
1. Keep the working parts (2B signal is most stable: 5/7 positive)
2. Fix the clearly broken parts (3S causes losses in 4/7 contracts)
3. Add new filtering/signal logic based on failure analysis

Key insights from previous work:
- 2B is the money maker (5/7 positive, strong in P2509 +1130)
- 3S is the worst (4/7 negative, especially P2201 -380, P2401 -277)
- 3B is highly variable (very good in P2505/P2601, terrible in P2201)
- P2201 and P2401 are the problem children

Iteration 1 Strategy Ideas:
A) 3S → close-long-only (never open short from 3S)
B) Add volatility regime filter (don't trade in low-vol choppy markets)
C) Improve 2B/2S: use actual divergence (weaker momentum = end of move)
D) Multi-timeframe trend strength requirement
E) Pivot stability: require pivot to "age" (survive N bars) before using
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import json
import sys
import itertools
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
        # New params for iteration 1
        self.disable_3s_short = kwargs.get("disable_3s_short", False)   # A: 3S only closes long
        self.vol_regime_filter = kwargs.get("vol_regime_filter", False)  # B: volatility filter
        self.vol_lookback = kwargs.get("vol_lookback", 50)
        self.vol_threshold = kwargs.get("vol_threshold", 0.5)           # percentile threshold
        self.real_divergence = kwargs.get("real_divergence", False)      # C: fix 2B/2S divergence
        self.trend_strength = kwargs.get("trend_strength", False)       # D: require strong trend
        self.trend_strength_bars = kwargs.get("trend_strength_bars", 20)
        self.pivot_min_age = kwargs.get("pivot_min_age", 0)            # E: pivot must survive N bars
        self.enable_3b = kwargs.get("enable_3b", True)
        self.enable_3s = kwargs.get("enable_3s", True)
        self.trend_filter = kwargs.get("trend_filter", False)           # from grid search


class ChanPivotTesterV2:
    """Enhanced backtester with new signal logic options."""
    
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
        df["macd_hist"] = macd - sig  # histogram for divergence detection

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

        # Volatility regime (ATR percentile)
        if self.p.vol_regime_filter:
            df["atr_pct"] = df["atr"].rolling(self.p.vol_lookback).rank(pct=True)
        
        # Trend strength: slope of close over N bars
        if self.p.trend_strength:
            n = self.p.trend_strength_bars
            df["trend_slope"] = (df["close"] - df["close"].shift(n)) / (df["atr"] * n).replace(0, np.nan)

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
             "macd_hist": float(bar["macd_hist"]) if not np.isnan(bar["macd_hist"]) else 0.0,
             "atr": float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0,
             "diff_15m": float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0.0,
             "dea_15m": float(bar["dea_15m"]) if not np.isnan(bar["dea_15m"]) else 0.0}
        
        # Volatility regime
        if self.p.vol_regime_filter and "atr_pct" in self.df_5m.columns:
            v = bar.get("atr_pct", np.nan) if hasattr(bar, "get") else getattr(bar, "atr_pct", np.nan)
            b["atr_pct"] = float(v) if not np.isnan(v) else 0.5
        
        # Trend slope
        if self.p.trend_strength and "trend_slope" in self.df_5m.columns:
            v = bar.get("trend_slope", np.nan) if hasattr(bar, "get") else getattr(bar, "trend_slope", np.nan)
            b["trend_slope"] = float(v) if not np.isnan(v) else 0.0

        self._inclusion(b)
        if self._bi():
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
            if "macd_hist" in nb: m["macd_hist"] = nb["macd_hist"]
            if "atr_pct" in nb: m["atr_pct"] = nb["atr_pct"]
            if "trend_slope" in nb: m["trend_slope"] = nb["trend_slope"]
            if self.inclusion_dir == 1:
                m["high"] = max(last["high"], nb["high"]); m["low"] = max(last["low"], nb["low"])
            else:
                m["high"] = min(last["high"], nb["high"]); m["low"] = min(last["low"], nb["low"])
            self.k_lines[-1] = m
        else:
            if nb["high"] > last["high"] and nb["low"] > last["low"]: self.inclusion_dir = 1
            elif nb["high"] < last["high"] and nb["low"] < last["low"]: self.inclusion_dir = -1
            self.k_lines.append(nb)

    def _bi(self):
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
                self.bi_points.append(cand); return cand
        return None

    def _update_pivots(self):
        if len(self.bi_points) < 4: return
        pts = self.bi_points[-4:]
        ranges = [(min(pts[i]["price"], pts[i+1]["price"]), max(pts[i]["price"], pts[i+1]["price"])) for i in range(3)]
        zg = min(r[1] for r in ranges); zd = max(r[0] for r in ranges)
        if zg > zd:
            self.pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points)-1,
                                "created_bar_count": len(self.k_lines)})

    def _signal(self, bar):
        self._update_pivots()
        if len(self.bi_points) < 5: return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0: return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])

        # Trend filter (from grid search - simple: 5m diff direction)
        if self.p.trend_filter:
            diff_5m = float(bar["diff"]) if not np.isnan(bar["diff"]) else 0
            if diff_5m <= 0: bull = False
            if diff_5m >= 0: bear = False

        sig = st = None; trig = sb = 0.0
        lp = self.pivots[-1] if self.pivots else None

        # Pivot age check
        if lp and self.p.pivot_min_age > 0:
            age = len(self.k_lines) - lp.get("created_bar_count", 0)
            if age < self.p.pivot_min_age:
                lp = None  # too young, ignore

        # Volatility regime check
        vol_ok = True
        if self.p.vol_regime_filter:
            last_kl = self.k_lines[-1] if self.k_lines else {}
            atr_pct = last_kl.get("atr_pct", 0.5)
            if atr_pct < self.p.vol_threshold:
                vol_ok = False  # low volatility = choppy, skip

        if lp and vol_ok:
            # 3B
            if self.p.enable_3b and pn["type"] == "bottom" and pn["price"] > lp["zg"] and pl["price"] > lp["zg"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bull:
                    sig = "Buy"; st = "3B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            # 3S
            if not sig and self.p.enable_3s and pn["type"] == "top" and pn["price"] < lp["zd"] and pl["price"] < lp["zd"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bear:
                    if self.p.disable_3s_short:
                        # Only close existing long position, don't open short
                        if self.position == 1:
                            # Force close long at market
                            t = pn["data"].get("time", None)
                            exit_px = float(pn["data"]["low"])
                            pnl = exit_px - self.entry_price
                            self.trades.append({"entry_time": self.entry_time, "exit_time": t,
                                                "direction": 1, "entry": self.entry_price, "exit": exit_px,
                                                "pnl": pnl, "signal_type": self.entry_signal_type + "_3S_exit"})
                            self.position = 0; self.entry_time = None; self.entry_signal_type = ""
                        # Don't set sig - we're not opening short
                    else:
                        sig = "Sell"; st = "3S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        # 2B/2S
        if not sig and self.p.enable_2b2s and vol_ok:
            if pn["type"] == "bottom":
                if self.p.real_divergence:
                    # Real divergence: price higher but momentum weaker (end of correction)
                    # This means: new low is higher than prev low, AND momentum (macd_hist) is less negative
                    price_higher = pn["price"] > pp["price"]
                    momentum_weaker = abs(pn["data"].get("macd_hist", 0)) < abs(pp["data"].get("macd_hist", 0))
                    if price_higher and momentum_weaker and bull:
                        sig = "Buy"; st = "2B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
                else:
                    # Original: momentum stronger (trend acceleration)
                    if pn["price"] > pp["price"] and float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bull:
                        sig = "Buy"; st = "2B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            elif pn["type"] == "top":
                if self.p.real_divergence:
                    price_lower = pn["price"] < pp["price"]
                    momentum_weaker = abs(pn["data"].get("macd_hist", 0)) < abs(pp["data"].get("macd_hist", 0))
                    if price_lower and momentum_weaker and bear:
                        sig = "Sell"; st = "2S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])
                else:
                    if pn["price"] < pp["price"] and float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bear:
                        sig = "Sell"; st = "2S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        # Trend strength filter
        if sig and self.p.trend_strength:
            last_kl = self.k_lines[-1] if self.k_lines else {}
            slope = last_kl.get("trend_slope", 0)
            if sig == "Buy" and slope < 0.05:
                sig = None  # not enough upward momentum
            elif sig == "Sell" and slope > -0.05:
                sig = None  # not enough downward momentum

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


def run_experiment(configs, data_dir, contracts):
    """Run multiple configs across all contracts and return results."""
    results = []
    
    for cfg_name, cfg in configs.items():
        p = StrategyParams(cfg_name, **cfg)
        contract_results = {}
        total_pnl = 0
        min_pnl = float("inf")
        min_trades = float("inf")
        
        for c in contracts:
            matches = list(data_dir.glob(f"{c}_1min_*.csv"))
            if not matches: continue
            df = load_csv(matches[0])
            tester = ChanPivotTesterV2(df, p)
            trades = tester.run()
            s = calc_stats(trades)
            sb = signal_breakdown(trades)
            contract_results[c.upper()] = {**s, "signals": sb}
            total_pnl += s["pnl"]
            if s["pnl"] < min_pnl: min_pnl = s["pnl"]
            if s["trades"] < min_trades: min_trades = s["trades"]
        
        # Score: weighted sum prioritizing worst contract
        above_800 = sum(1 for c, r in contract_results.items() if r["pnl"] >= 800)
        score = min_pnl + total_pnl / 7  # balance worst case + average
        
        results.append({
            "name": cfg_name,
            "cfg": cfg,
            "contracts": contract_results,
            "total_pnl": total_pnl,
            "min_pnl": min_pnl,
            "min_trades": min_trades,
            "above_800": above_800,
            "score": score,
        })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)


def print_result(r):
    print(f"\n{'='*70}")
    print(f"Config: {r['name']} | Score: {r['score']:.0f} | Total: {r['total_pnl']:.0f} | Min: {r['min_pnl']:.0f} | Pass≥800: {r['above_800']}/7")
    print(f"{'='*70}")
    for c in ["P2201", "P2205", "P2401", "P2405", "P2505", "P2509", "P2601"]:
        cr = r["contracts"].get(c, {})
        pnl = cr.get("pnl", 0)
        trades = cr.get("trades", 0)
        win = cr.get("win%", 0)
        dd = cr.get("maxdd", 0)
        pf = cr.get("pf", 0)
        marker = "OK" if pnl >= 800 else "XX"
        signals = cr.get("signals", {})
        sig_str = " | ".join(f"{k}:{v['pnl']:.0f}({v['n']})" for k, v in sorted(signals.items()))
        print(f"  {marker} {c}: pnl={pnl:>7.0f} trades={trades:>3} win={win:>5.1f}% dd={dd:>6.0f} pf={pf:.2f} | {sig_str}")


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]
    
    configs = {
        # Baseline (BEST params)
        "BASELINE": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4,
        },
        
        # Grid search best (for reference)
        "GRID_BEST": {
            "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
            "min_bi_gap": 7, "trend_filter": True,
        },
        
        # A: Disable 3S short — only close longs
        "A_no3S_short": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "disable_3s_short": True,
        },
        
        # A+grid: 3S close-only + grid best params
        "A_no3S_grid": {
            "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
            "min_bi_gap": 7, "trend_filter": True, "disable_3s_short": True,
        },
        
        # B: Volatility regime filter
        "B_vol_filter": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "vol_regime_filter": True, "vol_threshold": 0.3,
        },
        
        # C: Real divergence for 2B/2S
        "C_real_div": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "real_divergence": True,
        },
        
        # D: Trend strength filter
        "D_trend_str": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "trend_strength": True,
        },
        
        # E: Pivot aging (must survive 5 bars before signaling)
        "E_pivot_age5": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "pivot_min_age": 5,
        },
        
        "E_pivot_age10": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "pivot_min_age": 10,
        },
        
        # Combo: A+B (no 3S short + vol filter)
        "AB_no3S_vol": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "disable_3s_short": True, "vol_regime_filter": True, "vol_threshold": 0.3,
        },
        
        # Combo: A+C (no 3S short + real divergence)
        "AC_no3S_div": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "disable_3s_short": True, "real_divergence": True,
        },
        
        # Combo: A+D (no 3S short + trend strength)
        "AD_no3S_trend": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "disable_3s_short": True, "trend_strength": True,
        },
        
        # Combo: A+B+C (no 3S + vol filter + real div)
        "ABC_combo": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "disable_3s_short": True, "vol_regime_filter": True,
            "vol_threshold": 0.3, "real_divergence": True,
        },
        
        # Long-only (completely disable shorts)
        "LONG_ONLY": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "enable_3s": False,
            "disable_3s_short": True,  # no 3S shorts
        },
        
        # Grid best + real divergence
        "GRID_C": {
            "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
            "min_bi_gap": 7, "trend_filter": True, "real_divergence": True,
        },
        
        # Grid best + no 3S + real div
        "GRID_AC": {
            "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
            "min_bi_gap": 7, "trend_filter": True, "disable_3s_short": True, "real_divergence": True,
        },
        
        # Only 2B/2S, disable 3B/3S entirely
        "ONLY_2B2S": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "enable_3b": False, "enable_3s": False,
        },
        
        # Only 2B/2S with grid params
        "ONLY_2B2S_grid": {
            "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
            "min_bi_gap": 7, "trend_filter": True, "enable_3b": False, "enable_3s": False,
        },
        
        # 2B/2S + real divergence
        "ONLY_2B2S_div": {
            "activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5,
            "min_bi_gap": 4, "enable_3b": False, "enable_3s": False, "real_divergence": True,
        },
    }
    
    print(f"Running {len(configs)} configurations across {len(contracts)} contracts...")
    print(f"Target: all 7 contracts PnL >= 800 points")
    print()
    
    results = run_experiment(configs, data_dir, contracts)
    
    # Print all results
    for r in results:
        print_result(r)
    
    # Summary ranking
    print(f"\n\n{'='*90}")
    print("RANKING (by score = min_pnl + avg_pnl)")
    print(f"{'='*90}")
    for i, r in enumerate(results):
        c_status = []
        for c in ["P2201", "P2205", "P2401", "P2405", "P2505", "P2509", "P2601"]:
            p = r["contracts"].get(c, {}).get("pnl", 0)
            c_status.append(f"{c[-4:]}:{p:.0f}")
        print(f"#{i+1:>2} {r['name']:<20} score={r['score']:>7.0f} total={r['total_pnl']:>7.0f} min={r['min_pnl']:>7.0f} pass={r['above_800']}/7 | {' '.join(c_status)}")
    
    # Save results
    out_dir = Path("experiments") / datetime.now().strftime("%Y%m%d_%H%M") + "_iter1"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        # Convert for JSON serialization
        for r in results:
            for c, cr in r["contracts"].items():
                for k, v in cr.items():
                    if isinstance(v, (np.floating, np.integer)):
                        cr[k] = float(v)
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}")
    
    return results


if __name__ == "__main__":
    main()
