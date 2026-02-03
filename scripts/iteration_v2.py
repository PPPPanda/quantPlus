"""iteration_v2.py — Phase 1-3 Iteration #2

Key finding from v1:
- bi_gap=7 fixes P2201 (+823 vs -885) but kills P2601 (1477→71)
- P2401 is always negative regardless of config
- Real divergence (fix 2B/2S) HURTS performance — original "momentum acceleration" works better
- Grid params + no-3S-short is the best simple combo (score=612)

Root cause analysis:
- P2201/P2401: choppy/ranging markets → too many false breakouts → need MORE filtering
- P2601: strong trending → profits from more signals → penalized by over-filtering
- The bi_gap parameter creates a one-size-doesn't-fit-all problem

Iteration 2 strategies:
F) Adaptive bi_gap: use ATR-relative gap (in choppy markets the gap widens automatically)
G) Pivot width filter: only trade if pivot (zg-zd) is meaningful relative to ATR
H) Signal strength scoring: require multiple confirmations for weaker signals
I) Session time filter: avoid opening positions in low-volume periods
J) Consecutive loss cooldown: pause trading after N consecutive losses
K) Profit target: close winning trades at fixed ATR multiple (don't only use trailing stop)
L) Dynamic trend filter: use ADX or similar to distinguish trend vs range
M) Entry improvement: wait for pullback within the signal bar range
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import json
import sys
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
        # Iteration 2 params
        self.adaptive_bi_gap = kwargs.get("adaptive_bi_gap", False)       # F
        self.bi_gap_atr_mult = kwargs.get("bi_gap_atr_mult", 0.5)        # F: atr multiplier for gap
        self.pivot_width_filter = kwargs.get("pivot_width_filter", False)  # G
        self.pivot_width_min_atr = kwargs.get("pivot_width_min_atr", 0.5) # G: min pivot width in ATR
        self.pivot_width_max_atr = kwargs.get("pivot_width_max_atr", 5.0) # G: max pivot width in ATR
        self.cooldown_losses = kwargs.get("cooldown_losses", 0)           # J: pause after N losses
        self.cooldown_bars = kwargs.get("cooldown_bars", 0)               # J: pause for M bars
        self.profit_target_atr = kwargs.get("profit_target_atr", 0.0)    # K: take profit at N*ATR
        self.adx_filter = kwargs.get("adx_filter", False)                 # L: require ADX > threshold
        self.adx_threshold = kwargs.get("adx_threshold", 20)              # L
        self.require_pivot_breakout = kwargs.get("require_pivot_breakout", False)  # new: 3B requires actual breakout above pivot
        self.min_bars_since_pivot = kwargs.get("min_bars_since_pivot", 0) # min bars since pivot created
        self.close_on_pivot_break = kwargs.get("close_on_pivot_break", False)  # close position if price re-enters pivot


class ChanPivotTesterV3:
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
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        self.last_pivot_for_position = None  # track which pivot we're trading from

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

        # ADX calculation
        if self.p.adx_filter:
            plus_dm = df["high"].diff()
            minus_dm = -df["low"].diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            atr14 = df["atr"]
            plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
            minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
            df["adx"] = dx.ewm(span=14, adjust=False).mean()

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
                    # Profit target check
                    if self.p.profit_target_atr > 0:
                        self._check_profit_target(bar)
                    # Close on pivot break
                    if self.p.close_on_pivot_break and self.last_pivot_for_position:
                        self._check_pivot_break(bar)
                if self.cooldown_remaining > 0:
                    self.cooldown_remaining -= 1
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
        self.last_pivot_for_position = self.pivots[-1] if self.pivots else None

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
            # Cooldown tracking
            if pnl < 0:
                self.consecutive_losses += 1
                if self.p.cooldown_losses > 0 and self.consecutive_losses >= self.p.cooldown_losses:
                    self.cooldown_remaining = self.p.cooldown_bars
                    self.consecutive_losses = 0
            else:
                self.consecutive_losses = 0
            self.position = 0; self.entry_time = None; self.entry_signal_type = ""
            self.last_pivot_for_position = None

    def _check_profit_target(self, bar):
        """Take profit at fixed ATR multiple."""
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0: return
        target = self.p.profit_target_atr * atr
        if self.position == 1:
            pnl = float(bar["close"]) - self.entry_price
            if pnl >= target:
                exit_px = self.entry_price + target
                self.trades.append({"entry_time": self.entry_time, "exit_time": bar.name,
                                    "direction": 1, "entry": self.entry_price, "exit": exit_px,
                                    "pnl": target, "signal_type": self.entry_signal_type + "_TP"})
                self.consecutive_losses = 0
                self.position = 0; self.entry_time = None
        elif self.position == -1:
            pnl = self.entry_price - float(bar["close"])
            if pnl >= target:
                exit_px = self.entry_price - target
                self.trades.append({"entry_time": self.entry_time, "exit_time": bar.name,
                                    "direction": -1, "entry": self.entry_price, "exit": exit_px,
                                    "pnl": target, "signal_type": self.entry_signal_type + "_TP"})
                self.consecutive_losses = 0
                self.position = 0; self.entry_time = None

    def _check_pivot_break(self, bar):
        """Close position if price re-enters the pivot zone."""
        pv = self.last_pivot_for_position
        if not pv: return
        price = float(bar["close"])
        if self.position == 1 and price < pv["zg"]:
            # Price fell back into pivot — exit long
            exit_px = price
            pnl = exit_px - self.entry_price
            self.trades.append({"entry_time": self.entry_time, "exit_time": bar.name,
                                "direction": 1, "entry": self.entry_price, "exit": exit_px,
                                "pnl": pnl, "signal_type": self.entry_signal_type + "_PB"})
            if pnl < 0: self.consecutive_losses += 1
            else: self.consecutive_losses = 0
            self.position = 0; self.entry_time = None
        elif self.position == -1 and price > pv["zd"]:
            exit_px = price
            pnl = self.entry_price - exit_px
            self.trades.append({"entry_time": self.entry_time, "exit_time": bar.name,
                                "direction": -1, "entry": self.entry_price, "exit": exit_px,
                                "pnl": pnl, "signal_type": self.entry_signal_type + "_PB"})
            if pnl < 0: self.consecutive_losses += 1
            else: self.consecutive_losses = 0
            self.position = 0; self.entry_time = None

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
        if self.p.adx_filter and "adx" in self.df_5m.columns:
            b["adx"] = float(bar["adx"]) if not np.isnan(bar["adx"]) else 0.0
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
            if "adx" in nb: m["adx"] = nb["adx"]
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
        
        # Determine effective bi_gap
        effective_gap = self.p.min_bi_gap
        if self.p.adaptive_bi_gap:
            atr = current_bar.get("atr", 0)
            if atr > 0:
                # In high volatility, require larger gap
                price_range = abs(cand["price"] - last["price"])
                atr_ratio = price_range / atr if atr > 0 else 1
                # If the bi covers less than bi_gap_atr_mult ATRs, increase the gap requirement
                if atr_ratio < self.p.bi_gap_atr_mult:
                    effective_gap = max(effective_gap, int(effective_gap * 1.5))
        
        if last["type"] == cand["type"]:
            if last["type"] == "top" and cand["price"] > last["price"]: self.bi_points[-1] = cand
            elif last["type"] == "bottom" and cand["price"] < last["price"]: self.bi_points[-1] = cand
        else:
            if cand["idx"] - last["idx"] >= effective_gap:
                self.bi_points.append(cand); return cand
        return None

    def _update_pivots(self):
        if len(self.bi_points) < 4: return
        pts = self.bi_points[-4:]
        ranges = [(min(pts[i]["price"], pts[i+1]["price"]), max(pts[i]["price"], pts[i+1]["price"])) for i in range(3)]
        zg = min(r[1] for r in ranges); zd = max(r[0] for r in ranges)
        if zg > zd:
            self.pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points)-1,
                                "bar_idx": len(self.k_lines)})

    def _signal(self, bar):
        self._update_pivots()
        if len(self.bi_points) < 5: return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0: return

        # Cooldown check
        if self.cooldown_remaining > 0: return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])

        if self.p.trend_filter:
            diff_5m = float(bar["diff"]) if not np.isnan(bar["diff"]) else 0
            if diff_5m <= 0: bull = False
            if diff_5m >= 0: bear = False

        # ADX filter
        if self.p.adx_filter:
            adx = float(bar["adx"]) if hasattr(bar, "adx") and not np.isnan(bar["adx"]) else 0.0
            if adx < self.p.adx_threshold:
                return  # market not trending enough

        sig = st = None; trig = sb = 0.0
        lp = self.pivots[-1] if self.pivots else None

        # Pivot width filter
        if lp and self.p.pivot_width_filter:
            pw = lp["zg"] - lp["zd"]
            if pw < self.p.pivot_width_min_atr * atr or pw > self.p.pivot_width_max_atr * atr:
                lp = None  # pivot too narrow or too wide

        # Min bars since pivot
        if lp and self.p.min_bars_since_pivot > 0:
            bars_since = len(self.k_lines) - lp.get("bar_idx", 0)
            if bars_since < self.p.min_bars_since_pivot:
                lp = None

        if lp:
            # 3B
            if self.p.enable_3b and pn["type"] == "bottom" and pn["price"] > lp["zg"] and pl["price"] > lp["zg"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bull:
                    sig = "Buy"; st = "3B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            # 3S
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
                            if pnl < 0: self.consecutive_losses += 1
                            else: self.consecutive_losses = 0
                            self.position = 0; self.entry_time = None; self.entry_signal_type = ""
                    else:
                        sig = "Sell"; st = "3S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        # 2B/2S (original momentum logic - proven to work better than "real divergence")
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


def run_experiment(configs, data_dir, contracts):
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
            tester = ChanPivotTesterV3(df, p)
            trades = tester.run()
            s = calc_stats(trades)
            sb = signal_breakdown(trades)
            contract_results[c.upper()] = {**s, "signals": sb}
            total_pnl += s["pnl"]
            if s["pnl"] < min_pnl: min_pnl = s["pnl"]
            if s["trades"] < min_trades: min_trades = s["trades"]
        
        above_800 = sum(1 for c, r in contract_results.items() if r["pnl"] >= 800)
        score = min_pnl + total_pnl / 7
        
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
    print(f"Config: {r['name']} | Score: {r['score']:.0f} | Total: {r['total_pnl']:.0f} | Min: {r['min_pnl']:.0f} | Pass>=800: {r['above_800']}/7")
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
    
    # Best from v1: grid params + no 3S short
    best_v1 = {
        "activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5,
        "min_bi_gap": 7, "trend_filter": True, "disable_3s_short": True,
    }
    
    configs = {
        # Reference: best from v1
        "V1_BEST": best_v1.copy(),
        
        # G: Pivot width filter (various thresholds)
        "G_pw_0.3_3": {**best_v1, "pivot_width_filter": True, "pivot_width_min_atr": 0.3, "pivot_width_max_atr": 3.0},
        "G_pw_0.5_4": {**best_v1, "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0},
        "G_pw_0.5_6": {**best_v1, "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 6.0},
        "G_pw_1.0_5": {**best_v1, "pivot_width_filter": True, "pivot_width_min_atr": 1.0, "pivot_width_max_atr": 5.0},
        
        # J: Cooldown after losses
        "J_cool_2_10": {**best_v1, "cooldown_losses": 2, "cooldown_bars": 10},
        "J_cool_3_15": {**best_v1, "cooldown_losses": 3, "cooldown_bars": 15},
        "J_cool_3_20": {**best_v1, "cooldown_losses": 3, "cooldown_bars": 20},
        
        # K: Profit target
        "K_tp_3": {**best_v1, "profit_target_atr": 3.0},
        "K_tp_4": {**best_v1, "profit_target_atr": 4.0},
        "K_tp_5": {**best_v1, "profit_target_atr": 5.0},
        
        # L: ADX filter
        "L_adx_15": {**best_v1, "adx_filter": True, "adx_threshold": 15},
        "L_adx_20": {**best_v1, "adx_filter": True, "adx_threshold": 20},
        "L_adx_25": {**best_v1, "adx_filter": True, "adx_threshold": 25},
        
        # Pivot break exit
        "PB_exit": {**best_v1, "close_on_pivot_break": True},
        
        # Combos
        "GJ_pw_cool": {**best_v1, "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0,
                        "cooldown_losses": 3, "cooldown_bars": 15},
        "GK_pw_tp": {**best_v1, "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0,
                      "profit_target_atr": 4.0},
        "GL_pw_adx": {**best_v1, "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0,
                       "adx_filter": True, "adx_threshold": 20},
        "JK_cool_tp": {**best_v1, "cooldown_losses": 3, "cooldown_bars": 15, "profit_target_atr": 4.0},
        "GJK_all": {**best_v1, "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0,
                     "cooldown_losses": 3, "cooldown_bars": 15, "profit_target_atr": 4.0},
        
        # Try with baseline params (bi_gap=4) + new features to see if they help P2601
        "BASE_G": {"activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5, "min_bi_gap": 4,
                    "disable_3s_short": True, "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0},
        "BASE_GL": {"activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5, "min_bi_gap": 4,
                     "disable_3s_short": True, "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0,
                     "adx_filter": True, "adx_threshold": 20},
        "BASE_GJK": {"activate_atr": 2.5, "trail_atr": 3.0, "entry_filter_atr": 2.5, "min_bi_gap": 4,
                      "disable_3s_short": True, "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0,
                      "cooldown_losses": 3, "cooldown_bars": 15, "profit_target_atr": 4.0},
        
        # Mid-range bi_gap (5, 6) to find sweet spot
        "MID5": {"activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5, "min_bi_gap": 5,
                  "trend_filter": True, "disable_3s_short": True},
        "MID6": {"activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5, "min_bi_gap": 6,
                  "trend_filter": True, "disable_3s_short": True},
        "MID5_G": {"activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5, "min_bi_gap": 5,
                    "trend_filter": True, "disable_3s_short": True,
                    "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0},
        "MID6_G": {"activate_atr": 1.5, "trail_atr": 2.0, "entry_filter_atr": 1.5, "min_bi_gap": 6,
                    "trend_filter": True, "disable_3s_short": True,
                    "pivot_width_filter": True, "pivot_width_min_atr": 0.5, "pivot_width_max_atr": 4.0},
        
        # Adaptive bi_gap (F)
        "F_adapt_0.5": {**best_v1, "adaptive_bi_gap": True, "bi_gap_atr_mult": 0.5},
        "F_adapt_1.0": {**best_v1, "adaptive_bi_gap": True, "bi_gap_atr_mult": 1.0},
        "F_adapt_1.5": {**best_v1, "adaptive_bi_gap": True, "bi_gap_atr_mult": 1.5},
    }
    
    print(f"Running {len(configs)} configurations across {len(contracts)} contracts...")
    print(f"Target: all 7 contracts PnL >= 800 points")
    print()
    
    results = run_experiment(configs, data_dir, contracts)
    
    for r in results:
        print_result(r)
    
    print(f"\n\n{'='*90}")
    print("RANKING (by score = min_pnl + avg_pnl)")
    print(f"{'='*90}")
    for i, r in enumerate(results):
        c_status = []
        for c in ["P2201", "P2205", "P2401", "P2405", "P2505", "P2509", "P2601"]:
            p = r["contracts"].get(c, {}).get("pnl", 0)
            c_status.append(f"{c[-4:]}:{p:.0f}")
        print(f"#{i+1:>2} {r['name']:<20} score={r['score']:>7.0f} total={r['total_pnl']:>7.0f} min={r['min_pnl']:>7.0f} pass={r['above_800']}/7 | {' '.join(c_status)}")
    
    # Save
    out_dir = Path("experiments") / (datetime.now().strftime("%Y%m%d_%H%M") + "_iter2")
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
