"""backtest_v4_grid.py

Phase 4 Experiment #4: Parameter grid search.
Focus on parameters that can fundamentally change signal quality:
- min_bi_gap: 4,5,6,7,8 (stricter = fewer noisy bi, fewer signals)
- activate_atr: 1.5, 2.0, 2.5, 3.0 (when to start trailing)
- trail_atr: 2.0, 2.5, 3.0, 3.5 (how tight trailing stop)
- entry_filter_atr: 1.5, 2.0, 2.5, 3.0 (max distance to enter)
- long_only: True/False
- trend_filter (MA60): True/False

Objective: maximize min(PnL across 7 contracts), with constraint trades >= 40.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import itertools
import json


class GridTester:
    def __init__(self, df_1m, cfg):
        self.cfg = cfg
        self.df_1m = df_1m.reset_index(drop=True)
        self.trades = []
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

        self.activate_atr = cfg.get("activate_atr", 2.5)
        self.trail_atr = cfg.get("trail_atr", 3.0)
        self.entry_filter_atr = cfg.get("entry_filter_atr", 2.5)
        self.min_bi_gap = cfg.get("min_bi_gap", 4)
        self.pivot_valid_range = cfg.get("pivot_valid_range", 6)
        self.long_only = cfg.get("long_only", False)
        self.trend_filter = cfg.get("trend_filter", False)

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
        df["diff"] = e1 - e2; df["dea"] = df["diff"].ewm(span=9, adjust=False).mean()

        df_15m = df.resample("15min", closed="right", label="right").agg({"close": "last"}).dropna()
        x1 = df_15m["close"].ewm(span=12, adjust=False).mean()
        x2 = df_15m["close"].ewm(span=26, adjust=False).mean()
        m = x1 - x2; s = m.ewm(span=9, adjust=False).mean()
        al = pd.DataFrame({"diff": m, "dea": s}).shift(1).reindex(df.index, method="ffill")
        df["diff_15m"] = al["diff"]; df["dea_15m"] = al["dea"]

        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["ma60"] = df["close"].rolling(60).mean()

    def run(self):
        for _, row in self.df_1m.iterrows():
            ct = pd.to_datetime(row["datetime"])
            if self.position != 0: self._check_exit(row)
            if self.position == 0 and self.pending_signal: self._check_entry(row)
            if ct.minute % 5 == 0 and ct in self.df_5m.index:
                bar = self.df_5m.loc[ct]
                self._on_bar(bar)
                if self.position != 0: self._trail(bar)
        return self.trades

    def _check_entry(self, row):
        s = self.pending_signal
        if not s: return
        if s["type"] == "Buy":
            if row["low"] < s["sb"]: self.pending_signal = None; return
            if row["high"] > s["tp"]:
                f = max(s["tp"], row["open"])
                if f > row["high"]: f = row["close"]
                self._open(1, f, pd.to_datetime(row["datetime"]), s["sb"], s["st"])
        elif s["type"] == "Sell":
            if row["high"] > s["sb"]: self.pending_signal = None; return
            if row["low"] < s["tp"]:
                f = min(s["tp"], row["open"])
                if f < row["low"]: f = row["close"]
                self._open(-1, f, pd.to_datetime(row["datetime"]), s["sb"], s["st"])

    def _open(self, d, px, t, sb, st):
        self.position = d; self.entry_price = float(px); self.entry_time = t
        self.entry_signal_type = st
        self.stop_price = float(sb - 1 if d == 1 else sb + 1)
        self.pending_signal = None; self.trailing_active = False

    def _check_exit(self, row):
        hit = False; ep = 0.0; t = pd.to_datetime(row["datetime"])
        if self.position == 1 and row["low"] <= self.stop_price:
            hit = True; ep = min(float(row["open"]), float(self.stop_price))
        elif self.position == -1 and row["high"] >= self.stop_price:
            hit = True; ep = max(float(row["open"]), float(self.stop_price))
        if hit:
            pnl = (ep - self.entry_price) * self.position
            self.trades.append({"pnl": pnl, "signal_type": self.entry_signal_type, "dir": self.position})
            self.position = 0

    def _trail(self, bar):
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0: return
        pnl = (float(bar["close"]) - self.entry_price) * self.position
        if not self.trailing_active and pnl > self.activate_atr * atr: self.trailing_active = True
        if self.trailing_active:
            if self.position == 1:
                n = float(bar["high"]) - self.trail_atr * atr
                if n > self.stop_price: self.stop_price = n
            else:
                n = float(bar["low"]) + self.trail_atr * atr
                if n < self.stop_price: self.stop_price = n

    def _on_bar(self, bar):
        b = {"high": float(bar["high"]), "low": float(bar["low"]), "time": bar.name,
             "diff": float(bar["diff"]),
             "atr": float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0,
             "diff_15m": float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0.0,
             "dea_15m": float(bar["dea_15m"]) if not np.isnan(bar["dea_15m"]) else 0.0}
        self._inc(b)
        if self._bi(): self._sig(bar)

    def _inc(self, nb):
        if not self.k_lines: self.k_lines.append(nb); return
        l = self.k_lines[-1]
        il = nb["high"] <= l["high"] and nb["low"] >= l["low"]
        inw = l["high"] <= nb["high"] and l["low"] >= nb["low"]
        if il or inw:
            if self.inclusion_dir == 0: self.inclusion_dir = 1
            m = l.copy(); m["time"] = nb["time"]; m["diff"] = nb["diff"]; m["atr"] = nb["atr"]
            m["diff_15m"] = nb["diff_15m"]; m["dea_15m"] = nb["dea_15m"]
            if self.inclusion_dir == 1:
                m["high"] = max(l["high"], nb["high"]); m["low"] = max(l["low"], nb["low"])
            else:
                m["high"] = min(l["high"], nb["high"]); m["low"] = min(l["low"], nb["low"])
            self.k_lines[-1] = m
        else:
            if nb["high"] > l["high"] and nb["low"] > l["low"]: self.inclusion_dir = 1
            elif nb["high"] < l["high"] and nb["low"] < l["low"]: self.inclusion_dir = -1
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
        if not self.bi_points: self.bi_points.append(cand); return None
        last = self.bi_points[-1]
        if last["type"] == cand["type"]:
            if last["type"] == "top" and cand["price"] > last["price"]: self.bi_points[-1] = cand
            elif last["type"] == "bottom" and cand["price"] < last["price"]: self.bi_points[-1] = cand
        else:
            if cand["idx"] - last["idx"] >= self.min_bi_gap:
                self.bi_points.append(cand); return cand
        return None

    def _upd_piv(self):
        if len(self.bi_points) < 4: return
        pts = self.bi_points[-4:]
        rs = [(min(pts[i]["price"], pts[i+1]["price"]), max(pts[i]["price"], pts[i+1]["price"])) for i in range(3)]
        zg = min(r[1] for r in rs); zd = max(r[0] for r in rs)
        if zg > zd: self.pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points)-1})

    def _sig(self, bar):
        self._upd_piv()
        if len(self.bi_points) < 5: return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0: return
        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])

        # Trend filter
        ma60 = float(bar["ma60"]) if not np.isnan(bar.get("ma60", np.nan)) else 0
        trend_up = float(bar["close"]) > ma60 if ma60 > 0 else True
        trend_down = float(bar["close"]) < ma60 if ma60 > 0 else True

        sig = st = None; trig = sb = 0.0
        lp = self.pivots[-1] if self.pivots else None

        if lp:
            if pn["type"] == "bottom" and pn["price"] > lp["zg"] and pl["price"] > lp["zg"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.pivot_valid_range and bull:
                    if not self.trend_filter or trend_up:
                        sig = "Buy"; st = "3B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            elif not self.long_only and pn["type"] == "top" and pn["price"] < lp["zd"] and pl["price"] < lp["zd"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.pivot_valid_range and bear:
                    if not self.trend_filter or trend_down:
                        sig = "Sell"; st = "3S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        if not sig:
            if pn["type"] == "bottom" and pn["price"] > pp["price"] and float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bull:
                if not self.trend_filter or trend_up:
                    sig = "Buy"; st = "2B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            elif not self.long_only and pn["type"] == "top" and pn["price"] < pp["price"] and float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bear:
                if not self.trend_filter or trend_down:
                    sig = "Sell"; st = "2S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        if not sig: return
        if abs(trig - sb) >= self.entry_filter_atr * atr: return
        self.pending_signal = {"type": sig, "tp": trig, "sb": sb, "st": st}


def load_csv(fp):
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def run_grid():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]

    # Load all data
    data = {}
    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if matches: data[c] = load_csv(matches[0])

    # Grid
    grid = {
        "min_bi_gap": [4, 5, 6, 7],
        "activate_atr": [1.5, 2.0, 2.5, 3.0],
        "trail_atr": [2.0, 2.5, 3.0],
        "entry_filter_atr": [1.5, 2.0, 2.5, 3.0],
        "long_only": [False, True],
        "trend_filter": [False, True],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Total combinations: {len(combos)}")

    best_min = -float("inf")
    best_total = -float("inf")
    best_cfg = None
    best_results = None
    top_results = []

    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))

        total_pnl = 0; min_pnl = float("inf"); min_trades = float("inf")
        contract_results = {}

        for c in contracts:
            if c not in data: continue
            tester = GridTester(data[c], cfg)
            trades = tester.run()
            pnl = sum(t["pnl"] for t in trades)
            n = len(trades)
            contract_results[c.upper()] = {"pnl": pnl, "trades": n}
            total_pnl += pnl
            min_pnl = min(min_pnl, pnl)
            min_trades = min(min_trades, n)

        # Score: prioritize min PnL, then total
        score = min_pnl * 2 + total_pnl * 0.5

        if min_pnl > best_min or (min_pnl == best_min and total_pnl > best_total):
            best_min = min_pnl
            best_total = total_pnl
            best_cfg = cfg.copy()
            best_results = contract_results.copy()

        # Track top 20
        top_results.append({"cfg": cfg.copy(), "min": min_pnl, "total": total_pnl,
                           "min_trades": min_trades, "results": contract_results.copy(), "score": score})

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(combos)}] current best_min={best_min:.0f} best_total={best_total:.0f}")

    # Sort by score
    top_results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n\n{'='*130}")
    print(f"TOP 20 RESULTS (by score = 2*min + 0.5*total)")
    print(f"{'='*130}")
    for rank, r in enumerate(top_results[:20], 1):
        cfg_str = " ".join(f"{k}={v}" for k, v in r["cfg"].items())
        pnl_str = " ".join(f"{c}:{r['results'][c.upper()]['pnl']:.0f}({r['results'][c.upper()]['trades']})"
                           for c in contracts if c.upper() in r["results"])
        all_pass = all(r["results"][c.upper()]["pnl"] >= 800 for c in contracts if c.upper() in r["results"])
        pass_count = sum(1 for c in contracts if c.upper() in r["results"] and r["results"][c.upper()]["pnl"] >= 800)
        print(f"#{rank:2d} score={r['score']:.0f} min={r['min']:.0f} total={r['total']:.0f} "
              f"min_t={r['min_trades']} pass={pass_count}/7 {'PASS' if all_pass else ''}")
        print(f"     {pnl_str}")
        print(f"     {cfg_str}")

    # Save
    out = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/experiments/20260203_0150_baseline")
    with open(out / "grid_top20.json", "w") as f:
        json.dump([{"rank": i+1, **r, "results": {k: v for k, v in r["results"].items()}}
                   for i, r in enumerate(top_results[:20])], f, indent=2, default=str)


if __name__ == "__main__":
    run_grid()
