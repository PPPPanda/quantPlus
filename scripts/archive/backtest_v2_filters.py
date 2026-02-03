"""backtest_v2_filters.py

Phase 4 Experiment #2: Targeted filters based on diagnostic findings.
Key insight from Exp#1: Don't replace core logic, ADD filters.

Filters tested (independently and combined):
F1: Trend filter - use 60-bar MA slope on 5m to determine trend
    - 2B only when MA rising (don't buy in downtrend)
    - 2S only when MA falling (don't sell in uptrend)
F2: Pivot age filter - signals only when current pivot has survived >= N bi
F3: Cooldown - after 3 consecutive losses in same direction, pause that direction
F4: ATR volatility gate - skip signals when ATR > 2x median ATR (extreme volatility)
F5: 3S as exit-only (no short opening on 3S)
F6: Pivot width filter - skip 3B/3S when pivot range < 0.3*ATR (too narrow = noise)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from collections import deque


class FilteredChanPivotTester:
    def __init__(self, df_1m: pd.DataFrame, config: dict):
        self.cfg = config
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

        # Filter state
        self.consecutive_losses = {"Buy": 0, "Sell": 0}
        self.last_trade_dir = 0

        # Params
        self.activate_atr = config.get("activate_atr", 2.5)
        self.trail_atr = config.get("trail_atr", 3.0)
        self.entry_filter_atr = config.get("entry_filter_atr", 2.5)
        self.min_bi_gap = config.get("min_bi_gap", 4)
        self.pivot_valid_range = config.get("pivot_valid_range", 6)

        # Filter flags
        self.f_trend = config.get("f_trend", False)
        self.f_pivot_age = config.get("f_pivot_age", 0)  # min bi count for pivot
        self.f_cooldown = config.get("f_cooldown", 0)  # max consecutive losses
        self.f_vol_gate = config.get("f_vol_gate", False)
        self.f_no_3s_short = config.get("f_no_3s_short", False)
        self.f_pivot_width = config.get("f_pivot_width", 0.0)  # min pivot width as ATR multiple

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
        df["diff"] = macd; df["dea"] = sig

        df_15m = df.resample("15min", closed="right", label="right").agg({"close": "last"}).dropna()
        e1 = df_15m["close"].ewm(span=12, adjust=False).mean()
        e2 = df_15m["close"].ewm(span=26, adjust=False).mean()
        m = e1 - e2; s = m.ewm(span=9, adjust=False).mean()
        aligned = pd.DataFrame({"diff": m, "dea": s}).shift(1).reindex(df.index, method="ffill")
        df["diff_15m"] = aligned["diff"]; df["dea_15m"] = aligned["dea"]

        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # Trend: 60-bar MA on 5m close
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma_slope"] = df["ma60"] - df["ma60"].shift(5)  # slope over 5 bars

        # Median ATR for volatility gate
        df["atr_median"] = df["atr"].rolling(100, min_periods=20).median()

    def run(self):
        for _, row in self.df_1m.iterrows():
            ct = pd.to_datetime(row["datetime"])
            if self.position != 0: self._check_exit(row)
            if self.position == 0 and self.pending_signal: self._check_entry(row)
            if ct.minute % 5 == 0 and ct in self.df_5m.index:
                bar = self.df_5m.loc[ct]
                self._on_bar_close(bar)
                if self.position != 0: self._update_trailing(bar)
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
                                "entry": self.entry_price, "exit": exit_px, "pnl": pnl,
                                "signal_type": self.entry_signal_type})
            # Update cooldown state
            sig_dir = "Buy" if self.position == 1 else "Sell"
            if pnl < 0:
                self.consecutive_losses[sig_dir] += 1
            else:
                self.consecutive_losses[sig_dir] = 0
            self.position = 0

    def _update_trailing(self, bar):
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

    def _on_bar_close(self, bar):
        b = {"high": float(bar["high"]), "low": float(bar["low"]), "time": bar.name,
             "diff": float(bar["diff"]),
             "atr": float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0,
             "diff_15m": float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0.0,
             "dea_15m": float(bar["dea_15m"]) if not np.isnan(bar["dea_15m"]) else 0.0}
        self._inclusion(b)
        if self._bi():
            self._signal(bar)

    def _inclusion(self, nb):
        if not self.k_lines: self.k_lines.append(nb); return
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
            if cand["idx"] - last["idx"] >= self.min_bi_gap:
                self.bi_points.append(cand); return cand
        return None

    def _update_pivots(self):
        if len(self.bi_points) < 4: return
        pts = self.bi_points[-4:]
        ranges = [(min(pts[i]["price"], pts[i+1]["price"]), max(pts[i]["price"], pts[i+1]["price"])) for i in range(3)]
        zg = min(r[1] for r in ranges); zd = max(r[0] for r in ranges)
        if zg > zd:
            self.pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points)-1,
                                "created_bi": len(self.bi_points)-1})

    def _get_pivot_age(self):
        """How many bi since current pivot was created."""
        if not self.pivots: return 0
        return len(self.bi_points) - 1 - self.pivots[-1]["created_bi"]

    def _signal(self, bar):
        self._update_pivots()
        if len(self.bi_points) < 5: return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0: return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])

        # Trend filter state
        ma_slope = float(bar["ma_slope"]) if not np.isnan(bar.get("ma_slope", np.nan)) else 0.0
        trend_up = ma_slope > 0
        trend_down = ma_slope < 0

        # Volatility gate
        atr_median = float(bar["atr_median"]) if not np.isnan(bar.get("atr_median", np.nan)) else atr
        extreme_vol = atr > 2.0 * atr_median if atr_median > 0 else False

        sig = st = None; trig = sb = 0.0
        lp = self.pivots[-1] if self.pivots else None

        # F2: Pivot age filter
        pivot_age = self._get_pivot_age()
        pivot_mature = pivot_age >= self.f_pivot_age if self.f_pivot_age > 0 else True

        # F6: Pivot width filter
        pivot_wide_enough = True
        if lp and self.f_pivot_width > 0:
            width = lp["zg"] - lp["zd"]
            if width < self.f_pivot_width * atr:
                pivot_wide_enough = False

        # 3B/3S
        if lp and pivot_mature and pivot_wide_enough:
            if pn["type"] == "bottom" and pn["price"] > lp["zg"] and pl["price"] > lp["zg"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.pivot_valid_range and bull:
                    sig = "Buy"; st = "3B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            elif pn["type"] == "top" and pn["price"] < lp["zd"] and pl["price"] < lp["zd"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.pivot_valid_range and bear:
                    if self.f_no_3s_short:
                        pass  # Skip
                    else:
                        sig = "Sell"; st = "3S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        # 2B/2S (original logic - momentum, not divergence)
        if not sig:
            if pn["type"] == "bottom":
                if pn["price"] > pp["price"] and float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bull:
                    # F1: Trend filter for 2B
                    if self.f_trend and trend_down:
                        pass  # Don't buy in downtrend
                    else:
                        sig = "Buy"; st = "2B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            elif pn["type"] == "top":
                if pn["price"] < pp["price"] and float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bear:
                    # F1: Trend filter for 2S
                    if self.f_trend and trend_up:
                        pass  # Don't sell in uptrend
                    else:
                        sig = "Sell"; st = "2S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        if not sig: return

        # F4: Volatility gate
        if self.f_vol_gate and extreme_vol:
            # In extreme volatility, only allow signals in trend direction
            if sig == "Buy" and not trend_up: return
            if sig == "Sell" and not trend_down: return

        # F3: Cooldown
        if self.f_cooldown > 0:
            if self.consecutive_losses.get(sig, 0) >= self.f_cooldown:
                return

        # Entry distance filter
        if abs(trig - sb) >= self.entry_filter_atr * atr: return

        self.pending_signal = {"type": sig, "trigger_price": trig, "stop_base": sb, "signal_type": st}


def load_csv(fp):
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def stats(trades):
    if trades.empty: return {"pnl": 0, "trades": 0, "win%": 0, "maxdd": 0, "pf": 0}
    pnl = float(trades["pnl"].sum())
    n = len(trades)
    w = float((trades["pnl"] > 0).mean() * 100)
    eq = trades["pnl"].cumsum()
    dd = float(abs((eq - eq.cummax()).min()))
    gp = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gl = float(abs(trades.loc[trades["pnl"] < 0, "pnl"].sum()))
    pf = gp / gl if gl > 0 else float("inf")
    return {"pnl": pnl, "trades": n, "win%": w, "maxdd": dd, "pf": pf}


def sig_breakdown(trades):
    if trades.empty: return ""
    g = trades.groupby("signal_type").agg(pnl=("pnl", "sum"), n=("pnl", "size"),
        win=("pnl", lambda x: (x>0).mean()*100))
    return "  ".join(f"{st}:{r['pnl']:.0f}({int(r['n'])})" for st, r in g.sort_values("pnl").iterrows())


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]

    configs = {
        "BASE": {},
        "F1_trend": {"f_trend": True},
        "F2_age3": {"f_pivot_age": 3},
        "F3_cool3": {"f_cooldown": 3},
        "F5_no3S": {"f_no_3s_short": True},
        "F6_width03": {"f_pivot_width": 0.3},
        "F1+F5": {"f_trend": True, "f_no_3s_short": True},
        "F1+F3+F5": {"f_trend": True, "f_cooldown": 3, "f_no_3s_short": True},
        "F1+F3+F5+F6": {"f_trend": True, "f_cooldown": 3, "f_no_3s_short": True, "f_pivot_width": 0.3},
        "F1+F4+F5": {"f_trend": True, "f_vol_gate": True, "f_no_3s_short": True},
        "ALL": {"f_trend": True, "f_pivot_age": 3, "f_cooldown": 3, "f_vol_gate": True,
                "f_no_3s_short": True, "f_pivot_width": 0.3},
    }

    results = []
    for cfg_name, cfg in configs.items():
        row = {"config": cfg_name}
        total_pnl = 0
        min_pnl = float("inf")
        all_pass = True
        details = []

        for c in contracts:
            matches = list(data_dir.glob(f"{c}_1min_*.csv"))
            if not matches: continue
            df = load_csv(matches[0])
            tester = FilteredChanPivotTester(df, cfg)
            trades = tester.run()
            s = stats(trades)
            row[c.upper()] = s["pnl"]
            total_pnl += s["pnl"]
            min_pnl = min(min_pnl, s["pnl"])
            if s["pnl"] < 800: all_pass = False
            details.append(f"{c.upper()}:{s['pnl']:.0f}({s['trades']}t,{s['win%']:.0f}%,dd{s['maxdd']:.0f})")

        row["TOTAL"] = total_pnl
        row["MIN"] = min_pnl
        row["ALL_PASS"] = all_pass
        results.append(row)

        print(f"\n[{cfg_name}] TOTAL={total_pnl:.0f} MIN={min_pnl:.0f} {'PASS' if all_pass else 'FAIL'}")
        print(f"  {' | '.join(details)}")

    # Summary comparison
    print(f"\n\n{'='*120}")
    print("PnL COMPARISON TABLE")
    print(f"{'='*120}")
    header = f"{'Config':<20} " + " ".join(f"{'P'+c[1:]:<10}" for c in contracts) + f" {'TOTAL':>8} {'MIN':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r['config']:<20} "
        for c in contracts:
            v = r.get(c.upper(), 0)
            marker = " " if v >= 800 else "*"
            line += f"{v:>9.0f}{marker}"
        line += f" {r['TOTAL']:>8.0f} {r['MIN']:>8.0f}"
        print(line)


if __name__ == "__main__":
    main()
