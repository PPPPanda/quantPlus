"""backtest_with_pivot_sm.py

Experiment: Replace sliding-window pivot with stateful pivot (state machine).

Pivot states:
  FORMING  -> 3 bi overlaps detected, pivot created with zg/zd
  EXTENDING -> subsequent bi still within [zd, zg], pivot extends
  LEFT     -> bi exits beyond zg (upward leave) or below zd (downward leave)
  
Key changes from baseline:
1. Pivot persists (extends) when new bi stays within pivot range
2. Pivot is "left" when price exits, not replaced by a new one
3. 3B/3S only trigger on a confirmed "left + pullback" sequence
4. Don't create new pivot if current one is still active (extending)

Also tests: disabling 3S short (3S as exit-only), fixing 2B/2S direction.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd


class PivotStateMachine:
    """Manages pivot lifecycle: forming -> extending -> left."""

    def __init__(self):
        self.current = None  # Active pivot dict or None
        self.state = "NONE"  # NONE / ACTIVE / LEFT_UP / LEFT_DOWN
        self.history = []    # Completed pivots
        self.leave_bi_idx = 0  # Bi index when price left the pivot

    def update(self, bi_points: list, new_bi_idx: int):
        """Called after each new bi. Returns state change."""
        if len(bi_points) < 4:
            return

        latest_price = bi_points[-1]["price"]

        # If we have an active pivot, check extension or leave
        if self.current and self.state == "ACTIVE":
            if self.current["zd"] <= latest_price <= self.current["zg"]:
                # Still inside -> extend
                self.current["end_bi_idx"] = new_bi_idx
                self.current["extension_count"] += 1
                return "EXTEND"
            elif latest_price > self.current["zg"]:
                self.state = "LEFT_UP"
                self.leave_bi_idx = new_bi_idx
                return "LEFT_UP"
            elif latest_price < self.current["zd"]:
                self.state = "LEFT_DOWN"
                self.leave_bi_idx = new_bi_idx
                return "LEFT_DOWN"

        # If pivot was left, check for pullback
        if self.current and self.state in ("LEFT_UP", "LEFT_DOWN"):
            # After leaving, if price comes back into pivot, it's a failed leave
            if self.current["zd"] <= latest_price <= self.current["zg"]:
                self.state = "ACTIVE"  # Back inside, re-extend
                self.current["end_bi_idx"] = new_bi_idx
                return "BACK_INSIDE"
            # If still outside, this might be a 3B/3S setup
            # Don't create new pivot yet - let the signal logic handle it
            return None

        # No active pivot or pivot was completed - try to form new one
        if self.state in ("NONE", "LEFT_UP", "LEFT_DOWN") or self.current is None:
            # Try forming from recent 4 bi points
            pts = bi_points[-4:]
            ranges = [(min(pts[i]["price"], pts[i+1]["price"]),
                        max(pts[i]["price"], pts[i+1]["price"])) for i in range(3)]
            zg = min(r[1] for r in ranges)
            zd = max(r[0] for r in ranges)

            if zg > zd:
                # Check if this overlaps with current left pivot
                if self.current and self.state in ("LEFT_UP", "LEFT_DOWN"):
                    # Don't create new pivot if it overlaps with old one
                    old_overlap = min(self.current["zg"], zg) > max(self.current["zd"], zd)
                    if old_overlap:
                        return None  # Skip, old pivot still relevant

                if self.current:
                    self.history.append(self.current)

                self.current = {
                    "zg": zg, "zd": zd,
                    "start_bi_idx": len(bi_points) - 4,
                    "end_bi_idx": new_bi_idx,
                    "extension_count": 0
                }
                self.state = "ACTIVE"
                return "NEW"

        return None


class ImprovedChanPivotTester:
    """Improved backtest with pivot state machine and optional fixes."""

    def __init__(self, df_1m: pd.DataFrame, config: dict):
        self.config = config
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
        self.pivot_sm = PivotStateMachine()
        self.pending_signal = None

        # Config defaults
        self.activate_atr = config.get("activate_atr", 2.5)
        self.trail_atr = config.get("trail_atr", 3.0)
        self.entry_filter_atr = config.get("entry_filter_atr", 2.5)
        self.min_bi_gap = config.get("min_bi_gap", 4)
        self.disable_3s_short = config.get("disable_3s_short", False)
        self.fix_divergence = config.get("fix_divergence", False)  # True = use real divergence for 2B/2S

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
                                "entry": self.entry_price, "exit": exit_px, "pnl": pnl, "signal_type": self.entry_signal_type})
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
        new_bi = self._bi()
        if new_bi:
            self.pivot_sm.update(self.bi_points, len(self.bi_points) - 1)
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

    def _signal(self, bar):
        if len(self.bi_points) < 5: return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0: return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])

        sig = st = None; trig = sb = 0.0
        pivot = self.pivot_sm.current

        # 3B/3S with state machine: only when pivot is in LEFT state
        if pivot:
            if self.pivot_sm.state == "LEFT_UP" and pn["type"] == "bottom":
                # Left upward + pullback (bottom) above ZG = 3B
                if pn["price"] > pivot["zg"] and bull:
                    sig = "Buy"; st = "3B"
                    trig = float(pn["data"]["high"]); sb = float(pn["price"])

            elif self.pivot_sm.state == "LEFT_DOWN" and pn["type"] == "top":
                # Left downward + pullback (top) below ZD = 3S
                if pn["price"] < pivot["zd"] and bear:
                    if self.disable_3s_short:
                        pass  # Skip 3S short
                    else:
                        sig = "Sell"; st = "3S"
                        trig = float(pn["data"]["low"]); sb = float(pn["price"])

        # 2B/2S
        if not sig:
            if pn["type"] == "bottom" and pn["price"] > pp["price"]:
                if self.fix_divergence:
                    # Real divergence: price higher but momentum weaker
                    if float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bull:
                        sig = "Buy"; st = "2B"
                        trig = float(pn["data"]["high"]); sb = float(pn["price"])
                else:
                    # Original: momentum stronger (not divergence)
                    if float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bull:
                        sig = "Buy"; st = "2B"
                        trig = float(pn["data"]["high"]); sb = float(pn["price"])

            elif pn["type"] == "top" and pn["price"] < pp["price"]:
                if self.fix_divergence:
                    if float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bear:
                        sig = "Sell"; st = "2S"
                        trig = float(pn["data"]["low"]); sb = float(pn["price"])
                else:
                    if float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bear:
                        sig = "Sell"; st = "2S"
                        trig = float(pn["data"]["low"]); sb = float(pn["price"])

        if not sig: return
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


def signal_breakdown(trades):
    if trades.empty: return ""
    g = trades.groupby("signal_type").agg(pnl=("pnl", "sum"), n=("pnl", "size"),
        win=("pnl", lambda x: (x>0).mean()*100))
    lines = []
    for st, r in g.sort_values("pnl").iterrows():
        lines.append(f"    {st}: pnl={r['pnl']:.0f}, n={int(r['n'])}, win={r['win']:.1f}%")
    return "\n".join(lines)


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]

    configs = {
        "BASELINE": {},
        "PIVOT_SM": {},  # Just pivot state machine
        "PIVOT_SM+NO3S": {"disable_3s_short": True},
        "PIVOT_SM+FIX_DIV": {"fix_divergence": True},
        "PIVOT_SM+NO3S+FIX_DIV": {"disable_3s_short": True, "fix_divergence": True},
    }

    all_results = []

    for cfg_name, cfg in configs.items():
        print(f"\n{'#'*70}")
        print(f"CONFIG: {cfg_name}")
        print(f"{'#'*70}")

        use_sm = cfg_name != "BASELINE"

        for c in contracts:
            matches = list(data_dir.glob(f"{c}_1min_*.csv"))
            if not matches: continue
            df = load_csv(matches[0])

            if use_sm:
                tester = ImprovedChanPivotTester(df, cfg)
            else:
                # Use old-style tester for baseline comparison
                from backtest_all_contracts import ChanPivotTester, StrategyParams
                p = StrategyParams("BEST", activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5)
                tester = ChanPivotTester(df, p)

            trades = tester.run()
            s = stats(trades)
            row = {"config": cfg_name, "contract": c.upper(), **s}
            all_results.append(row)
            print(f"  {c.upper()}: pnl={s['pnl']:.0f} | trades={s['trades']} | win={s['win%']:.1f}% | mdd={s['maxdd']:.0f} | PF={s['pf']:.2f}")
            print(signal_breakdown(trades))

    # Summary comparison
    print(f"\n\n{'='*90}")
    print("COMPARISON TABLE (PnL by config x contract)")
    print(f"{'='*90}")
    df_r = pd.DataFrame(all_results)
    pivot_table = df_r.pivot(index="contract", columns="config", values="pnl").reindex(
        columns=list(configs.keys()))
    print(pivot_table.to_string())
    print(f"\nTOTAL:")
    print(pivot_table.sum().to_string())


if __name__ == "__main__":
    main()
