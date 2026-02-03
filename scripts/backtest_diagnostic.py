"""backtest_diagnostic.py

Extended backtest with structural state snapshot for each trade.
Outputs detailed trade logs for failure mode analysis (Phase 2).

Each trade record includes:
- Standard: entry/exit time, price, pnl, signal_type, direction
- Structure: bi state, pivot state, macd state, atr
- Market context: recent volatility regime, trend/range label
- Drawdown contribution
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import json
import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    name: str
    activate_atr: float = 2.5
    trail_atr: float = 3.0
    entry_filter_atr: float = 2.5
    pivot_valid_range: int = 6
    min_bi_gap: int = 4


class DiagnosticTester:
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
        self.entry_snapshot = {}

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

        # Volatility regime: ATR percentile over rolling 100 bars
        df["atr_pct"] = df["atr"].rolling(100, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    def _get_structure_snapshot(self):
        """Capture current structural state for trade diagnostics."""
        snap = {
            "bi_count": len(self.bi_points),
            "pivot_count": len(self.pivots),
            "k_lines_count": len(self.k_lines),
        }

        # Recent bi points
        if len(self.bi_points) >= 3:
            snap["bi_n1"] = {"type": self.bi_points[-1]["type"], "price": self.bi_points[-1]["price"]}
            snap["bi_n2"] = {"type": self.bi_points[-2]["type"], "price": self.bi_points[-2]["price"]}
            snap["bi_n3"] = {"type": self.bi_points[-3]["type"], "price": self.bi_points[-3]["price"]}

        # Last pivot
        if self.pivots:
            lp = self.pivots[-1]
            snap["last_pivot_zg"] = lp["zg"]
            snap["last_pivot_zd"] = lp["zd"]
            snap["last_pivot_end_bi"] = lp["end_bi_idx"]
            snap["bi_since_pivot"] = len(self.bi_points) - 1 - lp["end_bi_idx"]

            # Is current price inside pivot?
            if len(self.bi_points) >= 1:
                cp = self.bi_points[-1]["price"]
                if cp > lp["zg"]:
                    snap["price_vs_pivot"] = "above"
                elif cp < lp["zd"]:
                    snap["price_vs_pivot"] = "below"
                else:
                    snap["price_vs_pivot"] = "inside"

        return snap

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
        self.position = d
        self.entry_price = float(px)
        self.entry_time = t
        self.entry_signal_type = st
        self.stop_price = float(sb - 1 if d == 1 else sb + 1)
        self.pending_signal = None
        self.trailing_active = False
        self.entry_snapshot = self._get_structure_snapshot()
        # Add ATR and volatility regime at entry
        if t in self.df_5m.index:
            bar = self.df_5m.loc[t]
            self.entry_snapshot["atr"] = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0
            self.entry_snapshot["atr_pct"] = float(bar["atr_pct"]) if not np.isnan(bar.get("atr_pct", np.nan)) else 0.5
            self.entry_snapshot["diff_15m"] = float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0
            self.entry_snapshot["dea_15m"] = float(bar["dea_15m"]) if not np.isnan(bar["dea_15m"]) else 0

    def _check_exit(self, row):
        hit = False; exit_px = 0.0; t = pd.to_datetime(row["datetime"])
        if self.position == 1 and row["low"] <= self.stop_price:
            hit = True; exit_px = float(row["open"]) if row["open"] < self.stop_price else float(self.stop_price)
        elif self.position == -1 and row["high"] >= self.stop_price:
            hit = True; exit_px = float(row["open"]) if row["open"] > self.stop_price else float(self.stop_price)
        if hit:
            pnl = (exit_px - self.entry_price) * self.position
            # Calculate holding time
            hold_bars = 0
            if self.entry_time:
                hold_bars = int((t - self.entry_time).total_seconds() / 60)

            trade = {
                "entry_time": str(self.entry_time),
                "exit_time": str(t),
                "direction": self.position,
                "entry": self.entry_price,
                "exit": exit_px,
                "pnl": pnl,
                "signal_type": self.entry_signal_type,
                "hold_minutes": hold_bars,
                **{f"snap_{k}": v for k, v in self.entry_snapshot.items()
                   if not isinstance(v, dict)}
            }
            # Flatten bi snapshots
            for key in ["bi_n1", "bi_n2", "bi_n3"]:
                if key in self.entry_snapshot:
                    trade[f"snap_{key}_type"] = self.entry_snapshot[key]["type"]
                    trade[f"snap_{key}_price"] = self.entry_snapshot[key]["price"]
            self.trades.append(trade)
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
             "diff": float(bar["diff"]),
             "atr": float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0,
             "diff_15m": float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0.0,
             "dea_15m": float(bar["dea_15m"]) if not np.isnan(bar["dea_15m"]) else 0.0}
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
            self.pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points)-1})

    def _signal(self, bar):
        self._update_pivots()
        if len(self.bi_points) < 5: return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0: return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])

        sig = st = None; trig = sb = 0.0
        lp = self.pivots[-1] if self.pivots else None

        if lp:
            if pn["type"] == "bottom" and pn["price"] > lp["zg"] and pl["price"] > lp["zg"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bull:
                    sig = "Buy"; st = "3B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            elif pn["type"] == "top" and pn["price"] < lp["zd"] and pl["price"] < lp["zd"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bear:
                    sig = "Sell"; st = "3S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        if not sig:
            if pn["type"] == "bottom":
                if pn["price"] > pp["price"] and float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bull:
                    sig = "Buy"; st = "2B"; trig = float(pn["data"]["high"]); sb = float(pn["price"])
            elif pn["type"] == "top":
                if pn["price"] < pp["price"] and float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bear:
                    sig = "Sell"; st = "2S"; trig = float(pn["data"]["low"]); sb = float(pn["price"])

        if not sig: return
        if abs(trig - sb) >= self.p.entry_filter_atr * atr: return
        self.pending_signal = {"type": sig, "trigger_price": trig, "stop_base": sb, "signal_type": st}


def load_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def compute_drawdown_segments(trades: pd.DataFrame, top_n: int = 3):
    """Find top N drawdown segments."""
    if trades.empty:
        return []
    eq = trades["pnl"].cumsum()
    peak = eq.cummax()
    dd = eq - peak

    segments = []
    in_dd = False
    start_idx = 0
    for i in range(len(dd)):
        if dd.iloc[i] < 0 and not in_dd:
            in_dd = True
            start_idx = i
        elif dd.iloc[i] >= 0 and in_dd:
            in_dd = False
            seg_dd = dd.iloc[start_idx:i].min()
            segments.append({
                "start_trade": int(start_idx),
                "end_trade": int(i),
                "drawdown": float(seg_dd),
                "n_trades": i - start_idx,
                "start_time": trades.iloc[start_idx]["entry_time"],
                "end_time": trades.iloc[i-1]["exit_time"],
            })
    if in_dd:
        seg_dd = dd.iloc[start_idx:].min()
        segments.append({
            "start_trade": int(start_idx),
            "end_trade": int(len(dd)),
            "drawdown": float(seg_dd),
            "n_trades": len(dd) - start_idx,
            "start_time": trades.iloc[start_idx]["entry_time"],
            "end_time": trades.iloc[-1]["exit_time"],
        })

    segments.sort(key=lambda x: x["drawdown"])
    return segments[:top_n]


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    out_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/experiments/20260203_0150_baseline/trades")
    out_dir.mkdir(parents=True, exist_ok=True)

    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]
    params = StrategyParams("BEST")

    summary = {}

    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches: continue
        fp = matches[0]
        df = load_csv(fp)

        print(f"\n{'='*60}")
        print(f"Running diagnostic backtest: {c.upper()}")

        tester = DiagnosticTester(df, params)
        trades = tester.run()

        if trades.empty:
            print("  No trades")
            continue

        # Save trade log
        trades.to_csv(out_dir / f"{c}_trades.csv", index=False)

        # Summary stats
        pnl = trades["pnl"].sum()
        n = len(trades)
        w = (trades["pnl"] > 0).mean() * 100
        eq = trades["pnl"].cumsum()
        mdd = abs((eq - eq.cummax()).min())

        # Signal breakdown
        sig_stats = trades.groupby("signal_type").agg(
            pnl=("pnl", "sum"), n=("pnl", "size"),
            win=("pnl", lambda x: (x>0).mean()*100),
            avg_hold=("hold_minutes", "mean")
        ).to_dict("index")

        # Volatility regime breakdown
        vol_stats = {}
        if "snap_atr_pct" in trades.columns:
            trades["vol_regime"] = pd.cut(trades["snap_atr_pct"],
                bins=[0, 0.33, 0.67, 1.0], labels=["low_vol", "mid_vol", "high_vol"])
            vol_stats = trades.groupby("vol_regime", observed=True).agg(
                pnl=("pnl", "sum"), n=("pnl", "size"),
                win=("pnl", lambda x: (x>0).mean()*100)
            ).to_dict("index")

        # Pivot position breakdown
        pivot_stats = {}
        if "snap_price_vs_pivot" in trades.columns:
            pivot_stats = trades.groupby("snap_price_vs_pivot").agg(
                pnl=("pnl", "sum"), n=("pnl", "size"),
                win=("pnl", lambda x: (x>0).mean()*100)
            ).to_dict("index")

        # Top drawdown segments
        dd_segs = compute_drawdown_segments(trades)

        contract_summary = {
            "pnl": float(pnl), "trades": n, "win_pct": float(w), "mdd": float(mdd),
            "signal_breakdown": sig_stats,
            "volatility_breakdown": vol_stats,
            "pivot_position_breakdown": pivot_stats,
            "top_drawdowns": dd_segs,
        }
        summary[c.upper()] = contract_summary

        print(f"  PnL={pnl:.0f} | Trades={n} | Win={w:.1f}% | MDD={mdd:.0f}")
        print(f"  Signal breakdown:")
        for st, s in sig_stats.items():
            print(f"    {st}: pnl={s['pnl']:.0f}, n={s['n']}, win={s['win']:.1f}%, avg_hold={s['avg_hold']:.0f}min")
        if vol_stats:
            print(f"  Volatility regime:")
            for vr, s in vol_stats.items():
                print(f"    {vr}: pnl={s['pnl']:.0f}, n={s['n']}, win={s['win']:.1f}%")
        if pivot_stats:
            print(f"  Pivot position:")
            for pp, s in pivot_stats.items():
                print(f"    {pp}: pnl={s['pnl']:.0f}, n={s['n']}, win={s['win']:.1f}%")
        if dd_segs:
            print(f"  Top drawdown segments:")
            for seg in dd_segs:
                print(f"    DD={seg['drawdown']:.0f} | {seg['n_trades']} trades | {seg['start_time']} -> {seg['end_time']}")

    # Save full summary
    with open(out_dir / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {out_dir / 'diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
