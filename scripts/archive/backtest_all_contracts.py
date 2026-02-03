"""backtest_all_contracts.py

对 p2201/p2205/p2401/p2405/p2505/p2509/p2601 七个合约用 chan_pivot (BASE+BEST) 做回测，
输出汇总表 + 信号类型分解。
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import glob
import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    name: str
    activate_atr: float = 1.5
    trail_atr: float = 3.0
    entry_filter_atr: float = 2.0
    pivot_valid_range: int = 6
    min_bi_gap: int = 4
    enable_2b2s: bool = True


class ChanPivotTester:
    def __init__(self, df_1m: pd.DataFrame, p: StrategyParams):
        self.p = p
        self.df_1m = df_1m.reset_index(drop=True)
        self.trades: List[Dict] = []
        self.position = 0
        self.entry_price = 0.0
        self.entry_time: Optional[pd.Timestamp] = None
        self.stop_price = 0.0
        self.trailing_active = False
        self.entry_signal_type: str = ""
        self.k_lines = []
        self.inclusion_dir = 0
        self.bi_points = []
        self.pivots = []
        self.pending_signal: Optional[Dict] = None

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
        if not s:
            return
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


def load_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    if "volume" not in df.columns: df["volume"] = 0
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def stats(trades: pd.DataFrame) -> Dict:
    if trades.empty:
        return {"pnl": 0, "trades": 0, "win%": 0, "maxdd": 0, "avg": 0, "profit_factor": 0}
    pnl = float(trades["pnl"].sum())
    n = len(trades)
    w = float((trades["pnl"] > 0).mean() * 100)
    eq = trades.sort_values("exit_time")["pnl"].cumsum()
    dd = float(abs((eq - eq.cummax()).min()))
    avg = pnl / n
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = float(abs(trades.loc[trades["pnl"] < 0, "pnl"].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {"pnl": pnl, "trades": n, "win%": w, "maxdd": dd, "avg": avg, "profit_factor": pf}


def signal_breakdown(trades: pd.DataFrame) -> str:
    if trades.empty: return "  (no trades)"
    g = trades.groupby("signal_type").agg(
        pnl=("pnl", "sum"), n=("pnl", "size"),
        win=("pnl", lambda x: (x>0).mean()*100), avg=("pnl", "mean"))
    lines = []
    for st, r in g.sort_values("pnl").iterrows():
        lines.append(f"  {st}: pnl={r['pnl']:.0f}, n={int(r['n'])}, win={r['win']:.1f}%, avg={r['avg']:.1f}")
    return "\n".join(lines)


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]

    base_p = StrategyParams("BASE", activate_atr=1.5, trail_atr=3.0, entry_filter_atr=2.0)
    best_p = StrategyParams("BEST", activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5)

    all_results = []

    for c in contracts:
        # find file
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches:
            print(f"WARNING: no file for {c}"); continue
        fp = matches[0]

        df = load_csv(fp)
        date_range = f"{df['datetime'].min().strftime('%Y-%m-%d')} -> {df['datetime'].max().strftime('%Y-%m-%d')}"

        print(f"\n{'='*70}")
        print(f"{c.upper()} | {fp.name} | {len(df)} rows | {date_range}")
        print(f"{'='*70}")

        for p in [base_p, best_p]:
            tester = ChanPivotTester(df, p)
            trades = tester.run()
            s = stats(trades)
            row = {"contract": c.upper(), "params": p.name, **s}
            all_results.append(row)

            print(f"\n[{p.name}] pnl={s['pnl']:.0f} | trades={s['trades']} | win={s['win%']:.1f}% | maxdd={s['maxdd']:.0f} | avg={s['avg']:.1f} | PF={s['profit_factor']:.2f}")
            print(signal_breakdown(trades))

    # Summary table
    print("\n\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    df_res = pd.DataFrame(all_results)
    for p_name in ["BASE", "BEST"]:
        print(f"\n--- {p_name} ---")
        sub = df_res[df_res["params"] == p_name][["contract", "pnl", "trades", "win%", "maxdd", "avg", "profit_factor"]]
        sub = sub.copy()
        for col in ["pnl", "maxdd", "avg"]:
            sub[col] = sub[col].map(lambda x: f"{x:.0f}")
        sub["win%"] = sub["win%"].map(lambda x: f"{float(x):.1f}")
        sub["profit_factor"] = sub["profit_factor"].map(lambda x: f"{float(x):.2f}")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
