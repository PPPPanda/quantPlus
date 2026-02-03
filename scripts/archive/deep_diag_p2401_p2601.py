"""deep_diag_p2401_p2601.py — Deep diagnosis of the two hardest contracts.

Focuses on understanding WHY P2401 always loses and WHY P2601 is bi_gap sensitive.
"""
from pathlib import Path
import numpy as np
import pandas as pd


class StrategyParams:
    def __init__(self, name="TEST", **kwargs):
        self.name = name
        self.activate_atr = kwargs.get("activate_atr", 2.5)
        self.trail_atr = kwargs.get("trail_atr", 3.0)
        self.entry_filter_atr = kwargs.get("entry_filter_atr", 2.5)
        self.pivot_valid_range = kwargs.get("pivot_valid_range", 6)
        self.min_bi_gap = kwargs.get("min_bi_gap", 4)


class DiagTester:
    """Backtester that records detailed diagnostic info per trade."""
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
        self.entry_atr = 0.0
        self.entry_diff_15m = 0.0
        self.entry_bi_amp = 0.0
        self.entry_pivot = None
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

        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        
        # Price trend (20-bar slope normalized by ATR)
        df["trend_20"] = (df["close"] - df["close"].shift(20)) / (df["atr"] * 20).replace(0, np.nan)

    def run(self):
        for _, row in self.df_1m.iterrows():
            ct = pd.to_datetime(row["datetime"])
            if self.position != 0: self._check_exit(row)
            if self.position == 0 and self.pending_signal: self._check_entry(row)
            if ct.minute % 5 == 0 and ct in self.df_5m.index:
                bar = self.df_5m.loc[ct]
                self._on_bar(bar)
                if self.position != 0: self._trailing(bar)
        return pd.DataFrame(self.trades)

    def _check_entry(self, row):
        s = self.pending_signal
        if not s: return
        if s["type"] == "Buy":
            if row["low"] < s["stop_base"]: self.pending_signal = None; return
            if row["high"] > s["trigger_price"]:
                fill = max(s["trigger_price"], row["open"])
                if fill > row["high"]: fill = row["close"]
                self._open(1, fill, pd.to_datetime(row["datetime"]), s)
        elif s["type"] == "Sell":
            if row["high"] > s["stop_base"]: self.pending_signal = None; return
            if row["low"] < s["trigger_price"]:
                fill = min(s["trigger_price"], row["open"])
                if fill < row["low"]: fill = row["close"]
                self._open(-1, fill, pd.to_datetime(row["datetime"]), s)

    def _open(self, d, px, t, s):
        self.position = d; self.entry_price = float(px); self.entry_time = t
        self.entry_signal_type = s["signal_type"]
        self.entry_atr = s.get("atr", 0)
        self.entry_diff_15m = s.get("diff_15m", 0)
        self.entry_bi_amp = s.get("bi_amp", 0)
        self.entry_pivot = s.get("pivot", None)
        self.stop_price = float(s["stop_base"] - 1 if d == 1 else s["stop_base"] + 1)
        self.pending_signal = None; self.trailing_active = False

    def _check_exit(self, row):
        hit = False; exit_px = 0.0; t = pd.to_datetime(row["datetime"])
        if self.position == 1 and row["low"] <= self.stop_price:
            hit = True; exit_px = float(row["open"]) if row["open"] < self.stop_price else float(self.stop_price)
        elif self.position == -1 and row["high"] >= self.stop_price:
            hit = True; exit_px = float(row["open"]) if row["open"] > self.stop_price else float(self.stop_price)
        if hit:
            pnl = (exit_px - self.entry_price) * self.position
            self.trades.append({
                "entry_time": self.entry_time, "exit_time": t,
                "direction": self.position, "entry": self.entry_price, "exit": exit_px,
                "pnl": pnl, "signal_type": self.entry_signal_type,
                "entry_atr": self.entry_atr, "entry_diff_15m": self.entry_diff_15m,
                "bi_amp_atr": self.entry_bi_amp,
                "pivot_zg": self.entry_pivot["zg"] if self.entry_pivot else None,
                "pivot_zd": self.entry_pivot["zd"] if self.entry_pivot else None,
            })
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
             "trend_20": float(bar["trend_20"]) if not np.isnan(bar.get("trend_20", np.nan)) else 0}
        self._incl(b)
        if self._bi(): self._sig(bar)

    def _incl(self, nb):
        if not self.k_lines: self.k_lines.append(nb); return
        last = self.k_lines[-1]
        il = nb["high"] <= last["high"] and nb["low"] >= last["low"]
        inw = last["high"] <= nb["high"] and last["low"] >= nb["low"]
        if il or inw:
            if self.inclusion_dir == 0: self.inclusion_dir = 1
            m = last.copy(); m["time"] = nb["time"]; m["diff"] = nb["diff"]; m["atr"] = nb["atr"]
            m["diff_15m"] = nb["diff_15m"]; m["dea_15m"] = nb["dea_15m"]; m["trend_20"] = nb["trend_20"]
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
        if not self.bi_points: self.bi_points.append(cand); return None
        last = self.bi_points[-1]
        if last["type"] == cand["type"]:
            if last["type"] == "top" and cand["price"] > last["price"]: self.bi_points[-1] = cand
            elif last["type"] == "bottom" and cand["price"] < last["price"]: self.bi_points[-1] = cand
        else:
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

    def _sig(self, bar):
        self._upd_piv()
        if len(self.bi_points) < 5: return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0
        if atr <= 0: return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])

        bi_amp = abs(pn["price"] - pl["price"]) / atr if atr > 0 else 0

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
        self.pending_signal = {"type": sig, "trigger_price": trig, "stop_base": sb, "signal_type": st,
                               "atr": atr, "diff_15m": float(bar["diff_15m"]),
                               "bi_amp": bi_amp, "pivot": lp}


def load_csv(fp):
    df = pd.read_csv(fp); df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"]); df = df.sort_values("datetime")
    if "volume" not in df.columns: df["volume"] = 0
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def analyze_contract(name, df, bi_gaps=[4, 7]):
    """Run with multiple bi_gaps and compare."""
    print(f"\n{'='*80}")
    print(f"  DEEP DIAGNOSIS: {name}")
    print(f"{'='*80}")
    
    results = {}
    for bg in bi_gaps:
        p = StrategyParams(f"bg{bg}", min_bi_gap=bg)
        tester = DiagTester(df, p)
        trades = tester.run()
        results[bg] = trades
        
        if trades.empty:
            print(f"\n  [bi_gap={bg}] No trades"); continue
        
        print(f"\n  --- bi_gap={bg} | PnL={trades['pnl'].sum():.0f} | Trades={len(trades)} ---")
        
        # Signal type breakdown
        print(f"\n  Signal breakdown:")
        for st, g in trades.groupby("signal_type"):
            n = len(g); pnl = g["pnl"].sum()
            win = (g["pnl"] > 0).mean() * 100
            avg_win = g.loc[g["pnl"] > 0, "pnl"].mean() if (g["pnl"] > 0).any() else 0
            avg_loss = g.loc[g["pnl"] < 0, "pnl"].mean() if (g["pnl"] < 0).any() else 0
            print(f"    {st}: pnl={pnl:>7.0f} n={n:>3} win={win:>5.1f}% avg_win={avg_win:>6.1f} avg_loss={avg_loss:>7.1f}")
        
        # Monthly breakdown
        trades["month"] = pd.to_datetime(trades["entry_time"]).dt.to_period("M")
        print(f"\n  Monthly PnL:")
        for m, g in trades.groupby("month"):
            print(f"    {m}: pnl={g['pnl'].sum():>7.0f} trades={len(g):>3}")
        
        # Bi amplitude analysis
        print(f"\n  Bi Amplitude (ATR multiples) distribution:")
        if "bi_amp_atr" in trades.columns:
            amps = trades["bi_amp_atr"].dropna()
            if len(amps) > 0:
                for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    print(f"    P{int(q*100):>2}: {amps.quantile(q):.2f} ATR")
                
                # Winners vs Losers amp comparison
                w_amp = trades.loc[trades["pnl"] > 0, "bi_amp_atr"].mean()
                l_amp = trades.loc[trades["pnl"] < 0, "bi_amp_atr"].mean()
                print(f"    Winner avg amp: {w_amp:.2f} ATR")
                print(f"    Loser avg amp:  {l_amp:.2f} ATR")
        
        # Direction analysis (long vs short)
        print(f"\n  Direction breakdown:")
        for d, g in trades.groupby("direction"):
            d_name = "LONG" if d == 1 else "SHORT"
            print(f"    {d_name}: pnl={g['pnl'].sum():>7.0f} n={len(g):>3} win={((g['pnl']>0).mean()*100):>5.1f}%")
        
        # 15m MACD direction at entry
        print(f"\n  15m MACD direction at entry:")
        bull_trades = trades[trades["entry_diff_15m"] > 0]
        bear_trades = trades[trades["entry_diff_15m"] <= 0]
        if len(bull_trades) > 0:
            print(f"    Bull 15m: pnl={bull_trades['pnl'].sum():>7.0f} n={len(bull_trades):>3} win={((bull_trades['pnl']>0).mean()*100):>5.1f}%")
        if len(bear_trades) > 0:
            print(f"    Bear 15m: pnl={bear_trades['pnl'].sum():>7.0f} n={len(bear_trades):>3} win={((bear_trades['pnl']>0).mean()*100):>5.1f}%")
    
    # Compare bi_gap=4 vs 7
    if 4 in results and 7 in results and not results[4].empty and not results[7].empty:
        t4 = results[4]; t7 = results[7]
        print(f"\n  --- COMPARISON: bi_gap=4 vs 7 ---")
        print(f"  bg4: PnL={t4['pnl'].sum():.0f}, trades={len(t4)}")
        print(f"  bg7: PnL={t7['pnl'].sum():.0f}, trades={len(t7)}")
        print(f"  Difference: {t4['pnl'].sum() - t7['pnl'].sum():.0f} points, {len(t4) - len(t7)} fewer trades")
        
        # What trades exist in bg4 but not bg7?
        # Compare by entry time (within 30min window)
        t4_times = set(pd.to_datetime(t4["entry_time"]).dt.floor("30min"))
        t7_times = set(pd.to_datetime(t7["entry_time"]).dt.floor("30min"))
        only_in_4 = t4_times - t7_times
        
        if only_in_4:
            # Get trades that are unique to bg4
            t4["entry_30m"] = pd.to_datetime(t4["entry_time"]).dt.floor("30min")
            unique_t4 = t4[t4["entry_30m"].isin(only_in_4)]
            print(f"\n  Trades ONLY in bg4 (filtered out by bg7):")
            print(f"    Count: {len(unique_t4)}")
            print(f"    Total PnL: {unique_t4['pnl'].sum():.0f}")
            print(f"    Winners: {(unique_t4['pnl'] > 0).sum()}, Losers: {(unique_t4['pnl'] <= 0).sum()}")
            if len(unique_t4) > 0:
                for st, g in unique_t4.groupby("signal_type"):
                    print(f"    {st}: pnl={g['pnl'].sum():.0f} n={len(g)}")


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    
    # P2401
    matches = list(data_dir.glob("p2401_1min_*.csv"))
    if matches:
        df = load_csv(matches[0])
        analyze_contract("P2401", df, bi_gaps=[4, 5, 7])
    
    # P2601
    matches = list(data_dir.glob("p2601_1min_*.csv"))
    if matches:
        df = load_csv(matches[0])
        analyze_contract("P2601", df, bi_gaps=[4, 5, 7])
    
    # Also check P2201 (the other problem child)
    matches = list(data_dir.glob("p2201_1min_*.csv"))
    if matches:
        df = load_csv(matches[0])
        analyze_contract("P2201", df, bi_gaps=[4, 7])


if __name__ == "__main__":
    main()
