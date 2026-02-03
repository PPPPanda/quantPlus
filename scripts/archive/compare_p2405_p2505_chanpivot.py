"""compare_p2405_p2505_chanpivot.py

目标：对比 P2405 vs P2505 在 chan_pivot 策略上的“收益掉在哪里”，并做可验证的优化尝试。

数据源：
- P2405: 来自 temp/data/derived/P2405_1min_like_old_ALL.csv（由 Excel 导出数据标准化得到）
- P2505: 来自 quantPlus/data/analyse/p2505_1min_202501-202504.csv（旧分析数据）

输出：
1) BASE/BEST 参数下的总体统计
2) trade-by-trade 记录（含信号类型 3B/3S/2B/2S）并按月汇总 pnl
3) 找出最大回撤区间、最差月份
4) 提出并验证 2-3 个“可落地优化”：
   - 禁用 2B/2S，只交易 3B/3S
   - 增加中枢有效性：要求 pivot_range_width >= pivot_atr_mult * ATR
   - 过滤低波动：ATR 低于 rolling 分位时不交易（简单实现）

运行（Windows venv）：
  E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\.venv\Scripts\python.exe \
    E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\scripts\compare_p2405_p2505_chanpivot.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

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

    # direction toggles
    enable_long: bool = True
    enable_short: bool = True

    # optimization toggles
    enable_2b2s: bool = True
    pivot_atr_mult: float = 0.0  # 0 => disabled
    atr_trade_filter_quantile: float = 0.0  # 0 => disabled


class ChanPivotTesterPandas:
    """Pandas 批处理回测器（与 verify_chan_pivot 逻辑一致），扩展：记录 signal_type 与 entry/exit。"""

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

        df_1m_idx = self.df_1m.set_index("datetime")
        df_1m_idx.index = pd.to_datetime(df_1m_idx.index)

        self.df_5m = (
            df_1m_idx.resample("5min", label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )

        self._calc_indicators()

        # Precompute ATR filter threshold if needed
        self.atr_filter_threshold: float = 0.0
        if self.p.atr_trade_filter_quantile > 0:
            atr_series = self.df_5m["atr"].dropna()
            if not atr_series.empty:
                self.atr_filter_threshold = float(atr_series.quantile(self.p.atr_trade_filter_quantile))

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
            current_time = pd.to_datetime(row["datetime"])

            if self.position != 0:
                self._check_exit(row)

            if self.position == 0 and self.pending_signal:
                self._check_entry(row)

            if current_time.minute % 5 == 0 and current_time in self.df_5m.index:
                bar_5m = self.df_5m.loc[current_time]
                self._on_bar_close(bar_5m)

                if self.position != 0:
                    self._update_trailing_stop(bar_5m)

        return pd.DataFrame(self.trades)

    def _check_entry(self, row):
        s = self.pending_signal
        if not s:
            return

        if s["type"] == "Buy":
            if row["low"] < s["stop_base"]:
                self.pending_signal = None
                return
            if row["high"] > s["trigger_price"]:
                fill = max(s["trigger_price"], row["open"])
                if fill > row["high"]:
                    fill = row["close"]
                self._open_position(1, fill, pd.to_datetime(row["datetime"]), s["stop_base"], s["signal_type"])

        elif s["type"] == "Sell":
            if row["high"] > s["stop_base"]:
                self.pending_signal = None
                return
            if row["low"] < s["trigger_price"]:
                fill = min(s["trigger_price"], row["open"])
                if fill < row["low"]:
                    fill = row["close"]
                self._open_position(-1, fill, pd.to_datetime(row["datetime"]), s["stop_base"], s["signal_type"])

    def _open_position(self, direction: int, price: float, time: pd.Timestamp, stop_base: float, signal_type: str):
        self.position = direction
        self.entry_price = float(price)
        self.entry_time = time
        self.entry_signal_type = signal_type
        self.stop_price = float(stop_base - 1 if direction == 1 else stop_base + 1)
        self.pending_signal = None
        self.trailing_active = False

    def _check_exit(self, row):
        hit = False
        exit_px = 0.0
        time = pd.to_datetime(row["datetime"])

        if self.position == 1:
            if row["low"] <= self.stop_price:
                hit = True
                exit_px = float(row["open"]) if row["open"] < self.stop_price else float(self.stop_price)
        elif self.position == -1:
            if row["high"] >= self.stop_price:
                hit = True
                exit_px = float(row["open"]) if row["open"] > self.stop_price else float(self.stop_price)

        if hit:
            pnl = (exit_px - self.entry_price) * self.position
            self.trades.append(
                {
                    "entry_time": self.entry_time,
                    "exit_time": time,
                    "direction": self.position,
                    "entry": self.entry_price,
                    "exit": exit_px,
                    "pnl": pnl,
                    "signal_type": self.entry_signal_type,
                }
            )
            self.position = 0
            self.entry_time = None
            self.entry_signal_type = ""

    def _update_trailing_stop(self, curr_bar):
        atr = float(curr_bar["atr"]) if not np.isnan(curr_bar["atr"]) else 0.0
        if atr <= 0:
            return
        pnl = (float(curr_bar["close"]) - self.entry_price) * self.position

        if not self.trailing_active and pnl > self.p.activate_atr * atr:
            self.trailing_active = True

        if self.trailing_active:
            if self.position == 1:
                new = float(curr_bar["high"]) - self.p.trail_atr * atr
                if new > self.stop_price:
                    self.stop_price = new
            else:
                new = float(curr_bar["low"]) + self.p.trail_atr * atr
                if new < self.stop_price:
                    self.stop_price = new

    def _on_bar_close(self, curr_bar):
        bar = {
            "high": float(curr_bar["high"]),
            "low": float(curr_bar["low"]),
            "time": curr_bar.name,
            "diff": float(curr_bar["diff"]),
            "atr": float(curr_bar["atr"]) if not np.isnan(curr_bar["atr"]) else 0.0,
            "diff_15m": float(curr_bar["diff_15m"]) if not np.isnan(curr_bar["diff_15m"]) else 0.0,
            "dea_15m": float(curr_bar["dea_15m"]) if not np.isnan(curr_bar["dea_15m"]) else 0.0,
        }

        self._process_inclusion(bar)
        new_bi = self._process_bi()
        if new_bi:
            self._check_signal(curr_bar)

    def _process_inclusion(self, new_bar):
        if not self.k_lines:
            self.k_lines.append(new_bar)
            return

        last = self.k_lines[-1]
        in_last = new_bar["high"] <= last["high"] and new_bar["low"] >= last["low"]
        in_new = last["high"] <= new_bar["high"] and last["low"] >= new_bar["low"]

        if in_last or in_new:
            if self.inclusion_dir == 0:
                self.inclusion_dir = 1
            merged = last.copy()
            merged["time"] = new_bar["time"]
            merged["diff"] = new_bar["diff"]
            merged["atr"] = new_bar["atr"]
            merged["diff_15m"] = new_bar["diff_15m"]
            merged["dea_15m"] = new_bar["dea_15m"]
            if self.inclusion_dir == 1:
                merged["high"] = max(last["high"], new_bar["high"])
                merged["low"] = max(last["low"], new_bar["low"])
            else:
                merged["high"] = min(last["high"], new_bar["high"])
                merged["low"] = min(last["low"], new_bar["low"])
            self.k_lines[-1] = merged
        else:
            if new_bar["high"] > last["high"] and new_bar["low"] > last["low"]:
                self.inclusion_dir = 1
            elif new_bar["high"] < last["high"] and new_bar["low"] < last["low"]:
                self.inclusion_dir = -1
            self.k_lines.append(new_bar)

    def _process_bi(self):
        if len(self.k_lines) < 3:
            return None

        curr = self.k_lines[-1]
        mid = self.k_lines[-2]
        left = self.k_lines[-3]

        is_top = mid["high"] > left["high"] and mid["high"] > curr["high"]
        is_bot = mid["low"] < left["low"] and mid["low"] < curr["low"]

        cand = None
        if is_top:
            cand = {"type": "top", "price": mid["high"], "idx": len(self.k_lines) - 2, "data": mid}
        elif is_bot:
            cand = {"type": "bottom", "price": mid["low"], "idx": len(self.k_lines) - 2, "data": mid}

        if not cand:
            return None

        if not self.bi_points:
            self.bi_points.append(cand)
            return None

        last = self.bi_points[-1]

        if last["type"] == cand["type"]:
            if last["type"] == "top" and cand["price"] > last["price"]:
                self.bi_points[-1] = cand
            elif last["type"] == "bottom" and cand["price"] < last["price"]:
                self.bi_points[-1] = cand
        else:
            if cand["idx"] - last["idx"] >= self.p.min_bi_gap:
                self.bi_points.append(cand)
                return cand
        return None

    def _update_pivots(self):
        if len(self.bi_points) < 4:
            return

        b0 = self.bi_points[-4]
        b1 = self.bi_points[-3]
        b2 = self.bi_points[-2]
        b3 = self.bi_points[-1]

        r1 = (min(b0["price"], b1["price"]), max(b0["price"], b1["price"]))
        r2 = (min(b1["price"], b2["price"]), max(b1["price"], b2["price"]))
        r3 = (min(b2["price"], b3["price"]), max(b2["price"], b3["price"]))

        zg = min(r1[1], r2[1], r3[1])
        zd = max(r1[0], r2[0], r3[0])

        if zg > zd:
            self.pivots.append({"zg": zg, "zd": zd, "start_bi_idx": len(self.bi_points) - 4, "end_bi_idx": len(self.bi_points) - 1})

    def _check_signal(self, curr_bar):
        self._update_pivots()
        if len(self.bi_points) < 5:
            return

        atr = float(curr_bar["atr"]) if not np.isnan(curr_bar["atr"]) else 0.0
        if atr <= 0:
            return

        # ATR trade filter
        if self.p.atr_trade_filter_quantile > 0 and self.atr_filter_threshold > 0:
            if atr < self.atr_filter_threshold:
                return

        p_now = self.bi_points[-1]
        p_last = self.bi_points[-2]
        p_prev = self.bi_points[-3]

        is_bull = float(curr_bar["diff_15m"]) > float(curr_bar["dea_15m"])  # 15m trend
        is_bear = float(curr_bar["diff_15m"]) < float(curr_bar["dea_15m"])

        sig = None
        signal_type = ""
        trig = 0.0
        stop_base = 0.0

        last_pivot = self.pivots[-1] if self.pivots else None

        # 3B/3S
        if last_pivot:
            pivot_width = float(last_pivot["zg"] - last_pivot["zd"])
            if self.p.pivot_atr_mult > 0 and pivot_width < self.p.pivot_atr_mult * atr:
                # pivot too narrow => skip
                pass
            else:
                if p_now["type"] == "bottom" and self.p.enable_long:
                    if p_now["price"] > last_pivot["zg"] and p_last["price"] > last_pivot["zg"]:
                        if last_pivot["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and is_bull:
                            sig = "Buy"
                            signal_type = "3B"
                            trig = float(p_now["data"]["high"])
                            stop_base = float(p_now["price"])
                elif p_now["type"] == "top" and self.p.enable_short:
                    if p_now["price"] < last_pivot["zd"] and p_last["price"] < last_pivot["zd"]:
                        if last_pivot["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and is_bear:
                            sig = "Sell"
                            signal_type = "3S"
                            trig = float(p_now["data"]["low"])
                            stop_base = float(p_now["price"])

        # 2B/2S
        if not sig and self.p.enable_2b2s:
            if p_now["type"] == "bottom" and self.p.enable_long:
                div = float(p_now["data"]["diff"]) > float(p_prev["data"]["diff"])  # MACD divergence
                if p_now["price"] > p_prev["price"] and div and is_bull:
                    sig = "Buy"
                    signal_type = "2B"
                    trig = float(p_now["data"]["high"])
                    stop_base = float(p_now["price"])
            elif p_now["type"] == "top" and self.p.enable_short:
                div = float(p_now["data"]["diff"]) < float(p_prev["data"]["diff"])
                if p_now["price"] < p_prev["price"] and div and is_bear:
                    sig = "Sell"
                    signal_type = "2S"
                    trig = float(p_now["data"]["low"])
                    stop_base = float(p_now["price"])

        if not sig:
            return

        # entry filter by ATR
        distance = abs(trig - stop_base)
        if distance >= self.p.entry_filter_atr * atr:
            return

        self.pending_signal = {
            "type": sig,
            "trigger_price": trig,
            "stop_base": stop_base,
            "signal_type": signal_type,
        }


def load_1m_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def summary(trades: pd.DataFrame) -> Dict:
    if trades.empty:
        return {"pnl": 0.0, "trades": 0, "win": 0.0, "dd": 0.0}
    pnl = float(trades["pnl"].sum())
    n = int(len(trades))
    win = float((trades["pnl"] > 0).mean() * 100)
    eq = trades.sort_values("exit_time")["pnl"].cumsum()
    dd = float(abs((eq - eq.cummax()).min()))
    return {"pnl": pnl, "trades": n, "win": win, "dd": dd}


def month_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M").astype(str)
    g = t.groupby(["month"]).agg(pnl=("pnl", "sum"), trades=("pnl", "size"), win=("pnl", lambda x: (x>0).mean()*100))
    return g.sort_index()


def signal_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby("signal_type").agg(pnl=("pnl", "sum"), trades=("pnl", "size"), win=("pnl", lambda x: (x>0).mean()*100), avg=("pnl", "mean"))
    return g.sort_values("pnl")


def run_case(name: str, df: pd.DataFrame, p: StrategyParams) -> Tuple[Dict, pd.DataFrame]:
    tester = ChanPivotTesterPandas(df, p)
    trades = tester.run()
    s = summary(trades)
    s["name"] = name
    return s, trades


def main():
    p2405 = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/temp/data/derived/P2405_1min_like_old_ALL.csv")
    p2505 = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse/p2505_1min_202501-202504.csv")

    df2405 = load_1m_csv(p2405)
    df2505 = load_1m_csv(p2505)

    print("Data ranges:")
    print("P2405:", df2405["datetime"].min(), "->", df2405["datetime"].max(), "rows", len(df2405))
    print("P2505:", df2505["datetime"].min(), "->", df2505["datetime"].max(), "rows", len(df2505))

    base = StrategyParams("BASE", activate_atr=1.5, trail_atr=3.0, entry_filter_atr=2.0)
    best = StrategyParams("BEST", activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5)

    # Optimization variants
    only_3 = StrategyParams("BEST_only3", activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5, enable_2b2s=False)
    pivot_wide = StrategyParams("BEST_pivotWide", activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5, enable_2b2s=True, pivot_atr_mult=0.5)
    atr_filter = StrategyParams("BEST_atrQ50", activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5, enable_2b2s=True, atr_trade_filter_quantile=0.5)

    long_only = StrategyParams("BEST_longOnly", activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5, enable_short=False)

    params = [base, best, only_3, long_only, pivot_wide, atr_filter]

    for sym, df in [("P2405", df2405), ("P2505", df2505)]:
        print("\n" + "="*80)
        print(sym)
        print("="*80)

        results = []
        trade_store: Dict[str, pd.DataFrame] = {}

        for p in params:
            s, trades = run_case(f"{sym}_{p.name}", df, p)
            results.append(s)
            trade_store[p.name] = trades

        # print summary
        for r in results:
            print(f"{r['name']}: pnl={r['pnl']:.0f}, trades={r['trades']}, win={r['win']:.2f}%, dd={r['dd']:.0f}")

        # detailed breakdown for BEST
        t_best = trade_store.get("BEST")
        if t_best is not None and not t_best.empty:
            print("\n[BEST] signal breakdown:")
            print(signal_breakdown(t_best).to_string())
            print("\n[BEST] month breakdown (top losses):")
            mb = month_breakdown(t_best)
            if not mb.empty:
                print(mb.sort_values('pnl').head(8).to_string())


if __name__ == "__main__":
    main()
