"""trim_and_backtest_p2509_mainperiod.py

按“主要交易量”裁剪合约数据的首尾低成交阶段，使其更接近“主力阶段”的回测口径。

需求对齐：
- 目标数据：quantPlus/data/analyse/p2509_1min_202503-202508.csv（旧数据）
- 处理方式：按“日期级别”的成交量/活跃度裁剪首尾（不做日内时段过滤）
- 然后在裁剪后的数据上回测 chan_pivot（BASE/BEST 参数），并与未裁剪的结果对比。

裁剪判定（可解释）：
1) 以 1min 数据聚合得到 daily_volume（每日成交量合计）
2) 计算 daily_volume 的分位数阈值（默认 q=0.30）
3) 将 daily_volume >= threshold 的日期视为“活跃日”
4) 在时间轴上取“最长连续活跃区间”（允许中间少量断点），作为主力阶段

你可以调参：
- --q: 分位数（越高越严格）
- --max_gap: 允许连续活跃区间中间断点天数

运行（Windows venv）：
  E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\.venv\Scripts\python.exe \
    E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\scripts\trim_and_backtest_p2509_mainperiod.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


# --------- 回测：复用 verify_chan_pivot.py 的 Pandas 逻辑（参数化） ---------

class ChanPivotTesterPandas:
    def __init__(
        self,
        df_1m: pd.DataFrame,
        activate_atr: float = 1.5,
        trail_atr: float = 3.0,
        entry_filter_atr: float = 2.0,
        pivot_valid_range: int = 6,
        min_bi_gap: int = 4,
    ):
        self.df_1m = df_1m.reset_index(drop=True)

        self.ACTIVATE_ATR = float(activate_atr)
        self.TRAIL_ATR = float(trail_atr)
        self.ENTRY_FILTER_ATR = float(entry_filter_atr)
        self.PIVOT_VALID_RANGE = int(pivot_valid_range)
        self.MIN_BI_GAP = int(min_bi_gap)

        self.trades = []
        self.position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.trailing_active = False

        self.k_lines = []
        self.inclusion_dir = 0
        self.bi_points = []
        self.pivots = []
        self.pending_signal = None

        df_1m_idx = self.df_1m.set_index("datetime")
        df_1m_idx.index = pd.to_datetime(df_1m_idx.index)

        self.df_5m = (
            df_1m_idx.resample("5min", label="right", closed="right")
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
        signal = self.pending_signal
        if not signal:
            return

        if signal["type"] == "Buy":
            if row["low"] < signal["stop_base"]:
                self.pending_signal = None
                return
            if row["high"] > signal["trigger_price"]:
                fill = max(signal["trigger_price"], row["open"])
                if fill > row["high"]:
                    fill = row["close"]
                self._open_position(1, fill, signal["stop_base"])

        elif signal["type"] == "Sell":
            if row["high"] > signal["stop_base"]:
                self.pending_signal = None
                return
            if row["low"] < signal["trigger_price"]:
                fill = min(signal["trigger_price"], row["open"])
                if fill < row["low"]:
                    fill = row["close"]
                self._open_position(-1, fill, signal["stop_base"])

    def _open_position(self, direction, price, stop_base):
        self.position = direction
        self.entry_price = price
        self.stop_price = stop_base - 1 if direction == 1 else stop_base + 1
        self.pending_signal = None
        self.trailing_active = False

    def _check_exit(self, row):
        hit = False
        exit_px = 0
        if self.position == 1:
            if row["low"] <= self.stop_price:
                hit = True
                exit_px = row["open"] if row["open"] < self.stop_price else self.stop_price
        elif self.position == -1:
            if row["high"] >= self.stop_price:
                hit = True
                exit_px = row["open"] if row["open"] > self.stop_price else self.stop_price

        if hit:
            pnl = (exit_px - self.entry_price) * self.position
            self.trades.append({"time": row["datetime"], "type": "Stop/Trail", "pnl": pnl})
            self.position = 0

    def _update_trailing_stop(self, curr_bar):
        atr = curr_bar["atr"] if not np.isnan(curr_bar["atr"]) else 0
        pnl = (curr_bar["close"] - self.entry_price) * self.position
        if not self.trailing_active and pnl > self.ACTIVATE_ATR * atr:
            self.trailing_active = True
        if self.trailing_active:
            if self.position == 1:
                new = curr_bar["high"] - self.TRAIL_ATR * atr
                if new > self.stop_price:
                    self.stop_price = new
            else:
                new = curr_bar["low"] + self.TRAIL_ATR * atr
                if new < self.stop_price:
                    self.stop_price = new

    def _on_bar_close(self, curr_bar):
        bar = {
            "high": curr_bar["high"],
            "low": curr_bar["low"],
            "time": curr_bar.name,
            "diff": curr_bar["diff"],
            "atr": curr_bar["atr"],
            "diff_15m": curr_bar["diff_15m"],
            "dea_15m": curr_bar["dea_15m"],
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
            if cand["idx"] - last["idx"] >= self.MIN_BI_GAP:
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

        p_now = self.bi_points[-1]
        p_last = self.bi_points[-2]
        p_prev = self.bi_points[-3]

        is_bull = curr_bar["diff_15m"] > curr_bar["dea_15m"]
        is_bear = curr_bar["diff_15m"] < curr_bar["dea_15m"]

        sig = None
        if self.pivots:
            last_pivot = self.pivots[-1]
            if p_now["type"] == "bottom":
                if p_now["price"] > last_pivot["zg"] and p_last["price"] > last_pivot["zg"]:
                    if last_pivot["end_bi_idx"] >= len(self.bi_points) - self.PIVOT_VALID_RANGE and is_bull:
                        sig = "Buy"
            elif p_now["type"] == "top":
                if p_now["price"] < last_pivot["zd"] and p_last["price"] < last_pivot["zd"]:
                    if last_pivot["end_bi_idx"] >= len(self.bi_points) - self.PIVOT_VALID_RANGE and is_bear:
                        sig = "Sell"

        if not sig:
            if p_now["type"] == "bottom":
                div = p_now["data"]["diff"] > p_prev["data"]["diff"]
                if p_now["price"] > p_prev["price"] and div and is_bull:
                    sig = "Buy"
            elif p_now["type"] == "top":
                div = p_now["data"]["diff"] < p_prev["data"]["diff"]
                if p_now["price"] < p_prev["price"] and div and is_bear:
                    sig = "Sell"

        atr = curr_bar["atr"]
        if sig == "Buy":
            trig = p_now["data"]["high"]
            if (trig - p_now["price"]) < self.ENTRY_FILTER_ATR * atr:
                self.pending_signal = {"type": "Buy", "trigger_price": trig, "stop_base": p_now["price"]}
        elif sig == "Sell":
            trig = p_now["data"]["low"]
            if (p_now["price"] - trig) < self.ENTRY_FILTER_ATR * atr:
                self.pending_signal = {"type": "Sell", "trigger_price": trig, "stop_base": p_now["price"]}


# ----------------- 主力阶段裁剪：按 daily volume -----------------

@dataclass
class ActiveRange:
    start: pd.Timestamp
    end: pd.Timestamp
    threshold: float
    q: float
    days_total: int
    days_active: int


def find_active_range_by_daily_volume(
    df_1m: pd.DataFrame,
    q: float = 0.30,
    max_gap: int = 2,
) -> ActiveRange:
    """在日期层面找“活跃主区间”。

    - q: 以 daily_volume 的 q 分位作为阈值
    - max_gap: 允许连续区间内的断点天数（例如周末/节假日、缺数据）

    输出：最长连续“活跃日”区间的起止日期（包含端点）。
    """

    df = df_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.floor("D")

    daily = df.groupby("date")["volume"].sum().sort_index()
    daily = daily[daily > 0]

    if daily.empty:
        raise ValueError("daily volume empty")

    threshold = float(daily.quantile(q))
    active_days = daily[daily >= threshold].index

    # 最长连续区间（允许 gap）
    best: Tuple[pd.Timestamp, pd.Timestamp, int] | None = None

    start = None
    last = None
    length = 0

    def flush():
        nonlocal best, start, last, length
        if start is None or last is None:
            return
        if best is None or length > best[2]:
            best = (start, last, length)

    for d in active_days:
        if start is None:
            start = d
            last = d
            length = 1
            continue
        gap = (d - last).days
        if gap <= (max_gap + 1):
            # gap=1 表示严格连续；gap>1 但 <=max_gap+1 允许断点
            last = d
            length += 1
        else:
            flush()
            start = d
            last = d
            length = 1

    flush()

    assert best is not None
    start_d, end_d, _ = best

    return ActiveRange(
        start=pd.Timestamp(start_d),
        end=pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
        threshold=threshold,
        q=q,
        days_total=int(len(daily)),
        days_active=int(len(active_days)),
    )


def run_backtest(df: pd.DataFrame, label: str, params: dict) -> dict:
    tester = ChanPivotTesterPandas(df, **params)
    trades = tester.run()

    if trades.empty:
        return {"label": label, "pnl": 0.0, "trades": 0, "win": 0.0, "dd": 0.0, "bi": len(tester.bi_points), "pivot": len(tester.pivots)}

    pnl = float(trades["pnl"].sum())
    total = int(len(trades))
    win = float((trades["pnl"] > 0).mean() * 100)
    cumsum = trades["pnl"].cumsum()
    dd = float(abs((cumsum - cumsum.cummax()).min()))
    return {"label": label, "pnl": pnl, "trades": total, "win": win, "dd": dd, "bi": len(tester.bi_points), "pivot": len(tester.pivots)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.30, help="daily_volume 分位阈值，越大越严格")
    ap.add_argument("--max_gap", type=int, default=2, help="允许活跃日区间中断点天数")
    ap.add_argument("--save", action="store_true", help="保存裁剪后的CSV")
    return ap.parse_args()


def main():
    args = parse_args()

    fp = Path("E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse/p2509_1min_202503-202508.csv")
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # 参数：BASE / BEST
    base = dict(activate_atr=1.5, trail_atr=3.0, entry_filter_atr=2.0, pivot_valid_range=6, min_bi_gap=4)
    best = dict(activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5, pivot_valid_range=6, min_bi_gap=4)

    # 未裁剪
    s_base_all = run_backtest(df, "p2509_ALL_BASE", base)
    s_best_all = run_backtest(df, "p2509_ALL_BEST", best)

    # 裁剪
    ar = find_active_range_by_daily_volume(df, q=args.q, max_gap=args.max_gap)
    df_trim = df[(df["datetime"] >= ar.start) & (df["datetime"] <= ar.end)].copy()

    s_base_trim = run_backtest(df_trim, "p2509_TRIM_BASE", base)
    s_best_trim = run_backtest(df_trim, "p2509_TRIM_BEST", best)

    print("=" * 80)
    print("裁剪规则：按 daily_volume 分位数阈值筛活跃日，然后取最长连续活跃区间")
    print(f"q={ar.q}, threshold={ar.threshold:.0f}, days_total={ar.days_total}, days_active={ar.days_active}, max_gap={args.max_gap}")
    print(f"active range: {ar.start}  ->  {ar.end}")
    print("注意：这里不做日内时段过滤（与旧数据回测口径一致）")
    print("=" * 80)

    for s in [s_base_all, s_best_all, s_base_trim, s_best_trim]:
        print(f"{s['label']}: pnl={s['pnl']:.0f}, trades={s['trades']}, win={s['win']:.2f}%, dd={s['dd']:.0f}, bi={s['bi']}, pivot={s['pivot']}")

    if args.save:
        out = fp.with_name("p2509_1min_202503-202508_trimmed_by_volume.csv")
        df_trim.to_csv(out, index=False)
        print("saved:", out)


if __name__ == "__main__":
    main()
