"""trim_temp_1min_like_old_and_backtest.py

用户需求（修正理解）：
- 旧数据 p2509_1min_202503-202508.csv 不改。
- 对 temp/data 的 4 份新数据（P2201/P2205/P2401/P2405）做“与旧数据一致”的处理，再回测。

这里“与旧数据一致”的含义（按你这轮描述）：
1) 数据格式：统一为 1min OHLCV，列名/类型与旧数据一致（datetime, open, high, low, close, volume）。
2) 不做日内时段过滤（旧数据也没做）。
3) 只做“主力/活跃期”裁剪：按 daily_volume（每日成交量）剔除首尾低成交阶段，保留最长连续活跃区间。

输出：
- 每个合约：
  - ALL（不裁剪）BASE/BEST
  - TRIM（裁剪主力期）BASE/BEST
  - 裁剪区间、阈值、活跃日比例

运行（Windows venv）：
  E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\.venv\Scripts\python.exe \
    E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\scripts\trim_temp_1min_like_old_and_backtest.py

可调参数：
  --q 0.30        # daily_volume 分位阈值
  --max_gap 2     # 活跃区间允许断点天数
  --save          # 保存裁剪后的 CSV 到 temp/data/derived/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd


# ----------------- 回测：chan_pivot Pandas 版本（参数化） -----------------

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


# ----------------- 数据：读取 temp export csv 并标准化为 1min -----------------

def load_temp_export_1m(export_csv: Path) -> pd.DataFrame:
    """读取 Excel 导出的 *_export.csv（GBK）并统一为旧数据格式。"""
    df = pd.read_csv(export_csv, encoding="gbk")
    dt = pd.to_datetime(df.iloc[:, 2], errors="coerce")

    out = pd.DataFrame(
        {
            "datetime": dt,
            "open": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
            "high": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
            "low": pd.to_numeric(df.iloc[:, 5], errors="coerce"),
            "close": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
            "volume": pd.to_numeric(df.iloc[:, 10], errors="coerce").fillna(0),
        }
    )

    out = out.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime")
    out = out.drop_duplicates(subset=["datetime"], keep="last")
    return out


# ----------------- 日期级别裁剪：按 daily volume 找最长连续活跃区间 -----------------

@dataclass
class ActiveRange:
    start: pd.Timestamp
    end: pd.Timestamp
    threshold: float
    q: float
    days_total: int
    days_active: int


def find_active_range_by_daily_volume(df_1m: pd.DataFrame, q: float = 0.30, max_gap: int = 2) -> ActiveRange:
    df = df_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.floor("D")

    daily = df.groupby("date")["volume"].sum().sort_index()
    daily = daily[daily > 0]
    if daily.empty:
        raise ValueError("daily volume empty")

    threshold = float(daily.quantile(q))
    active_days = daily[daily >= threshold].index

    best = None
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
            last = d
            length += 1
        else:
            flush()
            start = d
            last = d
            length = 1

    flush()
    assert best is not None

    s, e, _ = best
    return ActiveRange(
        start=pd.Timestamp(s),
        end=pd.Timestamp(e) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
        threshold=threshold,
        q=q,
        days_total=int(len(daily)),
        days_active=int(len(active_days)),
    )


def stats_from_trades(trades: pd.DataFrame, bi: int, pivot: int) -> Dict:
    if trades.empty:
        return {"pnl": 0.0, "trades": 0, "win": 0.0, "dd": 0.0, "bi": bi, "pivot": pivot}

    pnl = float(trades["pnl"].sum())
    total = int(len(trades))
    win = float((trades["pnl"] > 0).mean() * 100)
    cumsum = trades["pnl"].cumsum()
    dd = float(abs((cumsum - cumsum.cummax()).min()))
    return {"pnl": pnl, "trades": total, "win": win, "dd": dd, "bi": bi, "pivot": pivot}


def run_backtest(df: pd.DataFrame, params: dict) -> Dict:
    tester = ChanPivotTesterPandas(df, **params)
    trades = tester.run()
    return stats_from_trades(trades, len(tester.bi_points), len(tester.pivots))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.30)
    ap.add_argument("--max_gap", type=int, default=2)
    ap.add_argument("--save", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    temp_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/temp/data")
    out_dir = temp_dir / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "P2201": temp_dir / "P2201.DCE._export.csv",
        "P2205": temp_dir / "P2205.DCE._export.csv",
        "P2401": temp_dir / "P2401.DCE._export.csv",
        "P2405": temp_dir / "P2405.DCE._export.csv",
    }

    params_base = dict(activate_atr=1.5, trail_atr=3.0, entry_filter_atr=2.0, pivot_valid_range=6, min_bi_gap=4)
    params_best = dict(activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5, pivot_valid_range=6, min_bi_gap=4)

    print("=" * 90)
    print("temp 1min 新数据 -> 统一格式(与旧数据一致) -> 按 daily_volume 裁剪主力期 -> 回测")
    print("说明：不做日内时段过滤；只裁剪首尾低成交日（更贴近主力阶段）。")
    print(f"q={args.q}, max_gap={args.max_gap}")
    print("=" * 90)

    for sym, fp in targets.items():
        if not fp.exists():
            print(f"[SKIP] missing: {fp}")
            continue

        df = load_temp_export_1m(fp)
        ar = find_active_range_by_daily_volume(df, q=args.q, max_gap=args.max_gap)
        df_trim = df[(df["datetime"] >= ar.start) & (df["datetime"] <= ar.end)].copy()

        s_all_base = run_backtest(df, params_base)
        s_all_best = run_backtest(df, params_best)
        s_trim_base = run_backtest(df_trim, params_base)
        s_trim_best = run_backtest(df_trim, params_best)

        print(f"\n--- {sym} ---")
        print(f"rows_all={len(df)}, dt={df['datetime'].min()}..{df['datetime'].max()}")
        print(f"active_range: {ar.start.date()} -> {ar.end.date()} | daily_vol_threshold(q={ar.q})={ar.threshold:.0f} | days_total={ar.days_total} active_days={ar.days_active}")
        print(f"rows_trim={len(df_trim)}, dt_trim={df_trim['datetime'].min()}..{df_trim['datetime'].max()}")

        print(f"ALL-BASE:  pnl={s_all_base['pnl']:.0f}, trades={s_all_base['trades']}, win={s_all_base['win']:.2f}%, dd={s_all_base['dd']:.0f}")
        print(f"ALL-BEST:  pnl={s_all_best['pnl']:.0f}, trades={s_all_best['trades']}, win={s_all_best['win']:.2f}%, dd={s_all_best['dd']:.0f}")
        print(f"TRIM-BASE: pnl={s_trim_base['pnl']:.0f}, trades={s_trim_base['trades']}, win={s_trim_base['win']:.2f}%, dd={s_trim_base['dd']:.0f}")
        print(f"TRIM-BEST: pnl={s_trim_best['pnl']:.0f}, trades={s_trim_best['trades']}, win={s_trim_best['win']:.2f}%, dd={s_trim_best['dd']:.0f}")

        if args.save:
            out_all = out_dir / f"{sym}_1min_like_old_ALL.csv"
            out_trim = out_dir / f"{sym}_1min_like_old_TRIM_q{args.q:.2f}.csv"
            df.to_csv(out_all, index=False)
            df_trim.to_csv(out_trim, index=False)
            print("saved:", out_all)
            print("saved:", out_trim)


if __name__ == "__main__":
    main()
