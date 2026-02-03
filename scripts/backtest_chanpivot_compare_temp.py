"""backtest_chanpivot_compare_temp.py

目标：
1) 对 work/quant/temp/data 下的 1min/5min 导出数据做统一清洗（标准化为 1min OHLCV）。
2) 自动识别“主要交易区间”（从数据本身推断）。
3) 在全量数据 vs 主要交易区间过滤数据 上运行 chan_pivot 回测，并输出统计。
4) 同时复跑 quantPlus/data/analyse 中的 p2509/p2601（原回测数据）做校验对比。

运行方式（Windows venv，推荐）：
  E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\.venv\Scripts\python.exe \
    E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\scripts\backtest_chanpivot_compare_temp.py

注：本脚本不依赖 openpyxl（避免 xlsx 解析问题），只读取 Excel 导出的 *_export.csv。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# =========================
# 1) 回测：复用 verify_chan_pivot.py 的 Pandas 批处理逻辑
# =========================

class ChanPivotTesterPandas:
    """使用原始脚本逻辑的测试器（与 qp/backtest/verify_chan_pivot.py 对齐），并支持参数化。

    注意：optimize_chan_pivot_results.csv 里优化的参数包括：
    - atr_trail (TRAIL_ATR)
    - atr_act   (ACTIVATE_ATR)
    - atr_entry (ENTRY_FILTER_ATR)
    - pivot_range (pivot_valid_range)

    这里实现为可配置，便于与优化结果对齐复现。
    """

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
        self.trades = []
        self.position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.trailing_active = False

        self.ACTIVATE_ATR = float(activate_atr)
        self.TRAIL_ATR = float(trail_atr)
        self.ENTRY_FILTER_ATR = float(entry_filter_atr)
        self.PIVOT_VALID_RANGE = int(pivot_valid_range)
        self.MIN_BI_GAP = int(min_bi_gap)

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

            if current_time.minute % 5 == 0:
                if current_time in self.df_5m.index:
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
                self._open_position(1, fill, row["datetime"], signal["stop_base"])

        elif signal["type"] == "Sell":
            if row["high"] > signal["stop_base"]:
                self.pending_signal = None
                return
            if row["low"] < signal["trigger_price"]:
                fill = min(signal["trigger_price"], row["open"])
                if fill < row["low"]:
                    fill = row["close"]
                self._open_position(-1, fill, row["datetime"], signal["stop_base"])

    def _open_position(self, direction, price, _time, stop_base):
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
            self._check_signal(curr_bar, new_bi)

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

    def _check_signal(self, curr_bar, _new_bi):
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


# =========================
# 2) 工具：读取 temp 导出数据并标准化
# =========================

def load_temp_export_as_1m(export_csv: Path) -> pd.DataFrame:
    """读取 Excel 导出的 *_export.csv（GBK），按固定列位解析成 1m bar。

    export 格式（已验证）：
      0=代码,1=名称,2=日期,3=开盘,4=最高,5=最低,6=收盘, ... ,10=成交量,11=持仓量

    这里返回：datetime, open, high, low, close, volume
    """
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

    # 去重：同一分钟重复时保留最后一条
    out = out.drop_duplicates(subset=["datetime"], keep="last")

    return out


@dataclass
class MainSessions:
    """主要交易区间：若跨午夜，用两段表示。"""

    ranges: List[Tuple[str, str]]  # [ ("09:00","11:30"), ... ]
    threshold: float
    total_days: int


def infer_main_trading_sessions(df_1m: pd.DataFrame, threshold: float = 0.6) -> MainSessions:
    """从数据本身推断“主要交易区间”。

    判断方法（可解释、可复现）：
    1) 只看有交易的分钟：volume>0
    2) 统计每个“分钟-of-day”(0~1439)在多少个交易日里出现过（presence）
    3) presence_ratio = 出现天数 / 总交易日数
    4) 将 presence_ratio >= threshold 的分钟视为“主要交易分钟”
    5) 把连续分钟合并成若干区间，输出 HH:MM-HH:MM

    这样得到的是“多数交易日都会出现数据”的时段，能自动适配不同品种/年份的日盘/夜盘结构。
    """

    df = df_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # 仅取有成交的分钟
    df = df[df["volume"].fillna(0) > 0]
    if df.empty:
        return MainSessions(ranges=[], threshold=threshold, total_days=0)

    df["date"] = df["datetime"].dt.date
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute

    days = df["date"].unique()
    total_days = len(days)

    # 每天每分钟出现过则记 1
    presence = df.drop_duplicates(subset=["date", "minute_of_day"])
    counts = presence.groupby("minute_of_day")["date"].nunique()
    ratio = (counts / total_days).reindex(range(1440), fill_value=0.0)

    main_minutes = ratio[ratio >= threshold].index.to_numpy()
    if len(main_minutes) == 0:
        # 降级：取 top 3 的 presence_ratio 连续区间
        top = ratio.sort_values(ascending=False).head(180).index.to_numpy()  # 约3小时
        main_minutes = np.sort(top)

    # 连续分钟合并
    ranges: List[Tuple[int, int]] = []
    start = main_minutes[0]
    prev = main_minutes[0]
    for m in main_minutes[1:]:
        if m == prev + 1:
            prev = m
        else:
            ranges.append((int(start), int(prev)))
            start = m
            prev = m
    ranges.append((int(start), int(prev)))

    def fmt(mm: int) -> str:
        h = mm // 60
        m = mm % 60
        return f"{h:02d}:{m:02d}"

    # 过滤太短的碎片（<20分钟）
    ranges = [r for r in ranges if (r[1] - r[0] + 1) >= 20]

    out_ranges = [(fmt(a), fmt(b)) for a, b in ranges]
    return MainSessions(ranges=out_ranges, threshold=threshold, total_days=total_days)


def filter_by_sessions(df_1m: pd.DataFrame, sessions: MainSessions) -> pd.DataFrame:
    if not sessions.ranges:
        return df_1m

    df = df_1m.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    mins = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    mask = pd.Series(False, index=df.index)

    def to_min(hhmm: str) -> int:
        hh, mm = hhmm.split(":")
        return int(hh) * 60 + int(mm)

    for a, b in sessions.ranges:
        ma = to_min(a)
        mb = to_min(b)
        if ma <= mb:
            mask = mask | ((mins >= ma) & (mins <= mb))
        else:
            # 跨午夜（理论上会出现）
            mask = mask | (mins >= ma) | (mins <= mb)

    return df.loc[mask].copy()


def calc_stats(trades: pd.DataFrame, bi_count: int, pivot_count: int) -> Dict:
    if trades.empty:
        return {"pnl": 0.0, "trades": 0, "win_rate": 0.0, "max_dd": 0.0, "bi": bi_count, "pivot": pivot_count}

    pnl = float(trades["pnl"].sum())
    total = int(len(trades))
    wins = int((trades["pnl"] > 0).sum())
    win_rate = wins / total * 100 if total else 0
    cumsum = trades["pnl"].cumsum()
    max_dd = float(abs((cumsum - cumsum.cummax()).min()))
    return {"pnl": pnl, "trades": total, "win_rate": float(win_rate), "max_dd": max_dd, "bi": bi_count, "pivot": pivot_count}


def run_one(
    name: str,
    df_1m: pd.DataFrame,
    *,
    activate_atr: float = 1.5,
    trail_atr: float = 3.0,
    entry_filter_atr: float = 2.0,
    pivot_valid_range: int = 6,
    min_bi_gap: int = 4,
) -> Dict:
    tester = ChanPivotTesterPandas(
        df_1m,
        activate_atr=activate_atr,
        trail_atr=trail_atr,
        entry_filter_atr=entry_filter_atr,
        pivot_valid_range=pivot_valid_range,
        min_bi_gap=min_bi_gap,
    )
    trades = tester.run()
    return calc_stats(trades, len(tester.bi_points), len(tester.pivots))


def print_block(title: str, stats: Dict):
    print(f"{title}: pnl={stats['pnl']:.0f}, trades={stats['trades']}, win={stats['win_rate']:.2f}%, dd={stats['max_dd']:.0f}, bi={stats['bi']}, pivot={stats['pivot']}")


def main():
    temp_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/temp/data")
    analyse_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")

    targets = [
        ("P2201", temp_dir / "P2201.DCE._export.csv"),
        ("P2205", temp_dir / "P2205.DCE._export.csv"),
        ("P2401", temp_dir / "P2401.DCE._export.csv"),
        ("P2405", temp_dir / "P2405.DCE._export.csv"),
    ]

    print("=" * 80)
    print("ChanPivot 回测：temp/data 1min 文件（全量 vs 主要交易区间）")
    print("主要交易区间判定：按 volume>0 的分钟，统计 minute-of-day 在交易日中的出现比例 >= threshold(默认0.6)")
    print("=" * 80)

    for sym, fp in targets:
        if not fp.exists():
            print(f"[SKIP] {sym} missing: {fp}")
            continue

        df_1m = load_temp_export_as_1m(fp)

        # 推断主要交易区间
        sessions = infer_main_trading_sessions(df_1m, threshold=0.6)
        print(f"\n--- {sym} ---")
        print(f"rows={len(df_1m)}, dt_range={df_1m['datetime'].min()} .. {df_1m['datetime'].max()}")
        print(f"main sessions (threshold={sessions.threshold}, days={sessions.total_days}): {sessions.ranges}")

        # 两套参数：
        # - baseline: verify_chan_pivot 默认 (act=1.5, trail=3.0, entry=2.0)
        # - best: 来自 optimize_chan_pivot_results.csv 的最佳附近 (act=2.5, trail=3.0, entry=2.5)
        param_sets = [
            ("BASE", dict(activate_atr=1.5, trail_atr=3.0, entry_filter_atr=2.0, pivot_valid_range=6, min_bi_gap=4)),
            ("BEST", dict(activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5, pivot_valid_range=6, min_bi_gap=4)),
        ]

        df_main = filter_by_sessions(df_1m, sessions)

        for tag, params in param_sets:
            stats_all = run_one(sym + "_ALL_" + tag, df_1m, **params)
            stats_main = run_one(sym + "_MAIN_" + tag, df_main, **params)
            print_block(f"ALL-{tag}", stats_all)
            print_block(f"MAIN-{tag}", stats_main)

    print("\n" + "=" * 80)
    print("复跑 quantPlus/data/analyse 旧数据（用于确认回测逻辑/数据格式一致）")
    print("=" * 80)

    # 复跑旧数据：p2509 / p2601
    for base in ["p2509_1min_202503-202508.csv", "p2601_1min_202507-202512.csv"]:
        fp = analyse_dir / base
        if not fp.exists():
            print(f"[SKIP] missing analyse file: {fp}")
            continue
        df = pd.read_csv(fp)
        df.columns = [c.strip() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        stats_base = run_one(base + "_BASE", df, activate_atr=1.5, trail_atr=3.0, entry_filter_atr=2.0)
        stats_best = run_one(base + "_BEST", df, activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5)
        print_block(base + " (BASE)", stats_base)
        print_block(base + " (BEST)", stats_best)


if __name__ == "__main__":
    main()
