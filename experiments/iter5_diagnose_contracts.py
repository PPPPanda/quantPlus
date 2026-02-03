"""iter5_diagnose_contracts.py

深度诊断 cta_chan_pivot 在 P2401 / P2601 上的失败模式。
- 基于 scripts/backtest_all_contracts.py 的回测复刻，增加：
  - 逐笔交易导出（含更多入场上下文）
  - 信号类型统计（胜率、均盈亏、盈亏比、PF）
  - 月度 PnL 拆解
  - P2401: 2B 亏损归因 & 亏损交易形态（追击/震荡止损）
  - P2601: bi_gap=4 vs 7 对比；找出 bi_gap=7 被过滤掉的交易及其贡献

输出：experiments/iter5_diagnosis.md
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
    enable_2b2s: bool = True


def load_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["datetime", "open", "high", "low", "close", "volume"]]


class ChanPivotTester:
    """从 backtest_all_contracts.py 拷贝并增强：在产生 pending_signal 时记录上下文，
    在开仓/平仓时写入 trades。"""

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
        self.entry_context: Dict = {}

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
        self._calc_extra_features()

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

    def _calc_extra_features(self):
        df = self.df_5m
        # 趋势/强度的简单代理：
        # 1) 60根(5m)均线斜率，2) 15m MACD 差值强度，3) 近N根涨跌幅
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma60_slope"] = df["ma60"].diff(5)  # 25分钟斜率
        df["macd15_gap"] = df["diff_15m"] - df["dea_15m"]
        df["ret_20"] = df["close"].pct_change(20)
        df["range_20_atr"] = (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / df["atr"]

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
                self.pending_signal = None
                return
            if row["high"] > s["trigger_price"]:
                fill = max(s["trigger_price"], row["open"])
                if fill > row["high"]:
                    fill = row["close"]
                self._open(1, fill, pd.to_datetime(row["datetime"]), s["stop_base"], s["signal_type"], s)
        elif s["type"] == "Sell":
            if row["high"] > s["stop_base"]:
                self.pending_signal = None
                return
            if row["low"] < s["trigger_price"]:
                fill = min(s["trigger_price"], row["open"])
                if fill < row["low"]:
                    fill = row["close"]
                self._open(-1, fill, pd.to_datetime(row["datetime"]), s["stop_base"], s["signal_type"], s)

    def _open(self, d, px, t, sb, st, sdict):
        self.position = d
        self.entry_price = float(px)
        self.entry_time = t
        self.entry_signal_type = st
        self.stop_price = float(sb - 1 if d == 1 else sb + 1)
        self.trailing_active = False

        ctx = {k: sdict.get(k) for k in [
            "sig_bar_time", "signal_type", "trigger_price", "stop_base",
            "atr", "macd15_gap", "ma60_slope", "ret_20", "range_20_atr",
            "bi_idx", "bi_type", "bi_price", "prev_bi_price", "prev2_bi_price",
            "pivot_zg", "pivot_zd", "pivot_age"
        ]}
        ctx["entry_time"] = t
        ctx["entry_price"] = float(px)
        ctx["direction"] = d
        self.entry_context = ctx

        self.pending_signal = None

    def _check_exit(self, row):
        hit = False
        exit_px = 0.0
        t = pd.to_datetime(row["datetime"])
        if self.position == 1 and row["low"] <= self.stop_price:
            hit = True
            exit_px = float(row["open"]) if row["open"] < self.stop_price else float(self.stop_price)
        elif self.position == -1 and row["high"] >= self.stop_price:
            hit = True
            exit_px = float(row["open"]) if row["open"] > self.stop_price else float(self.stop_price)

        if hit:
            pnl = (exit_px - self.entry_price) * self.position
            rec = {
                "entry_time": self.entry_time,
                "exit_time": t,
                "direction": self.position,
                "entry": self.entry_price,
                "exit": exit_px,
                "pnl": float(pnl),
                "signal_type": self.entry_signal_type,
                "stop_price": float(self.stop_price),
                "holding_min": float((t - self.entry_time).total_seconds() / 60.0) if self.entry_time else np.nan,
            }
            rec.update({f"ctx_{k}": v for k, v in self.entry_context.items() if k not in rec})

            # 亏损模式的粗分类（启发式）：
            # - chop_stop: 入场后很快止损 + 近期波动区间较小(横盘)
            # - chase_stop: 入场后很快止损 + 入场前20根(100min)波动区间很大(已走完一段)
            # - slow_bleed: 持仓久但仍止损
            atr = rec.get("ctx_atr")
            r20 = rec.get("ctx_range_20_atr")
            holding = rec.get("holding_min")
            mode = ""
            if pnl < 0:
                if holding is not None and holding <= 60:
                    if r20 is not None and pd.notna(r20):
                        if r20 >= 3.0:
                            mode = "chase_stop"
                        elif r20 <= 1.8:
                            mode = "chop_stop"
                        else:
                            mode = "fast_stop"
                    else:
                        mode = "fast_stop"
                elif holding is not None and holding > 180:
                    mode = "slow_bleed"
                else:
                    mode = "stop"
            else:
                mode = "win"
            rec["loss_mode"] = mode

            self.trades.append(rec)
            self.position = 0
            self.entry_time = None
            self.entry_signal_type = ""
            self.entry_context = {}

    def _update_trailing(self, bar):
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0:
            return
        pnl = (float(bar["close"]) - self.entry_price) * self.position
        if (not self.trailing_active) and pnl > self.p.activate_atr * atr:
            self.trailing_active = True
        if self.trailing_active:
            if self.position == 1:
                n = float(bar["high"]) - self.p.trail_atr * atr
                if n > self.stop_price:
                    self.stop_price = n
            else:
                n = float(bar["low"]) + self.p.trail_atr * atr
                if n < self.stop_price:
                    self.stop_price = n

    def _on_bar_close(self, bar):
        b = {
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "time": bar.name,
            "diff": float(bar["diff"]),
            "atr": float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0,
            "diff_15m": float(bar["diff_15m"]) if not np.isnan(bar["diff_15m"]) else 0.0,
            "dea_15m": float(bar["dea_15m"]) if not np.isnan(bar["dea_15m"]) else 0.0,
            "macd15_gap": float(bar["macd15_gap"]) if ("macd15_gap" in bar and not np.isnan(bar["macd15_gap"])) else 0.0,
            "ma60_slope": float(bar["ma60_slope"]) if ("ma60_slope" in bar and not np.isnan(bar["ma60_slope"])) else 0.0,
            "ret_20": float(bar["ret_20"]) if ("ret_20" in bar and not np.isnan(bar["ret_20"])) else 0.0,
            "range_20_atr": float(bar["range_20_atr"]) if ("range_20_atr" in bar and not np.isnan(bar["range_20_atr"])) else np.nan,
        }
        self._inclusion(b)
        if self._bi():
            self._signal(bar)

    def _inclusion(self, nb):
        if not self.k_lines:
            self.k_lines.append(nb)
            return
        last = self.k_lines[-1]
        il = nb["high"] <= last["high"] and nb["low"] >= last["low"]
        inw = last["high"] <= nb["high"] and last["low"] >= nb["low"]
        if il or inw:
            if self.inclusion_dir == 0:
                self.inclusion_dir = 1
            m = last.copy()
            m["time"] = nb["time"]
            # 直接覆盖最新指标
            for k in ["diff", "atr", "diff_15m", "dea_15m", "macd15_gap", "ma60_slope", "ret_20", "range_20_atr"]:
                if k in nb:
                    m[k] = nb[k]

            if self.inclusion_dir == 1:
                m["high"] = max(last["high"], nb["high"])
                m["low"] = max(last["low"], nb["low"])
            else:
                m["high"] = min(last["high"], nb["high"])
                m["low"] = min(last["low"], nb["low"])
            self.k_lines[-1] = m
        else:
            if nb["high"] > last["high"] and nb["low"] > last["low"]:
                self.inclusion_dir = 1
            elif nb["high"] < last["high"] and nb["low"] < last["low"]:
                self.inclusion_dir = -1
            self.k_lines.append(nb)

    def _bi(self):
        if len(self.k_lines) < 3:
            return None
        c, m2, l = self.k_lines[-1], self.k_lines[-2], self.k_lines[-3]
        cand = None
        if m2["high"] > l["high"] and m2["high"] > c["high"]:
            cand = {"type": "top", "price": m2["high"], "idx": len(self.k_lines) - 2, "data": m2}
        elif m2["low"] < l["low"] and m2["low"] < c["low"]:
            cand = {"type": "bottom", "price": m2["low"], "idx": len(self.k_lines) - 2, "data": m2}
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
        pts = self.bi_points[-4:]
        ranges = [
            (min(pts[i]["price"], pts[i + 1]["price"]), max(pts[i]["price"], pts[i + 1]["price"]))
            for i in range(3)
        ]
        zg = min(r[1] for r in ranges)
        zd = max(r[0] for r in ranges)
        if zg > zd:
            self.pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(self.bi_points) - 1})

    def _signal(self, bar):
        self._update_pivots()
        if len(self.bi_points) < 5:
            return
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else 0.0
        if atr <= 0:
            return

        pn, pl, pp = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        bull = float(bar["diff_15m"]) > float(bar["dea_15m"])  # 15m 多头
        bear = float(bar["diff_15m"]) < float(bar["dea_15m"])  # 15m 空头

        sig = None
        st = None
        trig = 0.0
        sb = 0.0
        lp = self.pivots[-1] if self.pivots else None

        # 3B/3S
        if lp:
            if pn["type"] == "bottom" and pn["price"] > lp["zg"] and pl["price"] > lp["zg"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bull:
                    sig = "Buy"
                    st = "3B"
                    trig = float(pn["data"]["high"])
                    sb = float(pn["price"])
            elif pn["type"] == "top" and pn["price"] < lp["zd"] and pl["price"] < lp["zd"]:
                if lp["end_bi_idx"] >= len(self.bi_points) - self.p.pivot_valid_range and bear:
                    sig = "Sell"
                    st = "3S"
                    trig = float(pn["data"]["low"])
                    sb = float(pn["price"])

        # 2B/2S
        if (not sig) and self.p.enable_2b2s:
            if pn["type"] == "bottom":
                if pn["price"] > pp["price"] and float(pn["data"]["diff"]) > float(pp["data"]["diff"]) and bull:
                    sig = "Buy"
                    st = "2B"
                    trig = float(pn["data"]["high"])
                    sb = float(pn["price"])
            elif pn["type"] == "top":
                if pn["price"] < pp["price"] and float(pn["data"]["diff"]) < float(pp["data"]["diff"]) and bear:
                    sig = "Sell"
                    st = "2S"
                    trig = float(pn["data"]["low"])
                    sb = float(pn["price"])

        if not sig:
            return

        # 入场幅度过滤
        if abs(trig - sb) >= self.p.entry_filter_atr * atr:
            return

        pivot_zg = lp["zg"] if lp else np.nan
        pivot_zd = lp["zd"] if lp else np.nan
        pivot_age = (len(self.bi_points) - 1) - lp["end_bi_idx"] if lp else np.nan

        self.pending_signal = {
            "type": sig,
            "trigger_price": trig,
            "stop_base": sb,
            "signal_type": st,
            "sig_bar_time": bar.name,
            "atr": float(bar["atr"]),
            "macd15_gap": float(bar.get("macd15_gap", 0.0)),
            "ma60_slope": float(bar.get("ma60_slope", 0.0)),
            "ret_20": float(bar.get("ret_20", 0.0)),
            "range_20_atr": float(bar.get("range_20_atr", np.nan)),
            "bi_idx": len(self.bi_points) - 1,
            "bi_type": pn["type"],
            "bi_price": float(pn["price"]),
            "prev_bi_price": float(pl["price"]),
            "prev2_bi_price": float(pp["price"]),
            "pivot_zg": float(pivot_zg) if pd.notna(pivot_zg) else np.nan,
            "pivot_zd": float(pivot_zd) if pd.notna(pivot_zd) else np.nan,
            "pivot_age": float(pivot_age) if pd.notna(pivot_age) else np.nan,
        }


def _profit_factor(x: pd.Series) -> float:
    gp = float(x[x > 0].sum())
    gl = float(abs(x[x < 0].sum()))
    return gp / gl if gl > 0 else float("inf")


def signal_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["signal_type", "n", "win%", "pnl", "avg", "avg_win", "avg_loss", "payoff", "pf"])
    g = trades.groupby("signal_type")
    out = g.agg(
        n=("pnl", "size"),
        pnl=("pnl", "sum"),
        win_pct=("pnl", lambda s: float((s > 0).mean() * 100)),
        avg=("pnl", "mean"),
        avg_win=("pnl", lambda s: float(s[s > 0].mean()) if (s > 0).any() else np.nan),
        avg_loss=("pnl", lambda s: float(s[s < 0].mean()) if (s < 0).any() else np.nan),
        pf=("pnl", _profit_factor),
    ).reset_index().rename(columns={"win_pct": "win%"})
    out["payoff"] = out["avg_win"] / out["avg_loss"].abs()
    return out.sort_values("pnl")


def monthly_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["month", "pnl", "n", "win%", "pf"])
    t = trades.copy()
    t["month"] = pd.to_datetime(t["exit_time"]).dt.to_period("M").astype(str)
    g = t.groupby("month")
    out = g.agg(
        pnl=("pnl", "sum"),
        n=("pnl", "size"),
        win_pct=("pnl", lambda s: float((s > 0).mean() * 100)),
        pf=("pnl", _profit_factor),
    ).reset_index().rename(columns={"win_pct": "win%"})
    return out.sort_values("month")


def key_contributors(trades: pd.DataFrame, topn=10) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    cols = ["entry_time", "exit_time", "direction", "signal_type", "pnl", "holding_min", "loss_mode",
            "ctx_sig_bar_time", "ctx_macd15_gap", "ctx_ma60_slope", "ctx_ret_20", "ctx_range_20_atr"]
    cols = [c for c in cols if c in t.columns]
    return t.sort_values("pnl", ascending=False)[cols].head(topn)


def match_missing_trades(trades_a: pd.DataFrame, trades_b: pd.DataFrame, tol_minutes=30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """找出 A 中存在但 B 中缺失的交易；A/B 都是逐笔交易表。

    注意：bi_gap 改变后，信号确认的时间可能整体后移，从而导致 entry_time 不可直接对齐。
    因此优先使用 `ctx_sig_bar_time`（信号确认的5m收盘时间）进行匹配；若不存在则退回 entry_time。

    匹配规则：time(±tol) + direction 相同 + signal_type 相同（尽量严格）。
    若 strict 匹配不到，再降级为仅 time+direction。
    """

    if trades_a.empty:
        return trades_a.copy(), trades_a.copy()

    A = trades_a.copy()
    B = trades_b.copy()

    # 优先用信号确认时间（5m bar close），否则退回 entry_time
    time_col_a = "ctx_sig_bar_time" if "ctx_sig_bar_time" in A.columns else "entry_time"
    time_col_b = "ctx_sig_bar_time" if "ctx_sig_bar_time" in B.columns else "entry_time"
    A["_t"] = pd.to_datetime(A[time_col_a])
    B["_t"] = pd.to_datetime(B[time_col_b])

    # 先 strict
    B_key = B[["_t", "direction", "signal_type"]].copy()
    B_key["_b_idx"] = np.arange(len(B_key))

    A2 = A[["_t", "direction", "signal_type"]].copy()
    A2["_a_idx"] = np.arange(len(A2))

    # 通过 merge_asof 做时间容忍匹配
    B_sorted = B_key.sort_values("_t")
    A_sorted = A2.sort_values("_t")

    merged = pd.merge_asof(
        A_sorted,
        B_sorted,
        on="_t",
        by=["direction", "signal_type"],
        tolerance=pd.Timedelta(minutes=tol_minutes),
        direction="nearest",
        suffixes=("_a", "_b"),
    )

    strict_missing = merged[merged["_b_idx"].isna()]

    # 对 strict_missing 再降级：只按 direction 匹配
    if len(strict_missing) > 0:
        B_key2 = B[["_t", "direction"]].copy()
        B_key2["_b2_idx"] = np.arange(len(B_key2))
        A_key2 = strict_missing[["_t", "direction", "_a_idx"]].copy()
        merged2 = pd.merge_asof(
            A_key2.sort_values("_t"),
            B_key2.sort_values("_t"),
            on="_t",
            by=["direction"],
            tolerance=pd.Timedelta(minutes=tol_minutes),
            direction="nearest",
        )
        missing_ids = merged2[merged2["_b2_idx"].isna()]["_a_idx"].astype(int).tolist()
    else:
        missing_ids = []

    missing = A.iloc[missing_ids].copy()
    kept = A.drop(index=missing.index).copy()
    return missing, kept


def fmt_df(df: pd.DataFrame, max_rows=60) -> str:
    if df is None or df.empty:
        return "(空)"
    if len(df) > max_rows:
        df2 = df.head(max_rows).copy()
        more = f"\n... (仅展示前 {max_rows} 行, 共 {len(df)} 行)"
    else:
        df2 = df.copy()
        more = ""
    # 数字列格式化
    for c in df2.columns:
        if pd.api.types.is_float_dtype(df2[c]) or pd.api.types.is_integer_dtype(df2[c]):
            if c in ["win%"]:
                df2[c] = df2[c].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")
            elif c in ["pnl", "avg", "avg_win", "avg_loss"]:
                df2[c] = df2[c].map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "")
            elif c in ["pf", "payoff"]:
                df2[c] = df2[c].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")
            elif c in ["holding_min", "ctx_macd15_gap", "ctx_ma60_slope", "ctx_ret_20", "ctx_range_20_atr"]:
                df2[c] = df2[c].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
    return df2.to_string(index=False) + more


def summarize_loss_modes(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["loss_mode", "n", "pnl", "avg_pnl", "median_hold_min"])
    t = trades.copy()
    g = t.groupby("loss_mode")
    out = g.agg(
        n=("pnl", "size"),
        pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        median_hold_min=("holding_min", "median"),
    ).reset_index()
    return out.sort_values("pnl")


def run_contract(data_dir: Path, symbol: str, params: StrategyParams) -> Tuple[pd.DataFrame, Path]:
    matches = list(data_dir.glob(f"{symbol}_1min_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No data file matches {symbol}_1min_*.csv in {data_dir}")
    fp = matches[0]
    df = load_csv(fp)
    tester = ChanPivotTester(df, params)
    trades = tester.run()
    return trades, fp


def main():
    data_dir = Path("/mnt/e/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    out_md = Path("/mnt/e/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/experiments/iter5_diagnosis.md")

    # BEST 参数（与 backtest_all_contracts.py 一致）
    best = StrategyParams("BEST", activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5, pivot_valid_range=6, min_bi_gap=4, enable_2b2s=True)

    # --- P2401 ---
    t2401, fp2401 = run_contract(data_dir, "p2401", best)

    # 导出逐笔
    out_trades_2401 = out_md.parent / "iter5_P2401_trades_best.csv"
    t2401.to_csv(out_trades_2401, index=False)

    s2401 = signal_stats(t2401)
    m2401 = monthly_pnl(t2401)
    lm2401 = summarize_loss_modes(t2401)

    t2401_2b = t2401[t2401["signal_type"] == "2B"].copy() if (not t2401.empty and "signal_type" in t2401.columns) else pd.DataFrame()
    s2401_2b = signal_stats(t2401_2b)
    m2401_2b = monthly_pnl(t2401_2b)

    # 2B 亏损交易Top（最差的几笔）
    worst_2b = pd.DataFrame()
    if not t2401_2b.empty:
        worst_2b = t2401_2b.sort_values("pnl").head(15)[[
            "entry_time", "exit_time", "direction", "pnl", "holding_min", "loss_mode",
            "ctx_sig_bar_time", "ctx_macd15_gap", "ctx_ma60_slope", "ctx_ret_20", "ctx_range_20_atr",
        ]]

    # --- P2601 ---
    p2601_gap4 = StrategyParams("P2601_gap4", activate_atr=best.activate_atr, trail_atr=best.trail_atr, entry_filter_atr=best.entry_filter_atr,
                               pivot_valid_range=best.pivot_valid_range, min_bi_gap=4, enable_2b2s=best.enable_2b2s)
    p2601_gap7 = StrategyParams("P2601_gap7", activate_atr=best.activate_atr, trail_atr=best.trail_atr, entry_filter_atr=best.entry_filter_atr,
                               pivot_valid_range=best.pivot_valid_range, min_bi_gap=7, enable_2b2s=best.enable_2b2s)

    t2601_g4, fp2601 = run_contract(data_dir, "p2601", p2601_gap4)
    t2601_g7, _ = run_contract(data_dir, "p2601", p2601_gap7)

    out_trades_2601_g4 = out_md.parent / "iter5_P2601_trades_gap4.csv"
    out_trades_2601_g7 = out_md.parent / "iter5_P2601_trades_gap7.csv"
    t2601_g4.to_csv(out_trades_2601_g4, index=False)
    t2601_g7.to_csv(out_trades_2601_g7, index=False)

    s2601_g4 = signal_stats(t2601_g4)
    s2601_g7 = signal_stats(t2601_g7)

    # gap4 中被 gap7 "砍掉" 的交易
    missing_2601, kept_2601 = match_missing_trades(t2601_g4, t2601_g7, tol_minutes=5)

    # missing 交易贡献拆解
    missing_stats = {
        "missing_n": int(len(missing_2601)),
        "missing_pnl": float(missing_2601["pnl"].sum()) if not missing_2601.empty else 0.0,
        "missing_win%": float((missing_2601["pnl"] > 0).mean() * 100) if not missing_2601.empty else 0.0,
        "missing_pf": _profit_factor(missing_2601["pnl"]) if not missing_2601.empty else 0.0,
    }

    # 找出最关键的“被砍掉的好交易”
    top_missing = pd.DataFrame()
    if not missing_2601.empty:
        top_missing = missing_2601.sort_values("pnl", ascending=False).head(20)[[
            "entry_time", "exit_time", "direction", "signal_type", "pnl", "holding_min", "loss_mode",
            "ctx_sig_bar_time", "ctx_macd15_gap", "ctx_ma60_slope", "ctx_ret_20", "ctx_range_20_atr",
        ]]

    # gap7 相比 gap4：哪些信号类型减少最多
    cnt4 = t2601_g4["signal_type"].value_counts() if (not t2601_g4.empty) else pd.Series(dtype=int)
    cnt7 = t2601_g7["signal_type"].value_counts() if (not t2601_g7.empty) else pd.Series(dtype=int)
    cnt_cmp = pd.DataFrame({"gap4_n": cnt4, "gap7_n": cnt7}).fillna(0).astype(int)
    cnt_cmp["drop_n"] = cnt_cmp["gap4_n"] - cnt_cmp["gap7_n"]
    cnt_cmp = cnt_cmp.sort_values("drop_n", ascending=False).reset_index().rename(columns={"index": "signal_type"})

    # 汇总
    def pnl_sum(df):
        return float(df["pnl"].sum()) if df is not None and not df.empty else 0.0

    md = []
    md.append("# Iter5 诊断：P2401 & P2601（cta_chan_pivot）\n")
    md.append("本报告由 `experiments/iter5_diagnose_contracts.py` 自动生成，回测逻辑基于 `scripts/backtest_all_contracts.py`，并在逐笔记录上增加了入场上下文特征用于诊断。\n")

    md.append("## 数据源\n")
    md.append(f"- P2401: `{fp2401.name}`\n")
    md.append(f"- P2601: `{fp2601.name}`\n")

    md.append("## 参数说明\n")
    md.append("- BEST（与 backtest_all_contracts.py 一致）：activate_atr=2.5, trail_atr=3.0, entry_filter_atr=2.5, pivot_valid_range=6, min_bi_gap=4, enable_2b2s=True\n")
    md.append("- P2601 对比：仅修改 min_bi_gap：4 vs 7（其余同 BEST）\n")

    md.append("\n---\n")
    md.append("## P2401 深度诊断（BEST）\n")
    md.append(f"逐笔导出：`{out_trades_2401.name}`\n\n")
    md.append(f"- 总交易数：{len(t2401)}\n")
    md.append(f"- 总PnL：{pnl_sum(t2401):.0f}\n")

    md.append("\n### 1) 信号类型统计\n")
    md.append("```\n" + fmt_df(s2401) + "\n```\n")

    md.append("\n### 2) 月度PnL拆解（按 exit_time 归属月份）\n")
    md.append("```\n" + fmt_df(m2401) + "\n```\n")

    md.append("\n### 3) 亏损交易形态（启发式分类）\n")
    md.append("分类逻辑：\n")
    md.append("- `chop_stop`：入场后<=60min 止损，且入场前 20 根(5m)区间/ATR <= 1.8（更像横盘反复打止损）\n")
    md.append("- `chase_stop`：入场后<=60min 止损，且入场前 20 根(5m)区间/ATR >= 3.0（更像追击趋势末端）\n")
    md.append("- `slow_bleed`：持仓>180min 后仍止损（更像钝刀子割肉）\n")
    md.append("```\n" + fmt_df(lm2401) + "\n```\n")

    md.append("\n### 4) 聚焦 2B：为什么在 P2401 上亏损\n")
    md.append(f"- 2B 笔数：{len(t2401_2b)}\n")
    md.append(f"- 2B 总PnL：{pnl_sum(t2401_2b):.0f}\n")
    md.append("\n2B 的信号统计：\n")
    md.append("```\n" + fmt_df(s2401_2b) + "\n```\n")
    md.append("\n2B 的月度分布：\n")
    md.append("```\n" + fmt_df(m2401_2b) + "\n```\n")
    md.append("\n2B 最差交易（按PnL升序）：\n")
    md.append("```\n" + fmt_df(worst_2b, max_rows=20) + "\n```\n")

    md.append("\n**读表提示**：\n")
    md.append("- `ctx_macd15_gap`：15m (diff-dea) 强度，越接近0越弱\n")
    md.append("- `ctx_ma60_slope`：5m MA60 斜率代理（>0偏多，<0偏空）\n")
    md.append("- `ctx_range_20_atr`：入场前20根(5m)的价格区间 / ATR，越大说明入场前已走出较大波段\n")

    md.append("\n---\n")
    md.append("## P2601 深度诊断：bi_gap 敏感性\n")
    md.append(f"逐笔导出：gap4=`{out_trades_2601_g4.name}`, gap7=`{out_trades_2601_g7.name}`\n\n")
    md.append(f"- gap4: trades={len(t2601_g4)}, pnl={pnl_sum(t2601_g4):.0f}\n")
    md.append(f"- gap7: trades={len(t2601_g7)}, pnl={pnl_sum(t2601_g7):.0f}\n")

    md.append("\n### 1) 信号类型统计对比\n")
    md.append("#### gap4\n")
    md.append("```\n" + fmt_df(s2601_g4) + "\n```\n")
    md.append("#### gap7\n")
    md.append("```\n" + fmt_df(s2601_g7) + "\n```\n")

    md.append("\n### 2) gap7 砍掉了哪些交易？（以 gap4 为基准）\n")
    md.append("信号数量变化（按 signal_type）：\n")
    md.append("```\n" + fmt_df(cnt_cmp) + "\n```\n")

    md.append("\n按 信号确认时间`ctx_sig_bar_time`（若无则退回entry_time，容忍±30min） + direction(+优先信号类型) 匹配，gap4 中缺失于 gap7 的交易：\n")
    md.append(f"- missing_n={missing_stats['missing_n']}\n")
    md.append(f"- missing_pnl={missing_stats['missing_pnl']:.0f}\n")
    md.append(f"- missing_win%={missing_stats['missing_win%']:.1f}%\n")
    md.append(f"- missing_PF={missing_stats['missing_pf']:.2f}\n")

    md.append("\n**被砍掉的交易中，贡献最大的 Top20（如果这些是好交易，被砍会导致 PnL 断崖）**：\n")
    md.append("```\n" + fmt_df(top_missing, max_rows=25) + "\n```\n")

    # --- 建议 ---
    md.append("\n---\n")
    md.append("## 失败模式总结（数据驱动）\n")

    # P2401 suggestions based on computed stats
    md.append("### P2401\n")
    if not s2401.empty:
        worst_sig = s2401.sort_values("pnl").head(1)
        best_sig = s2401.sort_values("pnl", ascending=False).head(1)
        md.append(f"- 亏损贡献最大的信号类型：{worst_sig.iloc[0]['signal_type']}（pnl={worst_sig.iloc[0]['pnl']:.0f}, n={int(worst_sig.iloc[0]['n'])}）\n")
        md.append(f"- 盈利贡献最大的信号类型：{best_sig.iloc[0]['signal_type']}（pnl={best_sig.iloc[0]['pnl']:.0f}, n={int(best_sig.iloc[0]['n'])}）\n")
    if not lm2401.empty:
        md.append("- 亏损模式占比（看 `loss_mode` 的n）：chop_stop vs chase_stop 的相对规模，能回答“震荡反复止损”还是“追趋势末端”更严重。\n")

    md.append("\n### P2601\n")
    md.append("- bi_gap 从 4→7 的核心影响不是单纯减少交易数，而是**减少了哪些类型的交易**、以及这些交易的净贡献（见 missing_pnl 和 Top missing 表）。\n")

    md.append("\n## 对策建议（可落地、以本诊断结果为依据）\n")
    md.append("1) **针对 P2401：给 2B 加一条“弱势禁止”过滤（先验证再上强逻辑）**\n")
    md.append("   - 如果 `2B` 的亏损集中在 `ctx_macd15_gap` 接近0 或 `ctx_ma60_slope<0` 的样本，说明 2B 在弱多/大级别下跌中抄底。\n")
    md.append("   - 建议先做最小改动：仅对 `2B` 要求 `ctx_macd15_gap` 大于某个分位数阈值，或要求 `ma60_slope>0`（仅用于 P2401/或对 2B 全局）。\n")
    md.append("2) **针对 P2401：分场景处理 chase_stop vs chop_stop**\n")
    md.append("   - 若 `chop_stop` 占比高：说明震荡区间反复打止损 → 考虑提高 entry_filter_atr / 加入“最近20根 range_20_atr 过小则不做”（横盘过滤）。\n")
    md.append("   - 若 `chase_stop` 占比高：说明追击末端 → 对 2B/3B 增加“入场前 range_20_atr 过大则不追”的反追击过滤。\n")
    md.append("3) **针对 P2601：不要用统一 bi_gap 砍交易；改为“动态 bi_gap / 或只对某类信号放宽”**\n")
    md.append("   - 如果 Top missing 里多数是 `3B/3S`（或某一类），说明 bi_gap=7 主要砍掉的是该类的好交易。\n")
    md.append("   - 可尝试：3B/3S 保持低 bi_gap（如4），2B/2S 提高 bi_gap（如7）来减少噪声，同时保住趋势腿。\n")
    md.append("4) **验证路径**\n")
    md.append("   - 先做单一改动（例如：仅对 2B 加 `ma60_slope>0`），重跑 7合约。\n")
    md.append("   - 用本脚本同样方式对比“被过滤掉的交易”是否主要是坏交易（missing_pnl 变负/或 Top missing 消失）。\n")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {out_md}")
    print(f"P2401 trades: {len(t2401)} -> {out_trades_2401}")
    print(f"P2601 trades: gap4 {len(t2601_g4)} / gap7 {len(t2601_g7)}")


if __name__ == "__main__":
    main()
