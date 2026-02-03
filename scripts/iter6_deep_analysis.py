"""iter6_deep_analysis.py — Multi-dimensional statistical comparison across contracts.

Goal: Find what makes P2401/P2601 different from profitable contracts (P2205/P2505/P2509),
using multiple technical and chan-theory indicators. Guide Iter6 strategy design.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import json
from datetime import datetime


def load_csv(fp):
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def compute_indicators(df_1m):
    """Compute a rich set of indicators on 5m bars."""
    df_idx = df_1m.set_index("datetime")
    df_idx.index = pd.to_datetime(df_idx.index)
    df = (
        df_idx.resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )

    # --- MACD 5m ---
    e1 = df["close"].ewm(span=12, adjust=False).mean()
    e2 = df["close"].ewm(span=26, adjust=False).mean()
    df["diff_5m"] = e1 - e2
    df["dea_5m"] = df["diff_5m"].ewm(span=9, adjust=False).mean()
    df["macd_5m"] = (df["diff_5m"] - df["dea_5m"]) * 2

    # --- MACD 15m ---
    df_15m = df.resample("15min", closed="right", label="right").agg({"close": "last"}).dropna()
    m1 = df_15m["close"].ewm(span=12, adjust=False).mean()
    m2 = df_15m["close"].ewm(span=26, adjust=False).mean()
    df_15m["diff_15m"] = m1 - m2
    df_15m["dea_15m"] = df_15m["diff_15m"].ewm(span=9, adjust=False).mean()
    df_15m["macd_15m"] = (df_15m["diff_15m"] - df_15m["dea_15m"]) * 2
    # Forward fill to 5m
    al = df_15m[["diff_15m", "dea_15m", "macd_15m"]].shift(1).reindex(df.index, method="ffill")
    df["diff_15m"] = al["diff_15m"]
    df["dea_15m"] = al["dea_15m"]
    df["macd_15m"] = al["macd_15m"]

    # --- ATR ---
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # --- RSI 14 ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # --- Bollinger Bands (20, 2) ---
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100  # % width
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])  # position in band

    # --- ADX (14) ---
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    # Zero out when opposite is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * plus_dm.rolling(14).sum() / tr14
    minus_di = 100 * minus_dm.rolling(14).sum() / tr14
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    df["adx"] = dx.rolling(14).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # --- MA slopes ---
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma5_slope"] = df["ma5"].pct_change(5) * 100
    df["ma20_slope"] = df["ma20"].pct_change(5) * 100

    # --- Volatility regime ---
    df["returns"] = df["close"].pct_change()
    df["realized_vol"] = df["returns"].rolling(20).std() * np.sqrt(252 * 48)  # annualized from 5m

    # --- Price momentum ---
    df["mom_10"] = df["close"].pct_change(10) * 100
    df["mom_20"] = df["close"].pct_change(20) * 100

    # --- MACD histogram direction change frequency ---
    macd_sign = np.sign(df["macd_5m"])
    df["macd_flip_20"] = (macd_sign != macd_sign.shift()).rolling(20).sum()

    # --- 15m MACD direction stability ---
    bull_15m = (df["diff_15m"] > df["dea_15m"]).astype(int)
    df["macd15_flip_20"] = (bull_15m != bull_15m.shift()).rolling(20).sum()

    return df


def chan_structure_analysis(df_1m, min_bi_gap=5):
    """Analyze Chan theory structural features: bi, pivot quality."""
    df_idx = df_1m.set_index("datetime")
    df_idx.index = pd.to_datetime(df_idx.index)
    df = (
        df_idx.resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )

    # ATR for amplitude measurement
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Inclusion processing + bi detection (simplified from strategy)
    k_lines = []
    inclusion_dir = 0
    bi_points = []
    pivots = []

    for idx, row in df.iterrows():
        nb = {"high": float(row["high"]), "low": float(row["low"]), "time": idx,
              "atr": float(row["atr"]) if not np.isnan(row["atr"]) else 0}

        # Inclusion
        if not k_lines:
            k_lines.append(nb)
            continue
        last = k_lines[-1]
        il = nb["high"] <= last["high"] and nb["low"] >= last["low"]
        inw = last["high"] <= nb["high"] and last["low"] >= nb["low"]
        if il or inw:
            if inclusion_dir == 0:
                inclusion_dir = 1
            m = last.copy()
            m["time"] = nb["time"]
            m["atr"] = nb["atr"]
            if inclusion_dir == 1:
                m["high"] = max(last["high"], nb["high"])
                m["low"] = max(last["low"], nb["low"])
            else:
                m["high"] = min(last["high"], nb["high"])
                m["low"] = min(last["low"], nb["low"])
            k_lines[-1] = m
        else:
            if nb["high"] > last["high"] and nb["low"] > last["low"]:
                inclusion_dir = 1
            elif nb["high"] < last["high"] and nb["low"] < last["low"]:
                inclusion_dir = -1
            k_lines.append(nb)

        # Bi detection
        if len(k_lines) < 3:
            continue
        c, m2, l = k_lines[-1], k_lines[-2], k_lines[-3]
        cand = None
        if m2["high"] > l["high"] and m2["high"] > c["high"]:
            cand = {"type": "top", "price": m2["high"], "idx": len(k_lines) - 2, "data": m2}
        elif m2["low"] < l["low"] and m2["low"] < c["low"]:
            cand = {"type": "bottom", "price": m2["low"], "idx": len(k_lines) - 2, "data": m2}
        if not cand:
            continue
        if not bi_points:
            bi_points.append(cand)
            continue
        last_bp = bi_points[-1]
        if last_bp["type"] == cand["type"]:
            if last_bp["type"] == "top" and cand["price"] > last_bp["price"]:
                bi_points[-1] = cand
            elif last_bp["type"] == "bottom" and cand["price"] < last_bp["price"]:
                bi_points[-1] = cand
        else:
            if cand["idx"] - last_bp["idx"] >= min_bi_gap:
                bi_points.append(cand)
                # Pivot detection
                if len(bi_points) >= 4:
                    pts = bi_points[-4:]
                    ranges = [
                        (min(pts[i]["price"], pts[i + 1]["price"]),
                         max(pts[i]["price"], pts[i + 1]["price"]))
                        for i in range(3)
                    ]
                    zg = min(r[1] for r in ranges)
                    zd = max(r[0] for r in ranges)
                    if zg > zd:
                        pivots.append({
                            "zg": zg, "zd": zd,
                            "width": zg - zd,
                            "end_bi_idx": len(bi_points) - 1,
                            "time": cand["data"]["time"]
                        })

    # Compute bi-level statistics
    bi_amplitudes = []
    bi_durations = []
    for i in range(1, len(bi_points)):
        amp = abs(bi_points[i]["price"] - bi_points[i - 1]["price"])
        dur = bi_points[i]["idx"] - bi_points[i - 1]["idx"]
        atr_at = bi_points[i]["data"].get("atr", 0)
        bi_amplitudes.append(amp / atr_at if atr_at > 0 else 0)
        bi_durations.append(dur)

    # Pivot overlap analysis
    pivot_overlaps = []
    pivot_widths = []
    pivot_lifespans = []
    for i in range(len(pivots)):
        pw = pivots[i]["width"]
        atr_est = pw  # rough
        pivot_widths.append(pw)
        if i > 0:
            a, b = pivots[i - 1], pivots[i]
            overlap_zg = min(a["zg"], b["zg"])
            overlap_zd = max(a["zd"], b["zd"])
            pivot_overlaps.append(1 if overlap_zg > overlap_zd else 0)
            pivot_lifespans.append(b["end_bi_idx"] - a["end_bi_idx"])

    # 2B condition analysis: count how many consecutive higher-bottom + stronger-diff patterns
    consecutive_2b_signals = 0
    consecutive_fails = 0
    for i in range(2, len(bi_points)):
        if bi_points[i]["type"] == "bottom" and i >= 2:
            prev_bottom_idx = None
            for j in range(i - 2, -1, -1):
                if bi_points[j]["type"] == "bottom":
                    prev_bottom_idx = j
                    break
            if prev_bottom_idx is not None:
                if bi_points[i]["price"] > bi_points[prev_bottom_idx]["price"]:
                    consecutive_2b_signals += 1

    return {
        "n_bis": len(bi_points),
        "n_pivots": len(pivots),
        "bi_amp_mean": np.mean(bi_amplitudes) if bi_amplitudes else 0,
        "bi_amp_median": np.median(bi_amplitudes) if bi_amplitudes else 0,
        "bi_amp_std": np.std(bi_amplitudes) if bi_amplitudes else 0,
        "bi_dur_mean": np.mean(bi_durations) if bi_durations else 0,
        "bi_dur_std": np.std(bi_durations) if bi_durations else 0,
        "pivot_overlap_rate": np.mean(pivot_overlaps) if pivot_overlaps else 0,
        "pivot_width_mean": np.mean(pivot_widths) if pivot_widths else 0,
        "pivot_width_std": np.std(pivot_widths) if pivot_widths else 0,
        "pivot_lifespan_mean": np.mean(pivot_lifespans) if pivot_lifespans else 0,
        "pivot_lifespan_std": np.std(pivot_lifespans) if pivot_lifespans else 0,
        "bi_amplitudes": bi_amplitudes,
        "bi_durations": bi_durations,
    }


def run_backtest_with_trade_context(df_1m, min_bi_gap=5, bi_amp_min_atr=1.5):
    """Run v4best-like backtest and record rich context for each trade."""
    df_idx = df_1m.set_index("datetime")
    df_idx.index = pd.to_datetime(df_idx.index)
    df = (
        df_idx.resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )

    # Compute indicators
    e1 = df["close"].ewm(span=12, adjust=False).mean()
    e2 = df["close"].ewm(span=26, adjust=False).mean()
    df["diff"] = e1 - e2
    df["dea"] = df["diff"].ewm(span=9, adjust=False).mean()
    df["macd"] = (df["diff"] - df["dea"]) * 2

    df_15m = df.resample("15min", closed="right", label="right").agg({"close": "last"}).dropna()
    m1 = df_15m["close"].ewm(span=12, adjust=False).mean()
    m2 = df_15m["close"].ewm(span=26, adjust=False).mean()
    df_15m["diff_15m"] = m1 - m2
    df_15m["dea_15m"] = df_15m["diff_15m"].ewm(span=9, adjust=False).mean()
    al = pd.DataFrame({"diff_15m": df_15m["diff_15m"], "dea_15m": df_15m["dea_15m"]}).shift(1).reindex(df.index, method="ffill")
    df["diff_15m"] = al["diff_15m"]
    df["dea_15m"] = al["dea_15m"]

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # ADX
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * plus_dm.rolling(14).sum() / tr14
    minus_di = 100 * minus_dm.rolling(14).sum() / tr14
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    df["adx"] = dx.rolling(14).mean()

    # BB
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_width"] = (4 * bb_std) / df["bb_mid"] * 100
    df["bb_pos"] = (df["close"] - (df["bb_mid"] - 2 * bb_std)) / (4 * bb_std)

    # MA
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    # Momentum
    df["mom_10"] = df["close"].pct_change(10) * 100
    df["mom_20"] = df["close"].pct_change(20) * 100

    # Returns volatility
    df["returns"] = df["close"].pct_change()
    df["rvol_20"] = df["returns"].rolling(20).std() * np.sqrt(252 * 48)

    # MACD flip frequency
    macd_sign = np.sign(df["macd"])
    df["macd_flip_20"] = (macd_sign != macd_sign.shift()).rolling(20).sum()

    # Now run simplified backtest with context recording
    k_lines = []
    inclusion_dir = 0
    bi_points = []
    pivots = []
    recent_diffs = []
    position = 0
    entry_price = 0.0
    entry_time = None
    stop_price = 0.0
    trailing_active = False
    entry_signal_type = ""
    pending_signal = None
    trades = []

    activate_atr = 1.5
    trail_atr = 2.0
    entry_filter_atr = 1.5
    macd_consistency = 3

    for bar_idx, (idx, row) in enumerate(df.iterrows()):
        atr = float(row["atr"]) if not np.isnan(row["atr"]) else 0
        close = float(row["close"])

        # Check entry
        if position == 0 and pending_signal:
            s = pending_signal
            if s["type"] == "Buy":
                if float(row["low"]) < s["stop_base"]:
                    pending_signal = None
                elif float(row["high"]) > s["trigger_price"]:
                    fill = max(s["trigger_price"], float(row["open"]))
                    position = 1
                    entry_price = fill
                    entry_time = idx
                    stop_price = s["stop_base"] - 1
                    entry_signal_type = s["signal_type"]
                    trailing_active = False
                    pending_signal = None
            elif s["type"] == "Sell":
                if float(row["high"]) > s["stop_base"]:
                    pending_signal = None
                elif float(row["low"]) < s["trigger_price"]:
                    fill = min(s["trigger_price"], float(row["open"]))
                    position = -1
                    entry_price = fill
                    entry_time = idx
                    stop_price = s["stop_base"] + 1
                    entry_signal_type = s["signal_type"]
                    trailing_active = False
                    pending_signal = None

        # Check exit
        if position != 0:
            hit = False
            exit_px = 0.0
            if position == 1 and float(row["low"]) <= stop_price:
                hit = True
                exit_px = float(row["open"]) if float(row["open"]) < stop_price else stop_price
            elif position == -1 and float(row["high"]) >= stop_price:
                hit = True
                exit_px = float(row["open"]) if float(row["open"]) > stop_price else stop_price
            if hit:
                pnl = (exit_px - entry_price) * position
                # Record trade with rich context at entry time
                entry_row = df.loc[entry_time] if entry_time in df.index else None
                ctx = {}
                if entry_row is not None:
                    for col in ["rsi", "adx", "bb_width", "bb_pos", "ma5", "ma20", "ma60",
                                "mom_10", "mom_20", "rvol_20", "macd_flip_20", "diff_15m", "dea_15m"]:
                        v = entry_row.get(col, np.nan) if hasattr(entry_row, "get") else getattr(entry_row, col, np.nan)
                        ctx[f"entry_{col}"] = float(v) if not np.isnan(v) else None
                    # MA alignment
                    ma5 = ctx.get("entry_ma5") or 0
                    ma20 = ctx.get("entry_ma20") or 0
                    ma60 = ctx.get("entry_ma60") or 0
                    ctx["entry_ma_aligned_bull"] = 1 if ma5 > ma20 > ma60 else 0
                    ctx["entry_ma_aligned_bear"] = 1 if ma5 < ma20 < ma60 else 0
                    ctx["entry_above_ma20"] = 1 if close > ma20 else 0
                    ctx["entry_above_ma60"] = 1 if close > ma60 else 0

                trades.append({
                    "entry_time": entry_time, "exit_time": idx,
                    "direction": position, "entry": entry_price, "exit": exit_px,
                    "pnl": pnl, "signal_type": entry_signal_type,
                    **ctx
                })
                position = 0

            # Trailing
            if position != 0 and atr > 0:
                pnl_now = (close - entry_price) * position
                if not trailing_active and pnl_now > activate_atr * atr:
                    trailing_active = True
                if trailing_active:
                    if position == 1:
                        n = float(row["high"]) - trail_atr * atr
                        if n > stop_price:
                            stop_price = n
                    else:
                        n = float(row["low"]) + trail_atr * atr
                        if n < stop_price:
                            stop_price = n

        # Structure update
        nb = {"high": float(row["high"]), "low": float(row["low"]), "time": idx,
              "diff": float(row["diff"]) if not np.isnan(row["diff"]) else 0,
              "atr": atr}
        if not k_lines:
            k_lines.append(nb)
        else:
            last = k_lines[-1]
            il = nb["high"] <= last["high"] and nb["low"] >= last["low"]
            inw = last["high"] <= nb["high"] and last["low"] >= nb["low"]
            if il or inw:
                if inclusion_dir == 0:
                    inclusion_dir = 1
                m = last.copy()
                m["time"] = nb["time"]
                m["diff"] = nb["diff"]
                m["atr"] = nb["atr"]
                if inclusion_dir == 1:
                    m["high"] = max(last["high"], nb["high"])
                    m["low"] = max(last["low"], nb["low"])
                else:
                    m["high"] = min(last["high"], nb["high"])
                    m["low"] = min(last["low"], nb["low"])
                k_lines[-1] = m
            else:
                if nb["high"] > last["high"] and nb["low"] > last["low"]:
                    inclusion_dir = 1
                elif nb["high"] < last["high"] and nb["low"] < last["low"]:
                    inclusion_dir = -1
                k_lines.append(nb)

        # Bi detection
        if len(k_lines) >= 3:
            c_k, m2_k, l_k = k_lines[-1], k_lines[-2], k_lines[-3]
            cand = None
            if m2_k["high"] > l_k["high"] and m2_k["high"] > c_k["high"]:
                cand = {"type": "top", "price": m2_k["high"], "idx": len(k_lines) - 2, "data": m2_k}
            elif m2_k["low"] < l_k["low"] and m2_k["low"] < c_k["low"]:
                cand = {"type": "bottom", "price": m2_k["low"], "idx": len(k_lines) - 2, "data": m2_k}
            if cand:
                if not bi_points:
                    bi_points.append(cand)
                else:
                    last_bp = bi_points[-1]
                    if last_bp["type"] == cand["type"]:
                        if last_bp["type"] == "top" and cand["price"] > last_bp["price"]:
                            bi_points[-1] = cand
                        elif last_bp["type"] == "bottom" and cand["price"] < last_bp["price"]:
                            bi_points[-1] = cand
                    else:
                        # Bi amp filter
                        amp = abs(cand["price"] - last_bp["price"])
                        if atr > 0 and amp < bi_amp_min_atr * atr:
                            pass  # filtered
                        elif cand["idx"] - last_bp["idx"] >= min_bi_gap:
                            bi_points.append(cand)
                            # Pivot
                            if len(bi_points) >= 4:
                                pts = bi_points[-4:]
                                ranges = [
                                    (min(pts[i_]["price"], pts[i_ + 1]["price"]),
                                     max(pts[i_]["price"], pts[i_ + 1]["price"]))
                                    for i_ in range(3)
                                ]
                                zg = min(r[1] for r in ranges)
                                zd = max(r[0] for r in ranges)
                                if zg > zd:
                                    pivots.append({"zg": zg, "zd": zd, "end_bi_idx": len(bi_points) - 1})

                            # Signal check
                            recent_diffs.append(nb["diff"])
                            if len(recent_diffs) > 10:
                                recent_diffs.pop(0)

                            if len(bi_points) >= 5 and atr > 0:
                                pn, pl, pp = bi_points[-1], bi_points[-2], bi_points[-3]
                                bull = float(row["diff_15m"]) > float(row["dea_15m"]) if not np.isnan(row["diff_15m"]) else False
                                bear = float(row["diff_15m"]) < float(row["dea_15m"]) if not np.isnan(row["diff_15m"]) else False

                                # Trend filter
                                d5 = float(row["diff"]) if not np.isnan(row["diff"]) else 0
                                if d5 <= 0:
                                    bull = False
                                if d5 >= 0:
                                    bear = False

                                # MACD consistency
                                if macd_consistency > 0 and len(recent_diffs) >= macd_consistency:
                                    recent = recent_diffs[-macd_consistency:]
                                    if bull and not all(d > 0 for d in recent):
                                        bull = False
                                    if bear and not all(d < 0 for d in recent):
                                        bear = False

                                sig = st = None
                                trig = sb = 0.0

                                lp = pivots[-1] if pivots else None
                                if lp:
                                    # 3B
                                    if pn["type"] == "bottom" and pn["price"] > lp["zg"] and pl["price"] > lp["zg"]:
                                        if lp["end_bi_idx"] >= len(bi_points) - 6 and bull:
                                            sig = "Buy"
                                            st = "3B"
                                            trig = float(pn["data"]["high"])
                                            sb = float(pn["price"])
                                    # 3S → only exit (disable_3s_short)
                                    if not sig and pn["type"] == "top" and pn["price"] < lp["zd"] and pl["price"] < lp["zd"]:
                                        if lp["end_bi_idx"] >= len(bi_points) - 6 and bear:
                                            if position == 1:
                                                exit_px = float(pn["data"]["low"])
                                                pnl_val = exit_px - entry_price
                                                entry_row = df.loc[entry_time] if entry_time in df.index else None
                                                ctx = {}
                                                if entry_row is not None:
                                                    for col in ["rsi", "adx", "bb_width", "bb_pos",
                                                                "mom_10", "mom_20", "rvol_20", "macd_flip_20"]:
                                                        v2 = entry_row.get(col, np.nan) if hasattr(entry_row, "get") else getattr(entry_row, col, np.nan)
                                                        ctx[f"entry_{col}"] = float(v2) if not np.isnan(v2) else None
                                                trades.append({
                                                    "entry_time": entry_time, "exit_time": idx,
                                                    "direction": 1, "entry": entry_price, "exit": exit_px,
                                                    "pnl": pnl_val, "signal_type": entry_signal_type + "_3S_exit",
                                                    **ctx
                                                })
                                                position = 0

                                # 2B/2S
                                if not sig:
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

                                if sig:
                                    if abs(trig - sb) < entry_filter_atr * atr:
                                        pending_signal = {
                                            "type": sig, "trigger_price": trig,
                                            "stop_base": sb, "signal_type": st
                                        }

    return pd.DataFrame(trades)


def main():
    data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
    contracts = ["p2201", "p2205", "p2401", "p2405", "p2505", "p2509", "p2601"]

    print("=" * 100)
    print("PART 1: MARKET REGIME INDICATORS BY CONTRACT")
    print("=" * 100)

    regime_stats = {}
    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches:
            continue
        df_1m = load_csv(matches[0])
        df = compute_indicators(df_1m)
        # Skip warmup
        df = df.iloc[100:]

        stats = {
            "atr_mean": df["atr"].mean(),
            "atr_std": df["atr"].std(),
            "adx_mean": df["adx"].mean(),
            "adx_median": df["adx"].median(),
            "adx_pct_above_25": (df["adx"] > 25).mean() * 100,
            "adx_pct_above_40": (df["adx"] > 40).mean() * 100,
            "rsi_mean": df["rsi"].mean(),
            "rsi_std": df["rsi"].std(),
            "bb_width_mean": df["bb_width"].mean(),
            "bb_width_std": df["bb_width"].std(),
            "rvol_mean": df["realized_vol"].mean(),
            "rvol_std": df["realized_vol"].std(),
            "macd5_flip_rate": df["macd_flip_20"].mean(),
            "macd15_flip_rate": df["macd15_flip_20"].mean(),
            "mom_10_mean": df["mom_10"].mean(),
            "mom_10_std": df["mom_10"].std(),
            "mom_20_mean": df["mom_20"].mean(),
            "mom_20_std": df["mom_20"].std(),
            "ma5_slope_mean": df["ma5_slope"].mean(),
            "ma5_slope_std": df["ma5_slope"].std(),
            "pct_above_ma20": (df["close"] > df["ma20"]).mean() * 100,
            "pct_above_ma60": (df["close"] > df["ma60"]).mean() * 100,
            "price_range_pct": (df["close"].max() - df["close"].min()) / df["close"].mean() * 100,
        }
        regime_stats[c.upper()] = stats
        print(f"\n--- {c.upper()} ---")
        for k, v in stats.items():
            print(f"  {k:30s}: {v:>10.3f}")

    print("\n\n" + "=" * 100)
    print("PART 2: CHAN THEORY STRUCTURE BY CONTRACT (bi_gap=5)")
    print("=" * 100)

    chan_stats = {}
    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches:
            continue
        df_1m = load_csv(matches[0])
        cs = chan_structure_analysis(df_1m, min_bi_gap=5)
        chan_stats[c.upper()] = cs
        print(f"\n--- {c.upper()} ---")
        for k, v in cs.items():
            if k in ("bi_amplitudes", "bi_durations"):
                continue
            print(f"  {k:30s}: {v:>10.3f}")

    print("\n\n" + "=" * 100)
    print("PART 3: TRADE-LEVEL INDICATOR ANALYSIS (v4best params)")
    print("=" * 100)

    all_trades = []
    for c in contracts:
        matches = list(data_dir.glob(f"{c}_1min_*.csv"))
        if not matches:
            continue
        df_1m = load_csv(matches[0])
        trades_df = run_backtest_with_trade_context(df_1m)
        if not trades_df.empty:
            trades_df["contract"] = c.upper()
            all_trades.append(trades_df)
            print(f"\n--- {c.upper()} ({len(trades_df)} trades, PnL={trades_df['pnl'].sum():.0f}) ---")
            for st in sorted(trades_df["signal_type"].unique()):
                sub = trades_df[trades_df["signal_type"] == st]
                print(f"  {st}: n={len(sub)}, pnl={sub['pnl'].sum():.0f}, "
                      f"win={100*(sub['pnl']>0).mean():.1f}%")

    if all_trades:
        all_df = pd.concat(all_trades, ignore_index=True)

        print("\n\n" + "=" * 100)
        print("PART 4: WINNING vs LOSING TRADE INDICATOR COMPARISON")
        print("=" * 100)

        indicator_cols = [c for c in all_df.columns if c.startswith("entry_") and c != "entry_time"]

        # Compare across contract groups
        profitable = ["P2205", "P2505", "P2509"]
        weak = ["P2401", "P2201"]

        for sig_type in ["2B", "2S", "3B"]:
            print(f"\n{'='*80}")
            print(f"Signal Type: {sig_type}")
            print(f"{'='*80}")

            # Profitable contracts
            prof_trades = all_df[(all_df["contract"].isin(profitable)) & (all_df["signal_type"] == sig_type)]
            weak_trades = all_df[(all_df["contract"].isin(weak)) & (all_df["signal_type"] == sig_type)]

            if prof_trades.empty or weak_trades.empty:
                print(f"  Insufficient data for comparison")
                continue

            print(f"  Profitable group ({profitable}): n={len(prof_trades)}, "
                  f"pnl={prof_trades['pnl'].sum():.0f}, win={100*(prof_trades['pnl']>0).mean():.1f}%")
            print(f"  Weak group ({weak}): n={len(weak_trades)}, "
                  f"pnl={weak_trades['pnl'].sum():.0f}, win={100*(weak_trades['pnl']>0).mean():.1f}%")

            for col in indicator_cols:
                pv = prof_trades[col].dropna()
                wv = weak_trades[col].dropna()
                if len(pv) < 5 or len(wv) < 5:
                    continue
                diff = pv.mean() - wv.mean()
                # Simple effect size
                pooled_std = np.sqrt((pv.std()**2 + wv.std()**2) / 2)
                effect = diff / pooled_std if pooled_std > 0 else 0
                if abs(effect) > 0.3:  # Only show meaningful differences
                    print(f"  *** {col:35s}: prof={pv.mean():>8.2f} vs weak={wv.mean():>8.2f} "
                          f"(diff={diff:>+8.2f}, effect={effect:>+.2f})")

            # Also compare winning vs losing within the WEAK group
            print(f"\n  Within WEAK group - Winners vs Losers:")
            winners = weak_trades[weak_trades["pnl"] > 0]
            losers = weak_trades[weak_trades["pnl"] <= 0]
            if len(winners) >= 3 and len(losers) >= 3:
                for col in indicator_cols:
                    wv = winners[col].dropna()
                    lv = losers[col].dropna()
                    if len(wv) < 3 or len(lv) < 3:
                        continue
                    diff = wv.mean() - lv.mean()
                    pooled_std = np.sqrt((wv.std()**2 + lv.std()**2) / 2)
                    effect = diff / pooled_std if pooled_std > 0 else 0
                    if abs(effect) > 0.3:
                        print(f"    *** {col:33s}: win={wv.mean():>8.2f} vs lose={lv.mean():>8.2f} "
                              f"(effect={effect:>+.2f})")

        # P2601 specific: bi_gap sensitivity deep dive
        print("\n\n" + "=" * 100)
        print("PART 5: P2601 SPECIFIC ANALYSIS")
        print("=" * 100)
        p2601_trades = all_df[all_df["contract"] == "P2601"]
        if not p2601_trades.empty:
            for st in sorted(p2601_trades["signal_type"].unique()):
                sub = p2601_trades[p2601_trades["signal_type"] == st]
                winners = sub[sub["pnl"] > 0]
                losers = sub[sub["pnl"] <= 0]
                print(f"\n  {st}: n={len(sub)}, pnl={sub['pnl'].sum():.0f}, "
                      f"win={100*(sub['pnl']>0).mean():.1f}%")
                for col in indicator_cols:
                    wv = winners[col].dropna()
                    lv = losers[col].dropna()
                    if len(wv) < 2 or len(lv) < 2:
                        continue
                    diff = wv.mean() - lv.mean()
                    pooled_std = np.sqrt((wv.std()**2 + lv.std()**2) / 2)
                    effect = diff / pooled_std if pooled_std > 0 else 0
                    if abs(effect) > 0.4:
                        print(f"    *** {col:33s}: win={wv.mean():>8.2f} vs lose={lv.mean():>8.2f} "
                              f"(effect={effect:>+.2f})")

        # Save all trades for further analysis
        out_dir = Path("experiments") / "iter6_diagnosis"
        out_dir.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(out_dir / "all_trades_with_context.csv", index=False)
        with open(out_dir / "regime_stats.json", "w") as f:
            json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in regime_stats.items()}, f, indent=2)
        with open(out_dir / "chan_stats.json", "w") as f:
            json.dump({k: {kk: (float(vv) if not isinstance(vv, list) else [float(x) for x in vv])
                           for kk, vv in v.items()} for k, v in chan_stats.items()}, f, indent=2)
        print(f"\nAll data saved to {out_dir}")


if __name__ == "__main__":
    main()
