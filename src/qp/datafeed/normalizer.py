"""
1分钟K线数据归一化模块.

将不同数据源（Wind / XTQuant / 其他）的1分钟K线归一化到统一口径：
- 只保留交易时段内 bars
- 剔除 session boundary 的集合竞价 bar
- 剔除零成交噪声 bar
- 时间戳标准化（:59 → 下一分钟 :00）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class SessionSpec:
    """交易时段定义."""
    start: time   # 时段开始时间（不含）
    end: time     # 时段结束时间（含）
    name: str = ""


# ============================================================
# 棕榈油期货交易时段（大商所）
# ============================================================
PALM_OIL_SESSIONS: List[SessionSpec] = [
    SessionSpec(start=time(21, 0), end=time(23, 0), name="night"),
    SessionSpec(start=time(9, 0), end=time(10, 15), name="am1"),
    SessionSpec(start=time(10, 30), end=time(11, 30), name="am2"),
    SessionSpec(start=time(13, 30), end=time(15, 0), name="pm"),
]


def _get_session_for_time(t: time, sessions: List[SessionSpec]) -> Optional[SessionSpec]:
    """判断一个日内时间属于哪个 session（使用 (start, end] 区间）."""
    for s in sessions:
        if s.end >= s.start:
            # 不跨午夜
            if t > s.start and t <= s.end:
                return s
        else:
            # 跨午夜（如 21:00-02:30）
            if t > s.start or t <= s.end:
                return s
    return None


def _session_bounds_for_dt(
    dt: datetime, s: SessionSpec
) -> Tuple[datetime, datetime]:
    """
    计算 dt 所属 session 的起止 datetime.
    
    返回 (session_start_dt, session_end_dt)，
    bar 的有效范围是 (start, end]。
    """
    d = dt.date()
    if s.end >= s.start:
        # 不跨午夜
        return (
            datetime.combine(d, s.start),
            datetime.combine(d, s.end),
        )
    else:
        # 跨午夜
        if dt.time() > s.start:
            return (
                datetime.combine(d, s.start),
                datetime.combine(d + timedelta(days=1), s.end),
            )
        else:
            return (
                datetime.combine(d - timedelta(days=1), s.start),
                datetime.combine(d, s.end),
            )


def compute_window_end(
    dt: datetime,
    sessions: List[SessionSpec],
    window_minutes: int,
) -> Optional[datetime]:
    """
    计算 dt 所属的 N 分钟窗口结束时间（session-aware）.
    
    窗口从 session_start 起按 window_minutes 对齐滚动，
    到 session_end 强制截断（不跨 session）。
    
    Args:
        dt: 1m bar 的时间戳（分钟结束时刻），可以是 tz-aware 或 naive
        sessions: 交易时段列表
        window_minutes: 窗口大小（5 或 15）
    
    Returns:
        窗口结束时间（naive datetime），若 dt 不在任何 session 内则返回 None
    """
    # 统一转为 naive datetime 以避免 tz 比较问题
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    
    t = dt.time() if isinstance(dt, datetime) else dt
    s = _get_session_for_time(t, sessions)
    if s is None:
        return None
    
    ss, ee = _session_bounds_for_dt(dt, s)
    
    # 验证 dt 在 (start, end] 范围内
    if not (dt > ss and dt <= ee):
        return None
    
    # 从 session_start 起算经过的分钟数
    elapsed_seconds = (dt - ss).total_seconds()
    elapsed_minutes = int(elapsed_seconds / 60)  # dt 是分钟结束时刻，elapsed >= 1
    
    # 计算窗口结束的分钟偏移：ceil(elapsed / window) * window
    win_end_min = ((elapsed_minutes + window_minutes - 1) // window_minutes) * window_minutes
    window_end = ss + timedelta(minutes=win_end_min)
    
    # 不超过 session_end（尾部截断）
    if window_end > ee:
        window_end = ee
    
    return window_end


def get_session_key(
    dt: datetime, sessions: List[SessionSpec]
) -> Optional[Tuple[datetime, datetime]]:
    """获取 dt 所属 session 的 (start, end) 作为唯一标识."""
    # 统一转为 naive
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    t = dt.time() if isinstance(dt, datetime) else dt
    s = _get_session_for_time(t, sessions)
    if s is None:
        return None
    return _session_bounds_for_dt(dt, s)


def normalize_1m_bars(
    df: pd.DataFrame,
    sessions: List[SessionSpec],
) -> pd.DataFrame:
    """
    将 1m K 线数据归一化到统一口径.
    
    处理步骤：
    1. 时间戳标准化（:59 → 下一分钟 :00，去重）
    2. 只保留交易时段内 (session_start, session_end] 的 bars
    3. 剔除 V=0 且 O=H=L=C 的噪声 bar
    
    Args:
        df: 原始 1m 数据，需包含 datetime, open, high, low, close, volume 列
        sessions: 交易时段列表
    
    Returns:
        归一化后的 DataFrame
    """
    if df.empty:
        return df.copy()
    
    df = df.copy()
    
    # 1. 时间戳标准化
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # :59 秒 → 下一分钟 :00
    mask_59 = df["datetime"].dt.second == 59
    if mask_59.any():
        df.loc[mask_59, "datetime"] = (
            df.loc[mask_59, "datetime"] + pd.Timedelta(seconds=1)
        )
    
    # 统一到分钟精度
    df["datetime"] = df["datetime"].dt.floor("min")
    
    # 去重（同一时间戳保留最后一条）
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # 2. 只保留交易时段内 (start, end] 的 bars
    def in_session(dt_val) -> bool:
        t = dt_val.time()
        return _get_session_for_time(t, sessions) is not None
    
    mask_session = df["datetime"].apply(in_session)
    df = df.loc[mask_session].copy()
    
    if df.empty:
        return df
    
    # 3. 剔除零成交噪声 bar（V=0 且 O=H=L=C）
    if "volume" in df.columns:
        same_price = (
            (df["open"] == df["high"])
            & (df["high"] == df["low"])
            & (df["low"] == df["close"])
        )
        noise_mask = (df["volume"] == 0) & same_price
        df = df.loc[~noise_mask].copy()
    
    return df.reset_index(drop=True)
