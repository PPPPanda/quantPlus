"""
常量定义：CSV 列、路径、交易时间等.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]  # quantPlus/

# ── 录制文件根目录 ──────────────────────────────────────────
RECORDINGS_DIR = REPO_ROOT / "data" / "recordings"
BAR_1M_DIR = RECORDINGS_DIR / "bar_1m"
TICK_DIR = RECORDINGS_DIR / "tick"

# ── vntrader 配置 ──────────────────────────────────────────
VNTRADER_DIR = REPO_ROOT / ".vntrader"
RECORDER_SETTING_FILE = VNTRADER_DIR / "data_recorder_setting.json"
SYNC_STATE_FILE = VNTRADER_DIR / "recording_sync_state.json"

# ── 1m Bar CSV 列定义（对齐 Wind 数据格式）──────────────────
BAR_CSV_COLUMNS = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "turnover",
]

# ── Tick CSV 列定义 ─────────────────────────────────────────
TICK_CSV_COLUMNS = [
    "datetime",
    "last_price",
    "volume",
    "turnover",
    "open_interest",
    "last_volume",
    "limit_up",
    "limit_down",
    "open_price",
    "high_price",
    "low_price",
    "pre_close",
    "bid_price_1",
    "ask_price_1",
    "bid_volume_1",
    "ask_volume_1",
    "bid_price_2",
    "ask_price_2",
    "bid_volume_2",
    "ask_volume_2",
    "bid_price_3",
    "ask_price_3",
    "bid_volume_3",
    "ask_volume_3",
    "bid_price_4",
    "ask_price_4",
    "bid_volume_4",
    "ask_volume_4",
    "bid_price_5",
    "ask_price_5",
    "bid_volume_5",
    "ask_volume_5",
]

# ── 日期时间格式 ────────────────────────────────────────────
DT_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── 交易时段 ────────────────────────────────────────────────
# 夜盘 21:00~次日凌晨 归属次日交易日
NIGHT_SESSION_START_HOUR = 21


def trading_date(dt: datetime) -> str:
    """将行情时间转为交易日期字符串.

    夜盘（21:00 及之后）归属次日交易日。
    """
    if dt.hour >= NIGHT_SESSION_START_HOUR:
        return (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")
