"""
CSV → 数据库增量同步器.

在 GUI DataRecorder 启动时调用，将 CSV 录制文件增量导入 database.db。
利用 vnpy SQLite 的 upsert (on_conflict_replace) 保证去重。

用法:
    # 作为模块直接调用（CLI 手动同步）
    python -m qp.recorder.db_sync

    # 在 RecorderEngine 中集成
    from qp.recorder.db_sync import sync_recordings_to_db
    stats = sync_recordings_to_db()
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from qp.common.logging import get_logger, setup_logging

from .constants import (
    BAR_1M_DIR,
    BAR_CSV_COLUMNS,
    DT_FORMAT,
    REPO_ROOT,
    SYNC_STATE_FILE,
    TICK_DIR,
)

if TYPE_CHECKING:
    from vnpy.trader.database import BaseDatabase
    from vnpy.trader.object import BarData, TickData

logger = get_logger(__name__)


def _load_sync_state() -> dict:
    """加载同步状态文件."""
    if not SYNC_STATE_FILE.exists():
        return {}
    try:
        with open(SYNC_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("同步状态文件损坏，将全量同步")
        return {}


def _save_sync_state(state: dict) -> None:
    """保存同步状态文件."""
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SYNC_STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(SYNC_STATE_FILE)


def _parse_vt_symbol(vt_symbol: str) -> tuple[str, str]:
    """解析 'p2605.DCE' -> ('p2605', 'DCE')."""
    parts = vt_symbol.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"无效的 vt_symbol: {vt_symbol}")
    return parts[0], parts[1]


def _csv_to_bars_incremental(
    csv_path: Path,
    symbol: str,
    exchange_str: str,
    after_dt: str | None = None,
) -> list:
    """
    读取 CSV 文件，返回 after_dt 之后的 BarData 列表.

    延迟导入 vnpy，避免在非 vnpy 环境中失败。
    """
    from vnpy.trader.constant import Exchange, Interval
    from vnpy.trader.object import BarData

    try:
        exchange = Exchange(exchange_str)
    except ValueError:
        logger.error("未知交易所: %s", exchange_str)
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("读取 CSV 失败: %s, 错误: %s", csv_path, e)
        return []

    if df.empty:
        return []

    # 验证列
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        logger.error("CSV 缺少必需列 %s: %s", missing, csv_path)
        return []

    # 增量过滤
    if after_dt:
        df = df[df["datetime"] > after_dt]

    if df.empty:
        return []

    bars: list[BarData] = []
    for _, row in df.iterrows():
        try:
            dt = datetime.strptime(str(row["datetime"]), DT_FORMAT)
        except (ValueError, TypeError) as e:
            logger.warning("跳过无效日期行: %s, 错误: %s", row["datetime"], e)
            continue

        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=Interval.MINUTE,
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            volume=float(row["volume"]),
            open_interest=float(row.get("open_interest", 0)),
            turnover=float(row.get("turnover", 0)),
            gateway_name="CSV",
        )
        bars.append(bar)

    return bars


def _csv_to_ticks_incremental(
    csv_path: Path,
    symbol: str,
    exchange_str: str,
    after_dt: str | None = None,
) -> list:
    """读取 Tick CSV 文件，返回 after_dt 之后的 TickData 列表."""
    from vnpy.trader.constant import Exchange
    from vnpy.trader.object import TickData

    try:
        exchange = Exchange(exchange_str)
    except ValueError:
        logger.error("未知交易所: %s", exchange_str)
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("读取 Tick CSV 失败: %s, 错误: %s", csv_path, e)
        return []

    if df.empty:
        return []

    required = {"datetime", "last_price", "volume"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Tick CSV 缺少必需列 %s: %s", missing, csv_path)
        return []

    if after_dt:
        df = df[df["datetime"] > after_dt]

    if df.empty:
        return []

    ticks: list[TickData] = []
    for _, row in df.iterrows():
        try:
            dt = datetime.strptime(str(row["datetime"]), DT_FORMAT)
        except (ValueError, TypeError) as e:
            logger.warning("跳过无效日期行: %s, 错误: %s", row["datetime"], e)
            continue

        tick = TickData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            name=symbol,
            last_price=float(row.get("last_price", 0)),
            volume=float(row.get("volume", 0)),
            turnover=float(row.get("turnover", 0)),
            open_interest=float(row.get("open_interest", 0)),
            last_volume=float(row.get("last_volume", 0)),
            limit_up=float(row.get("limit_up", 0)),
            limit_down=float(row.get("limit_down", 0)),
            open_price=float(row.get("open_price", 0)),
            high_price=float(row.get("high_price", 0)),
            low_price=float(row.get("low_price", 0)),
            pre_close=float(row.get("pre_close", 0)),
            bid_price_1=float(row.get("bid_price_1", 0)),
            ask_price_1=float(row.get("ask_price_1", 0)),
            bid_volume_1=float(row.get("bid_volume_1", 0)),
            ask_volume_1=float(row.get("ask_volume_1", 0)),
            bid_price_2=float(row.get("bid_price_2", 0)),
            ask_price_2=float(row.get("ask_price_2", 0)),
            bid_volume_2=float(row.get("bid_volume_2", 0)),
            ask_volume_2=float(row.get("ask_volume_2", 0)),
            bid_price_3=float(row.get("bid_price_3", 0)),
            ask_price_3=float(row.get("ask_price_3", 0)),
            bid_volume_3=float(row.get("bid_volume_3", 0)),
            ask_volume_3=float(row.get("ask_volume_3", 0)),
            bid_price_4=float(row.get("bid_price_4", 0)),
            ask_price_4=float(row.get("ask_price_4", 0)),
            bid_volume_4=float(row.get("bid_volume_4", 0)),
            ask_volume_4=float(row.get("ask_volume_4", 0)),
            bid_price_5=float(row.get("bid_price_5", 0)),
            ask_price_5=float(row.get("ask_price_5", 0)),
            bid_volume_5=float(row.get("bid_volume_5", 0)),
            ask_volume_5=float(row.get("ask_volume_5", 0)),
            gateway_name="CSV",
        )
        ticks.append(tick)

    return ticks


def _sync_directory(
    recordings_dir: Path,
    data_type: str,
    state: dict,
    database: "BaseDatabase",
) -> dict:
    """同步一个录制目录（bar_1m 或 tick）.

    Returns:
        {"synced": int, "skipped": int, "errors": int, "files": int}
    """
    stats = {"synced": 0, "skipped": 0, "errors": 0, "files": 0}

    if not recordings_dir.exists():
        return stats

    for symbol_dir in sorted(recordings_dir.iterdir()):
        if not symbol_dir.is_dir():
            continue

        vt_symbol = symbol_dir.name
        try:
            symbol, exchange_str = _parse_vt_symbol(vt_symbol)
        except ValueError as e:
            logger.warning("跳过无效目录: %s, %s", symbol_dir, e)
            stats["errors"] += 1
            continue

        for csv_file in sorted(symbol_dir.glob("*.csv")):
            state_key = f"{data_type}/{vt_symbol}/{csv_file.name}"
            last_dt = state.get(state_key)
            stats["files"] += 1

            try:
                if data_type == "bar_1m":
                    items = _csv_to_bars_incremental(
                        csv_file, symbol, exchange_str, last_dt
                    )
                    if items:
                        database.save_bar_data(items)
                else:
                    items = _csv_to_ticks_incremental(
                        csv_file, symbol, exchange_str, last_dt
                    )
                    if items:
                        database.save_tick_data(items)

                if items:
                    state[state_key] = items[-1].datetime.strftime(DT_FORMAT)
                    stats["synced"] += len(items)
                    logger.info(
                        "同步 %s: %d 条 (%s)",
                        csv_file.name,
                        len(items),
                        vt_symbol,
                    )
                else:
                    stats["skipped"] += 1

            except Exception:
                logger.exception("同步失败: %s", csv_file)
                stats["errors"] += 1

    return stats


def sync_recordings_to_db(
    bar_dir: Path | None = None,
    tick_dir: Path | None = None,
    database: "BaseDatabase | None" = None,
) -> dict:
    """
    将 CSV 录制文件增量同步到数据库.

    Args:
        bar_dir: 1m bar CSV 目录，默认 data/recordings/bar_1m
        tick_dir: tick CSV 目录，默认 data/recordings/tick
        database: vnpy 数据库实例，默认自动获取

    Returns:
        {
            "bar": {"synced": int, "skipped": int, "errors": int, "files": int},
            "tick": {"synced": int, "skipped": int, "errors": int, "files": int},
            "elapsed_ms": int,
        }
    """
    bar_dir = bar_dir or BAR_1M_DIR
    tick_dir = tick_dir or TICK_DIR

    if database is None:
        from vnpy.trader.database import get_database
        database = get_database()

    state = _load_sync_state()
    t0 = time.monotonic()

    bar_stats = _sync_directory(bar_dir, "bar_1m", state, database)
    tick_stats = _sync_directory(tick_dir, "tick", state, database)

    _save_sync_state(state)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    result = {
        "bar": bar_stats,
        "tick": tick_stats,
        "elapsed_ms": elapsed_ms,
    }

    total_synced = bar_stats["synced"] + tick_stats["synced"]
    total_errors = bar_stats["errors"] + tick_stats["errors"]
    logger.info(
        "同步完成: bar %d 条, tick %d 条, 错误 %d, 耗时 %dms",
        bar_stats["synced"],
        tick_stats["synced"],
        total_errors,
        elapsed_ms,
    )

    return result


def main() -> None:
    """CLI 入口: python -m qp.recorder.db_sync"""
    # 确保工作目录为仓库根目录
    os.chdir(REPO_ROOT)

    setup_logging(verbose=True)

    logger.info("开始手动同步 CSV → 数据库...")
    logger.info("仓库根目录: %s", REPO_ROOT)
    logger.info("Bar 目录: %s", BAR_1M_DIR)
    logger.info("Tick 目录: %s", TICK_DIR)

    result = sync_recordings_to_db()

    print("\n=== 同步结果 ===")
    print(f"Bar:  同步 {result['bar']['synced']} 条, "
          f"跳过 {result['bar']['skipped']} 文件, "
          f"错误 {result['bar']['errors']}")
    print(f"Tick: 同步 {result['tick']['synced']} 条, "
          f"跳过 {result['tick']['skipped']} 文件, "
          f"错误 {result['tick']['errors']}")
    print(f"耗时: {result['elapsed_ms']}ms")


if __name__ == "__main__":
    main()
