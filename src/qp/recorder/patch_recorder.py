"""
GUI DataRecorder 启动时自动触发 DB Sync 的 monkey-patch.

用法（在 vntrader 启动脚本中，add_app 之后调用）:

    from qp.recorder.patch_recorder import patch_recorder_engine
    patch_recorder_engine(main_engine)

效果：
    RecorderEngine.start() 被包装，启动前先执行 CSV → DB 同步。
"""

from __future__ import annotations

import functools
from pathlib import Path

from qp.common.logging import get_logger

logger = get_logger(__name__)


def patch_recorder_engine(main_engine) -> bool:
    """
    对已加载的 RecorderEngine 进行 monkey-patch.

    在 start() 前注入 sync_recordings_to_db() 调用。

    Args:
        main_engine: vnpy MainEngine 实例（已经 add_app(DataRecorderApp) 之后）

    Returns:
        True 如果成功 patch，False 如果未找到 RecorderEngine
    """
    try:
        recorder_engine = main_engine.get_engine("DataRecorder")
    except Exception:
        logger.warning("未找到 DataRecorder 引擎，跳过 DB Sync patch")
        return False

    if recorder_engine is None:
        logger.warning("DataRecorder 引擎为 None，跳过 DB Sync patch")
        return False

    # 保存原始 start 方法
    original_start = recorder_engine.start

    @functools.wraps(original_start)
    def patched_start() -> None:
        """包装后的 start：先同步 CSV → DB，再启动原始录制."""
        from qp.recorder.db_sync import sync_recordings_to_db

        recorder_engine.write_log("正在同步 CSV 录制数据到数据库...")

        try:
            result = sync_recordings_to_db(database=recorder_engine.database)

            bar_synced = result["bar"]["synced"]
            tick_synced = result["tick"]["synced"]
            bar_errors = result["bar"]["errors"]
            tick_errors = result["tick"]["errors"]
            elapsed = result["elapsed_ms"]

            recorder_engine.write_log(
                f"DB Sync 完成: bar {bar_synced} 条, tick {tick_synced} 条, "
                f"错误 {bar_errors + tick_errors}, 耗时 {elapsed}ms"
            )
        except Exception as e:
            recorder_engine.write_log(f"DB Sync 失败（不影响录制）: {e}")
            logger.exception("DB Sync 执行失败")

        # 调用原始 start
        original_start()

    recorder_engine.start = patched_start
    logger.info("✅ RecorderEngine.start() 已 patch，启动时会先执行 DB Sync")
    return True
