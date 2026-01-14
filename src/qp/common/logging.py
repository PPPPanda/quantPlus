"""
日志配置模块.

提供统一的日志配置函数，避免在各模块中重复配置。
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

# 默认日志格式
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    verbose: bool = False,
    *,
    level: int | None = None,
    format_str: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    stream: TextIO | None = None,
) -> None:
    """
    配置全局日志.

    Args:
        verbose: 是否启用详细日志（DEBUG 级别）
        level: 显式指定日志级别（覆盖 verbose 参数）
        format_str: 日志格式字符串
        date_format: 日期格式字符串
        stream: 输出流（默认 sys.stderr）

    Examples:
        >>> setup_logging()  # INFO 级别
        >>> setup_logging(verbose=True)  # DEBUG 级别
        >>> setup_logging(level=logging.WARNING)  # WARNING 级别
    """
    if level is None:
        level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt=date_format,
        stream=stream or sys.stderr,
        force=True,  # Python 3.8+，允许重新配置
    )


def get_logger(name: str) -> logging.Logger:
    """
    获取命名 Logger.

    Args:
        name: Logger 名称，通常使用 __name__

    Returns:
        Logger 实例
    """
    return logging.getLogger(name)
