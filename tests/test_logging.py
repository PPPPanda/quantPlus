"""
测试 qp.common.logging 模块.
"""

from __future__ import annotations

import logging
from io import StringIO

from qp.common.logging import setup_logging, get_logger


class TestSetupLogging:
    """测试 setup_logging 函数."""

    def test_default_level_is_info(self):
        """测试默认日志级别是 INFO."""
        setup_logging()
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_verbose_sets_debug(self):
        """测试 verbose=True 设置 DEBUG 级别."""
        setup_logging(verbose=True)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_explicit_level_override(self):
        """测试显式指定级别覆盖 verbose."""
        setup_logging(verbose=True, level=logging.WARNING)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_custom_stream(self):
        """测试自定义输出流."""
        stream = StringIO()
        setup_logging(stream=stream)

        test_logger = get_logger("test_custom_stream")
        test_logger.info("测试消息")

        output = stream.getvalue()
        assert "测试消息" in output


class TestGetLogger:
    """测试 get_logger 函数."""

    def test_returns_logger(self):
        """测试返回 Logger 实例."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        """测试 Logger 名称正确."""
        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_same_name_same_logger(self):
        """测试相同名称返回相同 Logger."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        assert logger1 is logger2
