"""
测试 qp.backtest.engine 模块.
"""

from __future__ import annotations

import pytest

from qp.backtest.engine import load_strategy_class, BacktestResult


class TestLoadStrategyClass:
    """测试 load_strategy_class 函数."""

    def test_load_palm_oil_strategy(self):
        """测试加载棕榈油双均线策略."""
        strategy_cls = load_strategy_class("CtaPalmOilStrategy")
        assert strategy_cls.__name__ == "CtaPalmOilStrategy"

    def test_load_turtle_strategy(self):
        """测试加载增强海龟策略."""
        strategy_cls = load_strategy_class("CtaTurtleEnhancedStrategy")
        assert strategy_cls.__name__ == "CtaTurtleEnhancedStrategy"

    def test_unknown_strategy_raises(self):
        """测试加载未知策略抛出异常."""
        with pytest.raises((ImportError, AttributeError)):
            load_strategy_class("NonExistentStrategy")


class TestBacktestResult:
    """测试 BacktestResult 数据类."""

    def test_create_result(self):
        """测试创建回测结果对象."""
        result = BacktestResult(
            stats={"sharpe_ratio": 1.5},
            trades=[],
            daily_results=[],
            history_data_count=100,
        )
        assert result.stats["sharpe_ratio"] == 1.5
        assert result.history_data_count == 100

    def test_result_fields(self):
        """测试结果字段类型."""
        result = BacktestResult(
            stats={},
            trades=[],
            daily_results=[],
            history_data_count=0,
        )
        assert isinstance(result.stats, dict)
        assert isinstance(result.trades, list)
        assert isinstance(result.daily_results, list)
        assert isinstance(result.history_data_count, int)
