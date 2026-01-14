"""
测试 qp.common.constants 模块.
"""

from __future__ import annotations

from vnpy.trader.constant import Exchange, Interval

from qp.common.constants import (
    EXCHANGE_MAP,
    INTERVAL_MAP,
    SYMBOL_MAP,
    CHINA_FUTURES_EXCHANGES,
    AM_BUFFER_SIZE,
)


class TestExchangeMap:
    """测试交易所映射表."""

    def test_major_exchanges_exist(self):
        """测试主要交易所都在映射表中."""
        assert "DCE" in EXCHANGE_MAP
        assert "SHFE" in EXCHANGE_MAP
        assert "CZCE" in EXCHANGE_MAP
        assert "CFFEX" in EXCHANGE_MAP
        assert "INE" in EXCHANGE_MAP

    def test_exchange_values_are_enums(self):
        """测试映射值都是 Exchange 枚举."""
        for key, value in EXCHANGE_MAP.items():
            assert isinstance(value, Exchange), f"{key} 的值不是 Exchange 枚举"

    def test_dce_mapping(self):
        """测试大商所映射."""
        assert EXCHANGE_MAP["DCE"] == Exchange.DCE


class TestIntervalMap:
    """测试周期映射表."""

    def test_standard_intervals_exist(self):
        """测试标准周期都在映射表中."""
        assert "DAILY" in INTERVAL_MAP
        assert "HOUR" in INTERVAL_MAP
        assert "MINUTE" in INTERVAL_MAP
        assert "WEEKLY" in INTERVAL_MAP

    def test_short_aliases_exist(self):
        """测试缩写别名都在映射表中."""
        assert "1d" in INTERVAL_MAP
        assert "1h" in INTERVAL_MAP
        assert "1m" in INTERVAL_MAP
        assert "d" in INTERVAL_MAP

    def test_interval_values_are_enums(self):
        """测试映射值都是 Interval 枚举."""
        for key, value in INTERVAL_MAP.items():
            assert isinstance(value, Interval), f"{key} 的值不是 Interval 枚举"

    def test_daily_aliases_consistent(self):
        """测试日线的多个别名指向同一枚举."""
        assert INTERVAL_MAP["DAILY"] == INTERVAL_MAP["1d"] == INTERVAL_MAP["d"]


class TestSymbolMap:
    """测试品种映射表."""

    def test_palm_oil_exists(self):
        """测试棕榈油映射存在."""
        assert "p0" in SYMBOL_MAP
        assert SYMBOL_MAP["p0"] == "P0"

    def test_short_alias(self):
        """测试品种缩写."""
        assert "p" in SYMBOL_MAP
        assert SYMBOL_MAP["p"] == "P0"

    def test_values_are_uppercase(self):
        """测试映射值都是大写."""
        for key, value in SYMBOL_MAP.items():
            assert value == value.upper(), f"{key} 的值 {value} 不是大写"


class TestChinaFuturesExchanges:
    """测试中国期货交易所集合."""

    def test_is_set(self):
        """测试是集合类型."""
        assert isinstance(CHINA_FUTURES_EXCHANGES, set)

    def test_major_exchanges_included(self):
        """测试主要交易所都在集合中."""
        assert "DCE" in CHINA_FUTURES_EXCHANGES
        assert "SHFE" in CHINA_FUTURES_EXCHANGES
        assert "CZCE" in CHINA_FUTURES_EXCHANGES
        assert "CFFEX" in CHINA_FUTURES_EXCHANGES
        assert "INE" in CHINA_FUTURES_EXCHANGES

    def test_stock_exchanges_not_included(self):
        """测试股票交易所不在集合中."""
        assert "SSE" not in CHINA_FUTURES_EXCHANGES
        assert "SZSE" not in CHINA_FUTURES_EXCHANGES


class TestAmBufferSize:
    """测试 ArrayManager 缓冲区大小."""

    def test_is_positive_integer(self):
        """测试是正整数."""
        assert isinstance(AM_BUFFER_SIZE, int)
        assert AM_BUFFER_SIZE > 0

    def test_reasonable_default(self):
        """测试默认值合理."""
        # 应该足够覆盖大多数指标计算
        assert AM_BUFFER_SIZE >= 20
