"""
测试 qp.common.utils 模块.
"""

from __future__ import annotations

import pytest
from vnpy.trader.constant import Exchange

from qp.common.utils import parse_vt_symbol


class TestParseVtSymbol:
    """测试 parse_vt_symbol 函数."""

    def test_parse_with_exchange_enum(self):
        """测试返回 Exchange 枚举."""
        symbol, exchange = parse_vt_symbol("p0.DCE")
        assert symbol == "p0"
        assert exchange == Exchange.DCE

    def test_parse_with_exchange_string(self):
        """测试返回交易所字符串."""
        symbol, exchange = parse_vt_symbol("p0.DCE", return_exchange_enum=False)
        assert symbol == "p0"
        assert exchange == "DCE"

    def test_parse_various_symbols(self, sample_vt_symbols):
        """测试多种合约代码."""
        for vt_symbol in sample_vt_symbols:
            symbol, exchange = parse_vt_symbol(vt_symbol)
            assert isinstance(symbol, str)
            assert isinstance(exchange, Exchange)
            assert len(symbol) > 0

    def test_parse_lowercase_exchange(self):
        """测试小写交易所代码."""
        symbol, exchange = parse_vt_symbol("p0.dce")
        assert exchange == Exchange.DCE

    def test_parse_mixed_case(self):
        """测试混合大小写."""
        symbol, exchange = parse_vt_symbol("P0.Dce")
        assert symbol == "P0"
        assert exchange == Exchange.DCE

    def test_missing_dot_raises(self):
        """测试缺少分隔符抛出异常."""
        with pytest.raises(ValueError, match="格式错误"):
            parse_vt_symbol("p0DCE")

    def test_empty_symbol_raises(self):
        """测试空 symbol 抛出异常."""
        with pytest.raises(ValueError, match="symbol 为空"):
            parse_vt_symbol(".DCE")

    def test_invalid_exchange_raises(self):
        """测试无效交易所抛出异常."""
        with pytest.raises(ValueError, match="未知的交易所"):
            parse_vt_symbol("p0.INVALID")

    def test_invalid_exchange_with_string_return(self):
        """测试返回字符串模式下不校验交易所."""
        # 返回字符串模式下不校验交易所有效性
        symbol, exchange = parse_vt_symbol("p0.INVALID", return_exchange_enum=False)
        assert symbol == "p0"
        assert exchange == "INVALID"

    def test_symbol_with_multiple_dots(self):
        """测试包含多个点的 symbol（如期权）."""
        # 只按最后一个点分割
        symbol, exchange = parse_vt_symbol("IO2501-C-4000.CFFEX")
        assert symbol == "IO2501-C-4000"
        assert exchange == Exchange.CFFEX
