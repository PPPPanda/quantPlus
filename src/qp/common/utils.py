"""
公共工具函数.

提供 vt_symbol 解析等常用功能。
"""

from __future__ import annotations

from typing import overload

from vnpy.trader.constant import Exchange

from qp.common.constants import EXCHANGE_MAP


@overload
def parse_vt_symbol(vt_symbol: str, *, return_exchange_enum: bool = True) -> tuple[str, Exchange]: ...


@overload
def parse_vt_symbol(vt_symbol: str, *, return_exchange_enum: bool = False) -> tuple[str, str]: ...


def parse_vt_symbol(
    vt_symbol: str,
    *,
    return_exchange_enum: bool = True,
) -> tuple[str, Exchange] | tuple[str, str]:
    """
    解析 vt_symbol 为 symbol 和 exchange.

    Args:
        vt_symbol: 合约代码，如 "p0.DCE" 或 "p2501.DCE"
        return_exchange_enum: 是否返回 Exchange 枚举（默认 True）
            - True: 返回 (symbol, Exchange)
            - False: 返回 (symbol, exchange_str)

    Returns:
        (symbol, exchange) 元组

    Raises:
        ValueError: vt_symbol 格式错误或交易所不支持

    Examples:
        >>> parse_vt_symbol("p0.DCE")
        ('p0', <Exchange.DCE: 'DCE'>)

        >>> parse_vt_symbol("p0.DCE", return_exchange_enum=False)
        ('p0', 'DCE')
    """
    if "." not in vt_symbol:
        raise ValueError(
            f"vt_symbol 格式错误: {vt_symbol}，应为 symbol.exchange 格式"
        )

    symbol, exchange_str = vt_symbol.rsplit(".", 1)
    exchange_upper = exchange_str.upper()

    if not symbol:
        raise ValueError(f"vt_symbol 中 symbol 为空: {vt_symbol}")

    if return_exchange_enum:
        if exchange_upper not in EXCHANGE_MAP:
            raise ValueError(
                f"未知的交易所: {exchange_str}，"
                f"支持的交易所: {list(EXCHANGE_MAP.keys())}"
            )
        return symbol, EXCHANGE_MAP[exchange_upper]

    return symbol, exchange_str
