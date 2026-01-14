"""QuantPlus - A股期货量化交易框架."""

__version__ = "0.1.0"

# 导出常用模块供便捷访问
from qp.common import (
    EXCHANGE_MAP,
    INTERVAL_MAP,
    SYMBOL_MAP,
    parse_vt_symbol,
    setup_logging,
)

__all__ = [
    "__version__",
    "EXCHANGE_MAP",
    "INTERVAL_MAP",
    "SYMBOL_MAP",
    "parse_vt_symbol",
    "setup_logging",
]
