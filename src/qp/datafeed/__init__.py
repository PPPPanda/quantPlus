"""
QuantPlus 数据获取模块.

支持多种数据源获取历史行情数据，包括 Tick 和 Bar 数据。

数据聚合规范:
    - 使用 SessionBarSynthesizer 按交易时段合成 K 线
    - 每日固定 6 根 K 线 (DCE 棕榈油)
    - 时间戳为 K 线结束时间: 10:00, 11:15, 14:15, 15:00, 22:00, 23:00
"""

from qp.datafeed.base import BaseDatafeed
from qp.datafeed.bar_generator import BarSynthesizer
from qp.datafeed.session_synthesizer import SessionBarSynthesizer, DCE_PALM_OIL_SESSIONS
from qp.datafeed.xtquant_feed import XTQuantDatafeed, create_xtquant_datafeed

__all__ = [
    "BaseDatafeed",
    "BarSynthesizer",
    "SessionBarSynthesizer",
    "DCE_PALM_OIL_SESSIONS",
    "XTQuantDatafeed",
    "create_xtquant_datafeed",
]
