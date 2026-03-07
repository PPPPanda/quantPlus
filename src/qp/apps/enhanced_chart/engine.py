"""增强版 ChartWizardEngine.

在官方 vnpy_chartwizard.engine.ChartWizardEngine 基础上增加：
1. 当 gateway/datafeed 查询不到历史数据时，自动回退数据库。
2. 保持与官方 EVENT_CHART_HISTORY / APP_NAME 兼容。
"""

from __future__ import annotations

from datetime import datetime

from vnpy.event import Event
from vnpy.trader.constant import Interval
from vnpy.trader.object import BarData, ContractData, HistoryRequest
from vnpy.trader.utility import extract_vt_symbol
from vnpy_chartwizard.engine import (
    APP_NAME,
    EVENT_CHART_HISTORY,
    ChartWizardEngine,
)


class EnhancedChartWizardEngine(ChartWizardEngine):
    """增强版图表引擎：datafeed 空结果时自动回退数据库。"""

    def _query_history(
        self,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> None:
        """查询历史数据，优先 gateway/datafeed，失败时回退数据库。"""
        symbol, exchange = extract_vt_symbol(vt_symbol)

        req: HistoryRequest = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end,
        )

        contract: ContractData | None = self.main_engine.get_contract(vt_symbol)
        data: list[BarData] | None = None

        if contract:
            if contract.history_data:
                data = self.main_engine.query_history(req, contract.gateway_name)

            if not data:
                data = self.datafeed.query_bar_history(req)

        # 关键修复：gateway/datafeed 返回空时，回退数据库
        if not data:
            data = self.database.load_bar_data(
                symbol,
                exchange,
                interval,
                start,
                end,
            )

        event: Event = Event(EVENT_CHART_HISTORY, data)
        self.event_engine.put(event)
