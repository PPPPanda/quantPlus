"""
数据源基类.

参考 vnpy datafeed 接口设计，提供统一的数据获取抽象。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vnpy.trader.object import BarData, TickData, HistoryRequest


class BaseDatafeed(ABC):
    """
    数据源抽象基类.

    所有数据源实现需要继承此类并实现以下方法：
    - query_bar_history: 查询 K 线历史数据
    - query_tick_history: 查询 Tick 历史数据
    """

    @abstractmethod
    def init(self, output: callable | None = None) -> bool:
        """
        初始化数据源连接.

        Args:
            output: 日志输出回调函数

        Returns:
            初始化是否成功
        """
        pass

    @abstractmethod
    def query_bar_history(
        self,
        req: "HistoryRequest",
        output: callable | None = None,
    ) -> list["BarData"]:
        """
        查询 K 线历史数据.

        Args:
            req: 历史数据请求，包含合约、时间范围、周期等
            output: 日志输出回调函数

        Returns:
            BarData 列表，按时间升序排列
        """
        pass

    @abstractmethod
    def query_tick_history(
        self,
        req: "HistoryRequest",
        output: callable | None = None,
    ) -> list["TickData"]:
        """
        查询 Tick 历史数据.

        Args:
            req: 历史数据请求，包含合约、时间范围等
            output: 日志输出回调函数

        Returns:
            TickData 列表，按时间升序排列
        """
        pass

    def close(self) -> None:
        """
        关闭数据源连接.
        """
        pass
