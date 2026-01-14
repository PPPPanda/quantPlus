"""
策略基类模块.

提供 QuantPlus CTA 策略的公共基类，统一 ArrayManager 初始化和参数校验。
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from vnpy.trader.object import BarData, TradeData
from vnpy.trader.utility import ArrayManager
from vnpy_ctastrategy import CtaTemplate

from qp.common.constants import AM_BUFFER_SIZE

logger = logging.getLogger(__name__)


class QuantPlusCtaStrategy(CtaTemplate):
    """
    QuantPlus CTA 策略基类.

    提供统一的：
    1. ArrayManager 大小计算（基于策略参数自动计算）
    2. 初始化日志记录
    3. 参数校验框架
    4. 成交回调标准处理

    子类需实现:
    - get_indicator_windows(): 返回策略使用的所有窗口参数列表
    - on_bar(): K线处理逻辑
    """

    author: str = "QuantPlus"

    # 子类可覆盖的配置
    am_buffer: int = AM_BUFFER_SIZE  # ArrayManager 额外缓冲区大小

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict[str, Any],
    ) -> None:
        """
        初始化策略.

        自动计算 ArrayManager 大小 = max(窗口参数) + am_buffer
        """
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 计算 ArrayManager 所需大小
        self._am_size = self._calculate_am_size()
        self.am = ArrayManager(size=self._am_size)

        logger.info(
            "策略初始化: %s, vt_symbol=%s, am_size=%d",
            strategy_name,
            vt_symbol,
            self._am_size,
        )

    def _calculate_am_size(self) -> int:
        """
        计算 ArrayManager 所需的最小 size.

        Returns:
            最大窗口参数 + 缓冲区大小
        """
        windows = self.get_indicator_windows()
        if not windows:
            return self.am_buffer + 30  # 默认最小值

        max_window = max(windows)
        return max_window + self.am_buffer

    @abstractmethod
    def get_indicator_windows(self) -> list[int]:
        """
        获取策略使用的所有窗口参数.

        子类必须实现此方法，返回所有用于计算指标的窗口大小。
        用于自动计算 ArrayManager 的 size。

        Returns:
            窗口参数列表

        Examples:
            >>> # 双均线策略
            >>> def get_indicator_windows(self):
            ...     return [self.fast_window, self.slow_window]

            >>> # 海龟策略
            >>> def get_indicator_windows(self):
            ...     return [self.entry_window, self.exit_window,
            ...             self.trend_ma_slow, self.atr_window]
        """
        raise NotImplementedError

    def on_init(self) -> None:
        """策略初始化回调."""
        self.write_log(f"策略初始化: {self.strategy_name}")

        # 调用子类参数校验
        self.validate_parameters()

        # 加载历史数据
        load_days = max(10, self._am_size // 20)
        self.load_bar(load_days)

        self.write_log(f"策略初始化完成，加载 {load_days} 天历史数据")

    def validate_parameters(self) -> None:
        """
        参数校验.

        子类可覆盖此方法添加自定义校验逻辑。
        校验失败应抛出 ValueError。
        """
        pass

    def on_start(self) -> None:
        """策略启动回调."""
        self.write_log("策略启动")
        logger.info("策略 %s 已启动", self.strategy_name)
        self.put_event()

    def on_stop(self) -> None:
        """策略停止回调."""
        self.write_log("策略停止")
        logger.info("策略 %s 已停止", self.strategy_name)
        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        """成交回调."""
        self.write_log(
            f"成交: {trade.direction.value} {trade.offset.value} "
            f"{trade.volume}手 @ {trade.price:.2f}"
        )
        logger.info(
            "策略 %s 成交: %s %s %.0f手 @ %.2f",
            self.strategy_name,
            trade.direction.value,
            trade.offset.value,
            trade.volume,
            trade.price,
        )
        self.sync_data()
        self.put_event()

    @abstractmethod
    def on_bar(self, bar: BarData) -> None:
        """
        K线数据更新回调.

        子类必须实现此方法，包含核心交易逻辑。
        """
        raise NotImplementedError
