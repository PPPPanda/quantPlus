"""
K 线合成器.

提供批量将 Tick 数据合成为 K 线的功能，支持：
1. Tick -> 1 分钟线
2. 1 分钟线 -> N 分钟线 / 小时线
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData

logger = logging.getLogger(__name__)


class BarSynthesizer:
    """
    K 线批量合成器.

    与 vnpy 的 BarGenerator 不同，本类专注于历史数据的批量转换，
    而非实时流式处理。
    """

    @staticmethod
    def ticks_to_bars(
        ticks: list[TickData],
        gateway_name: str = "DATAFEED",
    ) -> list[BarData]:
        """
        将 Tick 数据合成为 1 分钟 K 线.

        Args:
            ticks: Tick 数据列表，需按时间升序排列
            gateway_name: 网关名称

        Returns:
            1 分钟 BarData 列表，按时间升序排列
        """
        if not ticks:
            return []

        # 按分钟分组 tick
        minute_groups: dict[datetime, list[TickData]] = defaultdict(list)
        for tick in ticks:
            if not tick.last_price:  # 过滤无效 tick
                continue
            # 取整到分钟
            minute_dt = tick.datetime.replace(second=0, microsecond=0)
            minute_groups[minute_dt].append(tick)

        # 生成 K 线
        bars: list[BarData] = []
        for minute_dt in sorted(minute_groups.keys()):
            group_ticks = minute_groups[minute_dt]
            if not group_ticks:
                continue

            first_tick = group_ticks[0]
            bar = BarData(
                symbol=first_tick.symbol,
                exchange=first_tick.exchange,
                datetime=minute_dt,
                interval=Interval.MINUTE,
                gateway_name=gateway_name,
                open_price=group_ticks[0].last_price,
                high_price=max(t.last_price for t in group_ticks),
                low_price=min(t.last_price for t in group_ticks),
                close_price=group_ticks[-1].last_price,
                volume=sum(
                    max(0, group_ticks[i].volume - group_ticks[i - 1].volume)
                    for i in range(1, len(group_ticks))
                ) if len(group_ticks) > 1 else group_ticks[0].last_volume,
                turnover=sum(
                    max(0, group_ticks[i].turnover - group_ticks[i - 1].turnover)
                    for i in range(1, len(group_ticks))
                ) if len(group_ticks) > 1 else 0,
                open_interest=group_ticks[-1].open_interest,
            )
            bars.append(bar)

        logger.info("合成 1 分钟 K 线: %d 根 (从 %d 条 tick)", len(bars), len(ticks))
        return bars

    @staticmethod
    def resample_bars(
        bars: list[BarData],
        target_interval: Interval,
        window: int = 1,
    ) -> list[BarData]:
        """
        将 1 分钟 K 线重采样为目标周期.

        Args:
            bars: 1 分钟 K 线列表，需按时间升序排列
            target_interval: 目标周期 (MINUTE 或 HOUR)
            window: 周期窗口大小
                - MINUTE: 必须能整除 60，如 5, 15, 30
                - HOUR: 可以是任意正整数

        Returns:
            目标周期的 BarData 列表

        Raises:
            ValueError: 参数无效
        """
        if not bars:
            return []

        if target_interval == Interval.MINUTE:
            if 60 % window != 0:
                raise ValueError(f"分钟周期必须能整除 60: {window}")
            return BarSynthesizer._resample_to_minutes(bars, window)
        elif target_interval == Interval.HOUR:
            return BarSynthesizer._resample_to_hours(bars, window)
        else:
            raise ValueError(f"不支持的目标周期: {target_interval}")

    @staticmethod
    def _resample_to_minutes(
        bars: list[BarData],
        window: int,
    ) -> list[BarData]:
        """重采样为 N 分钟线."""
        result: list[BarData] = []
        window_bar: BarData | None = None

        for bar in bars:
            minute = bar.datetime.minute

            # 检查是否需要创建新的窗口 K 线
            if window_bar is None:
                window_dt = bar.datetime.replace(
                    minute=(minute // window) * window,
                    second=0,
                    microsecond=0,
                )
                window_bar = BarData(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    datetime=window_dt,
                    interval=Interval.MINUTE,
                    gateway_name=bar.gateway_name,
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    low_price=bar.low_price,
                    close_price=bar.close_price,
                    volume=bar.volume,
                    turnover=bar.turnover,
                    open_interest=bar.open_interest,
                )
            else:
                # 更新当前窗口
                window_bar.high_price = max(window_bar.high_price, bar.high_price)
                window_bar.low_price = min(window_bar.low_price, bar.low_price)
                window_bar.close_price = bar.close_price
                window_bar.volume += bar.volume
                window_bar.turnover += bar.turnover
                window_bar.open_interest = bar.open_interest

            # 检查窗口是否完成
            if (minute + 1) % window == 0:
                result.append(window_bar)
                window_bar = None

        # 处理最后一个未完成的窗口
        if window_bar is not None:
            result.append(window_bar)

        logger.info(
            "重采样为 %d 分钟 K 线: %d 根 (从 %d 根 1 分钟线)",
            window, len(result), len(bars)
        )
        return result

    @staticmethod
    def _resample_to_hours(
        bars: list[BarData],
        window: int,
    ) -> list[BarData]:
        """重采样为 N 小时线."""
        result: list[BarData] = []
        hour_bars: dict[datetime, BarData] = {}

        for bar in bars:
            # 取整到小时
            hour_dt = bar.datetime.replace(minute=0, second=0, microsecond=0)

            if hour_dt not in hour_bars:
                hour_bars[hour_dt] = BarData(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    datetime=hour_dt,
                    interval=Interval.HOUR,
                    gateway_name=bar.gateway_name,
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    low_price=bar.low_price,
                    close_price=bar.close_price,
                    volume=bar.volume,
                    turnover=bar.turnover,
                    open_interest=bar.open_interest,
                )
            else:
                hour_bar = hour_bars[hour_dt]
                hour_bar.high_price = max(hour_bar.high_price, bar.high_price)
                hour_bar.low_price = min(hour_bar.low_price, bar.low_price)
                hour_bar.close_price = bar.close_price
                hour_bar.volume += bar.volume
                hour_bar.turnover += bar.turnover
                hour_bar.open_interest = bar.open_interest

        # 将 1 小时线合成为 N 小时线
        sorted_hours = sorted(hour_bars.keys())
        if window == 1:
            result = [hour_bars[h] for h in sorted_hours]
        else:
            window_bar: BarData | None = None
            interval_count = 0

            for hour_dt in sorted_hours:
                hour_bar = hour_bars[hour_dt]

                if window_bar is None:
                    window_bar = BarData(
                        symbol=hour_bar.symbol,
                        exchange=hour_bar.exchange,
                        datetime=hour_dt,
                        interval=Interval.HOUR,
                        gateway_name=hour_bar.gateway_name,
                        open_price=hour_bar.open_price,
                        high_price=hour_bar.high_price,
                        low_price=hour_bar.low_price,
                        close_price=hour_bar.close_price,
                        volume=hour_bar.volume,
                        turnover=hour_bar.turnover,
                        open_interest=hour_bar.open_interest,
                    )
                    interval_count = 1
                else:
                    window_bar.high_price = max(window_bar.high_price, hour_bar.high_price)
                    window_bar.low_price = min(window_bar.low_price, hour_bar.low_price)
                    window_bar.close_price = hour_bar.close_price
                    window_bar.volume += hour_bar.volume
                    window_bar.turnover += hour_bar.turnover
                    window_bar.open_interest = hour_bar.open_interest
                    interval_count += 1

                if interval_count >= window:
                    result.append(window_bar)
                    window_bar = None
                    interval_count = 0

            # 处理最后一个未完成的窗口
            if window_bar is not None:
                result.append(window_bar)

        logger.info(
            "重采样为 %d 小时 K 线: %d 根 (从 %d 根 1 分钟线)",
            window, len(result), len(bars)
        )
        return result

    @staticmethod
    def ticks_to_target_bars(
        ticks: list[TickData],
        target_interval: Interval = Interval.MINUTE,
        window: int = 1,
        gateway_name: str = "DATAFEED",
    ) -> list[BarData]:
        """
        一步完成从 Tick 到目标周期 K 线的转换.

        Args:
            ticks: Tick 数据列表
            target_interval: 目标周期
            window: 周期窗口
            gateway_name: 网关名称

        Returns:
            目标周期的 BarData 列表
        """
        # 先合成 1 分钟线
        minute_bars = BarSynthesizer.ticks_to_bars(ticks, gateway_name)

        if target_interval == Interval.MINUTE and window == 1:
            return minute_bars

        # 再重采样到目标周期
        return BarSynthesizer.resample_bars(minute_bars, target_interval, window)
