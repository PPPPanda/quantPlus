"""
测试 qp.datafeed.bar_generator 模块.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData

from qp.datafeed.bar_generator import BarSynthesizer


def create_tick(
    dt: datetime,
    price: float,
    volume: float = 100,
    symbol: str = "p2505",
) -> TickData:
    """创建测试用 Tick 数据."""
    return TickData(
        symbol=symbol,
        exchange=Exchange.DCE,
        datetime=dt,
        last_price=price,
        volume=volume,
        turnover=price * volume,
        open_interest=1000,
        gateway_name="TEST",
    )


def create_bar(
    dt: datetime,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float = 100,
    symbol: str = "p2505",
) -> BarData:
    """创建测试用 Bar 数据."""
    return BarData(
        symbol=symbol,
        exchange=Exchange.DCE,
        datetime=dt,
        interval=Interval.MINUTE,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        volume=volume,
        turnover=close_price * volume,
        open_interest=1000,
        gateway_name="TEST",
    )


class TestTicksToBars:
    """测试 tick 转 bar 功能."""

    def test_empty_ticks(self):
        """空 tick 列表返回空 bar 列表."""
        bars = BarSynthesizer.ticks_to_bars([])
        assert bars == []

    def test_single_tick(self):
        """单个 tick 生成单根 bar."""
        tick = create_tick(datetime(2025, 1, 10, 9, 0, 30), 8000.0)
        bars = BarSynthesizer.ticks_to_bars([tick])

        assert len(bars) == 1
        assert bars[0].datetime == datetime(2025, 1, 10, 9, 0, 0)
        assert bars[0].open_price == 8000.0
        assert bars[0].close_price == 8000.0

    def test_multiple_ticks_same_minute(self):
        """同一分钟多个 tick 合成一根 bar."""
        ticks = [
            create_tick(datetime(2025, 1, 10, 9, 0, 10), 8000.0, 100),
            create_tick(datetime(2025, 1, 10, 9, 0, 20), 8010.0, 200),
            create_tick(datetime(2025, 1, 10, 9, 0, 40), 7990.0, 300),
            create_tick(datetime(2025, 1, 10, 9, 0, 55), 8005.0, 400),
        ]
        bars = BarSynthesizer.ticks_to_bars(ticks)

        assert len(bars) == 1
        bar = bars[0]
        assert bar.open_price == 8000.0
        assert bar.high_price == 8010.0
        assert bar.low_price == 7990.0
        assert bar.close_price == 8005.0

    def test_multiple_minutes(self):
        """跨分钟 tick 生成多根 bar."""
        ticks = [
            create_tick(datetime(2025, 1, 10, 9, 0, 30), 8000.0),
            create_tick(datetime(2025, 1, 10, 9, 1, 30), 8010.0),
            create_tick(datetime(2025, 1, 10, 9, 2, 30), 8020.0),
        ]
        bars = BarSynthesizer.ticks_to_bars(ticks)

        assert len(bars) == 3
        assert bars[0].datetime == datetime(2025, 1, 10, 9, 0, 0)
        assert bars[1].datetime == datetime(2025, 1, 10, 9, 1, 0)
        assert bars[2].datetime == datetime(2025, 1, 10, 9, 2, 0)

    def test_filter_zero_price(self):
        """过滤价格为 0 的 tick."""
        ticks = [
            create_tick(datetime(2025, 1, 10, 9, 0, 30), 8000.0),
            create_tick(datetime(2025, 1, 10, 9, 0, 40), 0),  # 应被过滤
            create_tick(datetime(2025, 1, 10, 9, 0, 50), 8010.0),
        ]
        bars = BarSynthesizer.ticks_to_bars(ticks)

        assert len(bars) == 1
        assert bars[0].high_price == 8010.0


class TestResampleBars:
    """测试 bar 重采样功能."""

    def test_empty_bars(self):
        """空 bar 列表返回空."""
        result = BarSynthesizer.resample_bars([], Interval.MINUTE, 5)
        assert result == []

    def test_resample_to_5min(self):
        """重采样为 5 分钟线."""
        bars = [
            create_bar(datetime(2025, 1, 10, 9, 0), 100, 105, 99, 102),
            create_bar(datetime(2025, 1, 10, 9, 1), 102, 106, 101, 104),
            create_bar(datetime(2025, 1, 10, 9, 2), 104, 107, 103, 105),
            create_bar(datetime(2025, 1, 10, 9, 3), 105, 108, 104, 106),
            create_bar(datetime(2025, 1, 10, 9, 4), 106, 109, 105, 108),
        ]
        result = BarSynthesizer.resample_bars(bars, Interval.MINUTE, 5)

        assert len(result) == 1
        bar = result[0]
        assert bar.datetime == datetime(2025, 1, 10, 9, 0)
        assert bar.open_price == 100
        assert bar.high_price == 109
        assert bar.low_price == 99
        assert bar.close_price == 108

    def test_resample_to_15min(self):
        """重采样为 15 分钟线."""
        bars = []
        for i in range(30):
            bars.append(create_bar(
                datetime(2025, 1, 10, 9, i),
                100 + i, 105 + i, 99 + i, 102 + i
            ))

        result = BarSynthesizer.resample_bars(bars, Interval.MINUTE, 15)

        assert len(result) == 2
        assert result[0].datetime == datetime(2025, 1, 10, 9, 0)
        assert result[1].datetime == datetime(2025, 1, 10, 9, 15)

    def test_resample_to_hour(self):
        """重采样为小时线."""
        bars = []
        for i in range(120):  # 2 小时
            hour = 9 + i // 60
            minute = i % 60
            bars.append(create_bar(
                datetime(2025, 1, 10, hour, minute),
                100 + i, 105 + i, 99 + i, 102 + i
            ))

        result = BarSynthesizer.resample_bars(bars, Interval.HOUR, 1)

        assert len(result) == 2
        assert result[0].datetime == datetime(2025, 1, 10, 9, 0)
        assert result[1].datetime == datetime(2025, 1, 10, 10, 0)

    def test_invalid_minute_window(self):
        """无效的分钟窗口抛出异常."""
        bars = [create_bar(datetime(2025, 1, 10, 9, 0), 100, 105, 99, 102)]

        with pytest.raises(ValueError, match="必须能整除 60"):
            BarSynthesizer.resample_bars(bars, Interval.MINUTE, 7)

    def test_invalid_interval(self):
        """不支持的周期抛出异常."""
        bars = [create_bar(datetime(2025, 1, 10, 9, 0), 100, 105, 99, 102)]

        with pytest.raises(ValueError, match="不支持的目标周期"):
            BarSynthesizer.resample_bars(bars, Interval.DAILY, 1)


class TestTicksToTargetBars:
    """测试一步转换功能."""

    def test_ticks_to_1min(self):
        """tick 转 1 分钟线."""
        ticks = [
            create_tick(datetime(2025, 1, 10, 9, 0, 30), 8000.0),
            create_tick(datetime(2025, 1, 10, 9, 1, 30), 8010.0),
        ]
        bars = BarSynthesizer.ticks_to_target_bars(
            ticks, Interval.MINUTE, 1
        )

        assert len(bars) == 2

    def test_ticks_to_5min(self):
        """tick 转 5 分钟线."""
        ticks = []
        for i in range(10):
            ticks.append(create_tick(
                datetime(2025, 1, 10, 9, i, 30),
                8000.0 + i * 10
            ))

        bars = BarSynthesizer.ticks_to_target_bars(
            ticks, Interval.MINUTE, 5
        )

        assert len(bars) == 2

    def test_ticks_to_hour(self):
        """tick 转小时线."""
        ticks = []
        for i in range(120):  # 2 小时
            hour = 9 + i // 60
            minute = i % 60
            ticks.append(create_tick(
                datetime(2025, 1, 10, hour, minute, 30),
                8000.0 + i
            ))

        bars = BarSynthesizer.ticks_to_target_bars(
            ticks, Interval.HOUR, 1
        )

        assert len(bars) == 2
