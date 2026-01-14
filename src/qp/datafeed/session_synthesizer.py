"""
按交易时段合成 K 线.

将 tick 或分钟数据按照中国期货交易时段聚合，
匹配 p0 (主力连续) 数据的结构:
- 每日 6 根 K 线
- 时间戳为 K 线结束时间: 10:00, 11:15, 14:15, 15:00, 22:00, 23:00

DCE (大商所) 棕榈油交易时间:
- 日盘: 09:00-10:15, 10:30-11:30, 13:30-15:00
- 夜盘: 21:00-23:00
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData

logger = logging.getLogger(__name__)


@dataclass
class TradingSession:
    """交易时段定义."""

    name: str
    start: time
    end: time
    bar_end_time: time  # K 线结束时间戳


# DCE 棕榈油交易时段 (6 根 K 线/天)
DCE_PALM_OIL_SESSIONS: list[TradingSession] = [
    # 日盘
    TradingSession("早盘1", time(9, 0), time(10, 0), time(10, 0)),
    TradingSession("早盘2", time(10, 0), time(11, 30), time(11, 15)),  # 包含 10:15-10:30 休息
    TradingSession("午盘1", time(13, 30), time(14, 15), time(14, 15)),
    TradingSession("午盘2", time(14, 15), time(15, 0), time(15, 0)),
    # 夜盘
    TradingSession("夜盘1", time(21, 0), time(22, 0), time(22, 0)),
    TradingSession("夜盘2", time(22, 0), time(23, 0), time(23, 0)),
]


class SessionBarSynthesizer:
    """
    按交易时段合成 K 线.

    将分钟级数据聚合为交易时段 K 线，每日固定 6 根。
    """

    def __init__(
        self,
        sessions: list[TradingSession] | None = None,
    ) -> None:
        """
        初始化.

        Args:
            sessions: 交易时段定义列表，默认使用 DCE 棕榈油时段
        """
        self.sessions = sessions or DCE_PALM_OIL_SESSIONS

    def _get_session(self, dt: datetime) -> TradingSession | None:
        """根据时间获取所属交易时段."""
        t = dt.time()

        for session in self.sessions:
            # 处理跨午夜的情况 (如夜盘)
            if session.start <= session.end:
                if session.start <= t < session.end:
                    return session
            else:
                # 跨午夜时段
                if t >= session.start or t < session.end:
                    return session

        return None

    def _get_bar_datetime(self, dt: datetime, session: TradingSession) -> datetime:
        """获取 K 线的结束时间戳."""
        # 对于夜盘，K 线时间戳归属当天
        # 对于日盘，K 线时间戳也是当天
        return dt.replace(
            hour=session.bar_end_time.hour,
            minute=session.bar_end_time.minute,
            second=0,
            microsecond=0,
        )

    def synthesize_from_minutes(
        self,
        bars: list[BarData],
        gateway_name: str = "SESSION",
    ) -> list[BarData]:
        """
        从分钟 K 线合成时段 K 线.

        Args:
            bars: 分钟 K 线列表，需按时间升序排列
            gateway_name: 网关名称

        Returns:
            时段 K 线列表
        """
        if not bars:
            return []

        # 按时段分组
        session_bars: dict[datetime, dict] = {}

        for bar in bars:
            # 移除时区信息进行处理
            dt = bar.datetime
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)

            session = self._get_session(dt)
            if not session:
                continue

            bar_dt = self._get_bar_datetime(dt, session)

            if bar_dt not in session_bars:
                session_bars[bar_dt] = {
                    "symbol": bar.symbol,
                    "exchange": bar.exchange,
                    "open": bar.open_price,
                    "high": bar.high_price,
                    "low": bar.low_price,
                    "close": bar.close_price,
                    "volume": bar.volume,
                    "turnover": bar.turnover,
                    "open_interest": bar.open_interest,
                }
            else:
                data = session_bars[bar_dt]
                data["high"] = max(data["high"], bar.high_price)
                data["low"] = min(data["low"], bar.low_price)
                data["close"] = bar.close_price
                data["volume"] += bar.volume
                data["turnover"] += bar.turnover
                data["open_interest"] = bar.open_interest

        # 转换为 BarData 列表
        result: list[BarData] = []
        for bar_dt in sorted(session_bars.keys()):
            data = session_bars[bar_dt]
            result.append(BarData(
                symbol=data["symbol"],
                exchange=data["exchange"],
                datetime=bar_dt,
                interval=Interval.HOUR,
                open_price=data["open"],
                high_price=data["high"],
                low_price=data["low"],
                close_price=data["close"],
                volume=data["volume"],
                turnover=data["turnover"],
                open_interest=data["open_interest"],
                gateway_name=gateway_name,
            ))

        logger.info(
            "合成时段 K 线: %d 根 (从 %d 根分钟线)",
            len(result), len(bars)
        )
        return result

    def synthesize_from_ticks(
        self,
        ticks: list[TickData],
        gateway_name: str = "SESSION",
    ) -> list[BarData]:
        """
        从 tick 数据合成时段 K 线.

        Args:
            ticks: Tick 数据列表，需按时间升序排列
            gateway_name: 网关名称

        Returns:
            时段 K 线列表
        """
        if not ticks:
            return []

        # 按时段分组
        session_bars: dict[datetime, dict] = {}
        last_tick: TickData | None = None

        for tick in ticks:
            if not tick.last_price:
                continue

            # 移除时区信息
            dt = tick.datetime
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)

            session = self._get_session(dt)
            if not session:
                continue

            bar_dt = self._get_bar_datetime(dt, session)

            if bar_dt not in session_bars:
                session_bars[bar_dt] = {
                    "symbol": tick.symbol,
                    "exchange": tick.exchange,
                    "open": tick.last_price,
                    "high": tick.last_price,
                    "low": tick.last_price,
                    "close": tick.last_price,
                    "volume": 0.0,
                    "turnover": 0.0,
                    "open_interest": tick.open_interest,
                    "last_volume": tick.volume,
                    "last_turnover": tick.turnover,
                }
            else:
                data = session_bars[bar_dt]
                data["high"] = max(data["high"], tick.last_price)
                data["low"] = min(data["low"], tick.last_price)
                data["close"] = tick.last_price
                data["open_interest"] = tick.open_interest

                # 增量计算成交量
                if last_tick and last_tick.volume <= tick.volume:
                    data["volume"] += tick.volume - last_tick.volume
                    data["turnover"] += tick.turnover - last_tick.turnover

            last_tick = tick

        # 转换为 BarData 列表
        result: list[BarData] = []
        for bar_dt in sorted(session_bars.keys()):
            data = session_bars[bar_dt]
            result.append(BarData(
                symbol=data["symbol"],
                exchange=data["exchange"],
                datetime=bar_dt,
                interval=Interval.HOUR,
                open_price=data["open"],
                high_price=data["high"],
                low_price=data["low"],
                close_price=data["close"],
                volume=data["volume"],
                turnover=data["turnover"],
                open_interest=data["open_interest"],
                gateway_name=gateway_name,
            ))

        logger.info(
            "从 tick 合成时段 K 线: %d 根 (从 %d 条 tick)",
            len(result), len(ticks)
        )
        return result

    def synthesize_daily(
        self,
        session_bars: list[BarData],
        gateway_name: str = "SESSION",
    ) -> list[BarData]:
        """
        从时段 K 线合成日 K 线.

        Args:
            session_bars: 时段 K 线列表
            gateway_name: 网关名称

        Returns:
            日 K 线列表
        """
        if not session_bars:
            return []

        # 按日期分组
        daily_data: dict[str, dict] = {}

        for bar in session_bars:
            dt = bar.datetime
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)

            # 夜盘属于下一个交易日
            date_key = dt.strftime("%Y-%m-%d")
            if dt.hour >= 21:
                # 夜盘数据归属下一个交易日
                next_day = dt + timedelta(days=1)
                date_key = next_day.strftime("%Y-%m-%d")

            if date_key not in daily_data:
                daily_data[date_key] = {
                    "symbol": bar.symbol,
                    "exchange": bar.exchange,
                    "open": bar.open_price,
                    "high": bar.high_price,
                    "low": bar.low_price,
                    "close": bar.close_price,
                    "volume": bar.volume,
                    "turnover": bar.turnover,
                    "open_interest": bar.open_interest,
                }
            else:
                data = daily_data[date_key]
                data["high"] = max(data["high"], bar.high_price)
                data["low"] = min(data["low"], bar.low_price)
                data["close"] = bar.close_price
                data["volume"] += bar.volume
                data["turnover"] += bar.turnover
                data["open_interest"] = bar.open_interest

        # 转换为 BarData 列表
        result: list[BarData] = []
        for date_str in sorted(daily_data.keys()):
            data = daily_data[date_str]
            bar_dt = datetime.strptime(date_str, "%Y-%m-%d")

            result.append(BarData(
                symbol=data["symbol"],
                exchange=data["exchange"],
                datetime=bar_dt,
                interval=Interval.DAILY,
                open_price=data["open"],
                high_price=data["high"],
                low_price=data["low"],
                close_price=data["close"],
                volume=data["volume"],
                turnover=data["turnover"],
                open_interest=data["open_interest"],
                gateway_name=gateway_name,
            ))

        logger.info(
            "合成日 K 线: %d 根 (从 %d 根时段线)",
            len(result), len(session_bars)
        )
        return result
