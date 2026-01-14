"""
迅投研 (XTQuant) 数据源适配器.

使用 xtquant 库获取期货历史行情数据，包括 Tick 和 K 线数据。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, HistoryRequest, TickData

from qp.datafeed.base import BaseDatafeed

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# vnpy Exchange -> XTQuant market 映射
EXCHANGE_TO_XT_MARKET: dict[Exchange, str] = {
    Exchange.DCE: "DF",    # 大连商品交易所
    Exchange.SHFE: "SF",   # 上海期货交易所
    Exchange.CZCE: "ZF",   # 郑州商品交易所
    Exchange.CFFEX: "IF",  # 中国金融期货交易所
    Exchange.INE: "INE",   # 上海国际能源交易中心
}

# XTQuant market -> vnpy Exchange 映射
XT_MARKET_TO_EXCHANGE: dict[str, Exchange] = {
    v: k for k, v in EXCHANGE_TO_XT_MARKET.items()
}

# vnpy Interval -> XTQuant period 映射
INTERVAL_TO_XT_PERIOD: dict[Interval, str] = {
    Interval.MINUTE: "1m",
    Interval.HOUR: "1h",
    Interval.DAILY: "1d",
}


class XTQuantDatafeed(BaseDatafeed):
    """
    迅投研数据源.

    使用 xtquant 库的 xtdata 模块获取期货历史数据。
    需要先安装 xtquant: pip install xtquant
    """

    def __init__(self) -> None:
        """初始化."""
        self._inited: bool = False
        self._xtdata = None  # 延迟导入

    def init(self, output: callable | None = None) -> bool:
        """
        初始化数据源.

        检查 xtquant 是否安装，并尝试初始化连接。
        """
        if self._inited:
            return True

        try:
            from xtquant import xtdata

            self._xtdata = xtdata
            self._inited = True

            msg = "XTQuant 数据源初始化成功"
            logger.info(msg)
            if output:
                output(msg)

            return True

        except ImportError as e:
            msg = f"XTQuant 数据源初始化失败: {e}. 请安装 xtquant: pip install xtquant"
            logger.error(msg)
            if output:
                output(msg)
            return False

    def _to_xt_symbol(self, symbol: str, exchange: Exchange) -> str:
        """
        将 vnpy 合约代码转换为 XTQuant 格式.

        Args:
            symbol: vnpy 合约代码，如 "p2501"
            exchange: 交易所

        Returns:
            XTQuant 格式合约代码，如 "p2501.DF"
        """
        market = EXCHANGE_TO_XT_MARKET.get(exchange)
        if not market:
            raise ValueError(f"不支持的交易所: {exchange}")
        return f"{symbol}.{market}"

    def _download_data(
        self,
        xt_symbol: str,
        period: str,
        start_time: str,
        end_time: str,
        output: callable | None = None,
    ) -> None:
        """
        下载数据到本地缓存.

        Args:
            xt_symbol: XTQuant 格式合约代码
            period: 周期，如 "tick", "1m", "1h", "1d"
            start_time: 开始时间，格式 "YYYYMMDD" 或 "YYYYMMDDHHMMSS"
            end_time: 结束时间
            output: 日志回调
        """
        msg = f"下载数据: {xt_symbol}, 周期: {period}, {start_time} - {end_time}"
        logger.info(msg)
        if output:
            output(msg)

        self._xtdata.download_history_data(
            stock_code=xt_symbol,
            period=period,
            start_time=start_time,
            end_time=end_time,
        )

    def query_bar_history(
        self,
        req: HistoryRequest,
        output: callable | None = None,
    ) -> list[BarData]:
        """
        查询 K 线历史数据.

        Args:
            req: 历史请求，包含 symbol, exchange, start, end, interval
            output: 日志回调

        Returns:
            BarData 列表
        """
        if not self._inited:
            if not self.init(output):
                return []

        # 转换合约代码
        xt_symbol = self._to_xt_symbol(req.symbol, req.exchange)
        period = INTERVAL_TO_XT_PERIOD.get(req.interval, "1d")

        # 格式化时间
        start_time = req.start.strftime("%Y%m%d")
        end_time = req.end.strftime("%Y%m%d") if req.end else ""

        # 下载数据
        self._download_data(xt_symbol, period, start_time, end_time, output)

        # 获取数据
        data = self._xtdata.get_market_data(
            field_list=[],
            stock_list=[xt_symbol],
            period=period,
            start_time=start_time,
            end_time=end_time,
            dividend_type="none",
            fill_data=True,
        )

        # 转换为 BarData
        bars: list[BarData] = []
        if xt_symbol in data:
            df_data = data[xt_symbol]
            if hasattr(df_data, "iterrows"):
                # DataFrame 格式
                for _, row in df_data.iterrows():
                    bar = self._row_to_bar(row, req.symbol, req.exchange, req.interval)
                    if bar:
                        bars.append(bar)
            elif isinstance(df_data, dict):
                # 字典格式 (numpy array)
                bars = self._dict_to_bars(df_data, req.symbol, req.exchange, req.interval)

        msg = f"获取 K 线数据: {len(bars)} 条"
        logger.info(msg)
        if output:
            output(msg)

        return bars

    def query_tick_history(
        self,
        req: HistoryRequest,
        output: callable | None = None,
    ) -> list[TickData]:
        """
        查询 Tick 历史数据.

        Args:
            req: 历史请求
            output: 日志回调

        Returns:
            TickData 列表
        """
        if not self._inited:
            if not self.init(output):
                return []

        # 转换合约代码
        xt_symbol = self._to_xt_symbol(req.symbol, req.exchange)

        # 格式化时间
        start_time = req.start.strftime("%Y%m%d")
        end_time = req.end.strftime("%Y%m%d") if req.end else ""

        # 下载 tick 数据
        self._download_data(xt_symbol, "tick", start_time, end_time, output)

        # 获取数据
        data = self._xtdata.get_market_data(
            field_list=[],
            stock_list=[xt_symbol],
            period="tick",
            start_time=start_time,
            end_time=end_time,
            dividend_type="none",
            fill_data=False,
        )

        # 转换为 TickData
        ticks: list[TickData] = []
        if xt_symbol in data:
            tick_data = data[xt_symbol]
            ticks = self._convert_tick_data(tick_data, req.symbol, req.exchange)

        msg = f"获取 Tick 数据: {len(ticks)} 条"
        logger.info(msg)
        if output:
            output(msg)

        return ticks

    def _row_to_bar(
        self,
        row,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
    ) -> BarData | None:
        """将 DataFrame 行转换为 BarData."""
        try:
            # 解析时间
            if "time" in row:
                dt = self._parse_timestamp(row["time"])
            else:
                return None

            return BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                interval=interval,
                open_price=float(row.get("open", 0)),
                high_price=float(row.get("high", 0)),
                low_price=float(row.get("low", 0)),
                close_price=float(row.get("close", 0)),
                volume=float(row.get("volume", 0)),
                turnover=float(row.get("amount", 0)),
                open_interest=float(row.get("openInterest", 0)),
                gateway_name="XTQUANT",
            )
        except Exception as e:
            logger.warning("转换 Bar 数据失败: %s", e)
            return None

    def _dict_to_bars(
        self,
        data: dict,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
    ) -> list[BarData]:
        """将字典格式数据转换为 BarData 列表."""
        bars: list[BarData] = []

        times = data.get("time", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        volumes = data.get("volume", [])
        amounts = data.get("amount", [])
        ois = data.get("openInterest", [])

        for i in range(len(times)):
            try:
                dt = self._parse_timestamp(times[i])
                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=dt,
                    interval=interval,
                    open_price=float(opens[i]) if i < len(opens) else 0,
                    high_price=float(highs[i]) if i < len(highs) else 0,
                    low_price=float(lows[i]) if i < len(lows) else 0,
                    close_price=float(closes[i]) if i < len(closes) else 0,
                    volume=float(volumes[i]) if i < len(volumes) else 0,
                    turnover=float(amounts[i]) if i < len(amounts) else 0,
                    open_interest=float(ois[i]) if i < len(ois) else 0,
                    gateway_name="XTQUANT",
                )
                bars.append(bar)
            except Exception as e:
                logger.warning("转换第 %d 条 Bar 数据失败: %s", i, e)

        return bars

    def _convert_tick_data(
        self,
        data: dict,
        symbol: str,
        exchange: Exchange,
    ) -> list[TickData]:
        """将 XTQuant tick 数据转换为 TickData 列表."""
        ticks: list[TickData] = []

        times = data.get("time", [])
        last_prices = data.get("lastPrice", [])
        volumes = data.get("volume", [])
        amounts = data.get("amount", [])
        ois = data.get("openInt", data.get("openInterest", []))

        # 盘口数据
        bid_prices = [data.get(f"bidPrice{i}", []) for i in range(1, 6)]
        ask_prices = [data.get(f"askPrice{i}", []) for i in range(1, 6)]
        bid_volumes = [data.get(f"bidVol{i}", []) for i in range(1, 6)]
        ask_volumes = [data.get(f"askVol{i}", []) for i in range(1, 6)]

        # 日内统计
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        pre_closes = data.get("lastClose", [])
        upper_limits = data.get("upperLimit", [])
        lower_limits = data.get("lowerLimit", [])

        for i in range(len(times)):
            try:
                dt = self._parse_timestamp(times[i])

                tick = TickData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=dt,
                    last_price=float(last_prices[i]) if i < len(last_prices) else 0,
                    volume=float(volumes[i]) if i < len(volumes) else 0,
                    turnover=float(amounts[i]) if i < len(amounts) else 0,
                    open_interest=float(ois[i]) if i < len(ois) else 0,
                    open_price=float(opens[i]) if i < len(opens) else 0,
                    high_price=float(highs[i]) if i < len(highs) else 0,
                    low_price=float(lows[i]) if i < len(lows) else 0,
                    pre_close=float(pre_closes[i]) if i < len(pre_closes) else 0,
                    limit_up=float(upper_limits[i]) if i < len(upper_limits) else 0,
                    limit_down=float(lower_limits[i]) if i < len(lower_limits) else 0,
                    bid_price_1=float(bid_prices[0][i]) if i < len(bid_prices[0]) else 0,
                    bid_price_2=float(bid_prices[1][i]) if i < len(bid_prices[1]) else 0,
                    bid_price_3=float(bid_prices[2][i]) if i < len(bid_prices[2]) else 0,
                    bid_price_4=float(bid_prices[3][i]) if i < len(bid_prices[3]) else 0,
                    bid_price_5=float(bid_prices[4][i]) if i < len(bid_prices[4]) else 0,
                    ask_price_1=float(ask_prices[0][i]) if i < len(ask_prices[0]) else 0,
                    ask_price_2=float(ask_prices[1][i]) if i < len(ask_prices[1]) else 0,
                    ask_price_3=float(ask_prices[2][i]) if i < len(ask_prices[2]) else 0,
                    ask_price_4=float(ask_prices[3][i]) if i < len(ask_prices[3]) else 0,
                    ask_price_5=float(ask_prices[4][i]) if i < len(ask_prices[4]) else 0,
                    bid_volume_1=float(bid_volumes[0][i]) if i < len(bid_volumes[0]) else 0,
                    bid_volume_2=float(bid_volumes[1][i]) if i < len(bid_volumes[1]) else 0,
                    bid_volume_3=float(bid_volumes[2][i]) if i < len(bid_volumes[2]) else 0,
                    bid_volume_4=float(bid_volumes[3][i]) if i < len(bid_volumes[3]) else 0,
                    bid_volume_5=float(bid_volumes[4][i]) if i < len(bid_volumes[4]) else 0,
                    ask_volume_1=float(ask_volumes[0][i]) if i < len(ask_volumes[0]) else 0,
                    ask_volume_2=float(ask_volumes[1][i]) if i < len(ask_volumes[1]) else 0,
                    ask_volume_3=float(ask_volumes[2][i]) if i < len(ask_volumes[2]) else 0,
                    ask_volume_4=float(ask_volumes[3][i]) if i < len(ask_volumes[3]) else 0,
                    ask_volume_5=float(ask_volumes[4][i]) if i < len(ask_volumes[4]) else 0,
                    gateway_name="XTQUANT",
                )
                ticks.append(tick)
            except Exception as e:
                logger.warning("转换第 %d 条 Tick 数据失败: %s", i, e)

        return ticks

    def _parse_timestamp(self, timestamp) -> datetime:
        """
        解析 XTQuant 时间戳.

        XTQuant 返回的时间格式可能是:
        - 毫秒时间戳 (整数)
        - 字符串 "YYYYMMDDHHMMSS" 或 "YYYYMMDDHHMMSSfff"
        """
        if isinstance(timestamp, (int, float)):
            # 毫秒时间戳
            if timestamp > 1e12:
                return datetime.fromtimestamp(timestamp / 1000)
            else:
                return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            if len(timestamp) >= 14:
                return datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S")
            elif len(timestamp) == 8:
                return datetime.strptime(timestamp, "%Y%m%d")
        raise ValueError(f"无法解析时间戳: {timestamp}")

    def close(self) -> None:
        """关闭数据源."""
        self._inited = False
        self._xtdata = None


# 便捷工厂函数
def create_xtquant_datafeed() -> XTQuantDatafeed:
    """创建 XTQuant 数据源实例."""
    feed = XTQuantDatafeed()
    return feed
