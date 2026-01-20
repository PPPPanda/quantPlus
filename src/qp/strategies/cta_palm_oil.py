"""
棕榈油双均线 CTA 策略.

基于快慢均线交叉的趋势跟踪策略，适用于大商所棕榈油期货。

K线周期说明：
- 回测时：直接使用数据库中对应周期的 K 线数据
- 实盘时：通过 BarGenerator 将 Tick 合成为指定周期的 K 线
- bar_window: K线窗口大小（1=1分钟, 15=15分钟, 60=60分钟）
- bar_interval: K线周期类型 ("MINUTE" 或 "HOUR")

================================================================================
如何在 GUI 的 CTA 策略页添加该策略
================================================================================

1. 启动 Trader GUI:
   uv run python -m qp.runtime.trader_app --profile trade

2. 在菜单栏点击「功能」->「CTA策略」打开 CTA 策略管理界面

3. 点击「添加策略」按钮，在弹出的对话框中填写：
   - class_name (策略类名): CtaPalmOilStrategy
   - strategy_name (策略实例名): 自定义，如 palm_oil_ma_01
   - vt_symbol (合约代码): p2501.DCE
     * p = 棕榈油品种代码
     * 2501 = 2025年01月合约
     * DCE = 大连商品交易所

4. 参数设置 (setting):
   - fast_window: 快速均线周期，默认 10
   - slow_window: 慢速均线周期，默认 20
   - fixed_size: 每次交易手数，默认 1

5. 点击「添加」完成策略添加

6. 在策略列表中找到刚添加的策略，点击「初始化」->「启动」

================================================================================
vt_symbol 规范
================================================================================

格式: {品种代码}{合约月份}.{交易所}

棕榈油示例:
- p2501.DCE  -> 棕榈油 2025年01月合约 (大商所)
- p2505.DCE  -> 棕榈油 2025年05月合约 (大商所)

交易所代码:
- DCE  = 大连商品交易所 (大商所)
- SHFE = 上海期货交易所 (上期所)
- CZCE = 郑州商品交易所 (郑商所)
- CFFEX = 中国金融期货交易所 (中金所)

================================================================================
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from vnpy.trader.object import BarData, OrderData, TickData, TradeData
from vnpy.trader.utility import ArrayManager, BarGenerator
from vnpy.trader.constant import Interval
from vnpy_ctastrategy import CtaTemplate

logger = logging.getLogger(__name__)


class CtaPalmOilStrategy(CtaTemplate):
    """
    棕榈油双均线策略.

    策略逻辑：
    - 当快速均线上穿慢速均线时，做多
    - 当快速均线下穿慢速均线时，做空
    - 持仓时发生反向信号则先平仓再反向开仓
    """

    author: str = "QuantPlus"

    # 策略参数（可在 GUI 中配置）
    fast_window: int = 10
    slow_window: int = 20
    fixed_size: int = 1

    # K线周期参数（实盘使用）
    bar_window: int = 1             # K线窗口大小（1=1分钟, 15=15分钟）
    bar_interval: str = "MINUTE"    # K线周期类型 ("MINUTE" 或 "HOUR")

    # 参数列表（GUI 显示用）
    parameters: list[str] = [
        "fast_window",
        "slow_window",
        "fixed_size",
        "bar_window",
        "bar_interval",
    ]

    # 策略变量（运行时状态）
    fast_ma: float = 0.0
    slow_ma: float = 0.0
    ma_trend: int = 0  # 1=多头趋势, -1=空头趋势, 0=无趋势

    # 变量列表（GUI 显示用）
    variables: list[str] = [
        "fast_ma",
        "slow_ma",
        "ma_trend",
    ]

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict[str, Any],
    ) -> None:
        """初始化策略."""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 初始化 K 线管理器，用于计算技术指标
        # 修复：使用 slow_window + 10，避免 size 过大导致 am.inited 无法初始化
        self.am: ArrayManager = ArrayManager(size=self.slow_window + 10)

        # K线生成器（实盘时将 Tick 转换为 Bar）
        self.bg: Optional[BarGenerator] = None

        logger.info(
            "策略初始化: %s, 合约: %s, 快线: %d, 慢线: %d, 手数: %d",
            strategy_name,
            vt_symbol,
            self.fast_window,
            self.slow_window,
            self.fixed_size,
        )

    def on_init(self) -> None:
        """策略初始化回调."""
        self.write_log("策略初始化开始")

        # 参数校验
        if self.fast_window >= self.slow_window:
            error_msg = (
                f"参数错误: fast_window({self.fast_window}) "
                f"必须小于 slow_window({self.slow_window})"
            )
            self.write_log(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.fixed_size <= 0:
            error_msg = f"参数错误: fixed_size({self.fixed_size}) 必须大于 0"
            self.write_log(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 创建 K 线生成器（实盘时 Tick -> Bar）
        interval = Interval.HOUR if self.bar_interval == "HOUR" else Interval.MINUTE
        if self.bar_window <= 1 and interval == Interval.MINUTE:
            # 1 分钟线：直接使用 on_bar 作为回调
            self.bg = BarGenerator(self.on_bar)
        else:
            # N 分钟/小时线：先生成 1 分钟线，再合成目标周期
            self.bg = BarGenerator(
                self.on_bar,  # 1 分钟线回调（用于预热历史数据）
                window=self.bar_window,
                on_window_bar=self.on_bar,  # 目标周期回调
                interval=interval,
            )
        self.write_log(
            f"K线生成器: window={self.bar_window}, interval={self.bar_interval}"
        )

        # 加载历史数据用于初始化均线（周期需与 bar_interval 匹配）
        load_interval = Interval.HOUR if self.bar_interval == "HOUR" else Interval.MINUTE
        self.load_bar(days=10, interval=load_interval)

        self.write_log("策略初始化完成")
        logger.info("策略 %s 初始化完成", self.strategy_name)

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

    def on_tick(self, tick: TickData) -> None:
        """Tick 数据回调（实盘时由 CTA 引擎调用）."""
        if self.bg:
            self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        """K 线数据更新回调."""
        # 更新 K 线数据到管理器
        self.am.update_bar(bar)

        # 等待足够的数据来计算均线
        if not self.am.inited:
            return

        # 计算快慢均线
        self.fast_ma = self.am.sma(self.fast_window)
        self.slow_ma = self.am.sma(self.slow_window)

        # 判断均线趋势
        if self.fast_ma > self.slow_ma:
            new_trend = 1  # 多头趋势
        elif self.fast_ma < self.slow_ma:
            new_trend = -1  # 空头趋势
        else:
            new_trend = 0  # 无趋势

        # 检测趋势变化并生成交易信号
        if new_trend != self.ma_trend and new_trend != 0:
            if new_trend == 1:
                # 快线上穿慢线，做多
                self._handle_long_signal(bar)
            elif new_trend == -1:
                # 快线下穿慢线，做空
                self._handle_short_signal(bar)

        self.ma_trend = new_trend

        # 更新 GUI 显示
        self.put_event()

    def _handle_long_signal(self, bar: BarData) -> None:
        """处理做多信号."""
        self.write_log(
            f"做多信号: 快线={self.fast_ma:.2f}, 慢线={self.slow_ma:.2f}, "
            f"当前持仓={self.pos}"
        )
        logger.info(
            "策略 %s 做多信号: 价格=%.2f, 快线=%.2f, 慢线=%.2f",
            self.strategy_name,
            bar.close_price,
            self.fast_ma,
            self.slow_ma,
        )

        # 如果有空仓，先平仓
        if self.pos < 0:
            self.cover(bar.close_price, abs(self.pos))
            self.write_log(f"平空仓: {abs(self.pos)} 手")

        # 开多仓
        self.buy(bar.close_price, self.fixed_size)
        self.write_log(f"开多仓: {self.fixed_size} 手")

    def _handle_short_signal(self, bar: BarData) -> None:
        """处理做空信号."""
        self.write_log(
            f"做空信号: 快线={self.fast_ma:.2f}, 慢线={self.slow_ma:.2f}, "
            f"当前持仓={self.pos}"
        )
        logger.info(
            "策略 %s 做空信号: 价格=%.2f, 快线=%.2f, 慢线=%.2f",
            self.strategy_name,
            bar.close_price,
            self.fast_ma,
            self.slow_ma,
        )

        # 如果有多仓，先平仓
        if self.pos > 0:
            self.sell(bar.close_price, self.pos)
            self.write_log(f"平多仓: {self.pos} 手")

        # 开空仓
        self.short(bar.close_price, self.fixed_size)
        self.write_log(f"开空仓: {self.fixed_size} 手")

    def on_trade(self, trade: TradeData) -> None:
        """成交回调."""
        self.write_log(
            f"成交: {trade.direction.value} {trade.offset.value} "
            f"{trade.volume}手 @ {trade.price}"
        )
        logger.info(
            "策略 %s 成交: %s %s %.0f手 @ %.2f",
            self.strategy_name,
            trade.direction.value,
            trade.offset.value,
            trade.volume,
            trade.price,
        )
        # 同步数据到磁盘（用于断点恢复）
        self.sync_data()
        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """订单状态更新回调."""
        self.write_log(
            f"订单: {order.vt_orderid} "
            f"{order.direction.value} {order.offset.value} "
            f"{order.volume}@{order.price:.0f} -> {order.status.value}"
        )
        self.put_event()  # 刷新 GUI 显示
