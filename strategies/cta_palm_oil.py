"""
棕榈油双均线 CTA 策略（回测用）.

此文件放置在仓库根目录的 strategies/ 下，供 vnpy_ctabacktester GUI 加载。

================================================================================
如何在 GUI 的 CTA 回测页使用该策略
================================================================================

1. 启动 Trader GUI:
   uv run python -m qp.runtime.trader_app --profile research

2. 在菜单栏点击「功能」→「CTA回测」打开回测界面

3. 在「交易策略」下拉框中选择：CtaPalmOilStrategy

4. 填写回测参数：
   - 本地代码: p0.DCE
   - K线周期: d (日线)
   - 开始日期: 2025-01-15（约一年前）
   - 结束日期: 2026-01-13
   - 手续费率: 0.0001 (万分之一)
   - 交易滑点: 2
   - 合约乘数: 10
   - 价格跳动: 2
   - 回测资金: 1000000

5. 点击「开始回测」

================================================================================
ArrayManager 预热与回测数据量关系
================================================================================

策略使用 ArrayManager 计算均线，需要累积足够的 K 线数据才能初始化：
- ArrayManager size = slow_window + 10（默认 30）
- 数据量必须 > size，否则 am.inited 永远为 False，无交易信号

**重要**：回测数据量应至少为 ArrayManager size 的 2 倍，以产生足够的交易信号。

================================================================================
vt_symbol 规范
================================================================================

格式: {品种代码}{合约月份}.{交易所}

棕榈油示例:
- p0.DCE    -> 棕榈油连续合约 (大商所)
- p2501.DCE -> 棕榈油 2025年01月合约 (大商所)

================================================================================
"""

import logging
from vnpy_ctastrategy import CtaTemplate
from vnpy.trader.object import BarData, TradeData
from vnpy.trader.utility import ArrayManager

logger = logging.getLogger(__name__)


class CtaPalmOilStrategy(CtaTemplate):
    """
    棕榈油双均线策略.

    策略逻辑：
    - 当快速均线上穿慢速均线时，做多
    - 当快速均线下穿慢速均线时，做空
    - 持仓时发生反向信号则先平仓再反向开仓

    注意：ArrayManager 需要累积 slow_window + 10 根 K 线才能初始化。
    """

    author: str = "QuantPlus"

    # 策略参数（可在 GUI 中配置）
    fast_window: int = 10
    slow_window: int = 20
    fixed_size: int = 1

    # 参数列表（GUI 显示用）
    parameters: list[str] = [
        "fast_window",
        "slow_window",
        "fixed_size",
    ]

    # 策略变量（运行时状态）
    fast_ma: float = 0.0
    slow_ma: float = 0.0
    ma_trend: int = 0  # 1=多头趋势, -1=空头趋势, 0=无趋势
    bar_count: int = 0  # K 线计数（诊断用）

    # 变量列表（GUI 显示用）
    variables: list[str] = [
        "fast_ma",
        "slow_ma",
        "ma_trend",
        "bar_count",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """初始化策略."""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 初始化 K 线管理器
        # 重要：size 必须小于回测数据量，否则 am.inited 永远为 False
        # 修复：使用 slow_window + 10，而非 max(slow_window, 50) + 10
        self.am: ArrayManager = ArrayManager(size=self.slow_window + 10)

    def on_init(self) -> None:
        """策略初始化回调."""
        self.write_log(f"策略初始化: ArrayManager size={self.am.size}")
        logger.info(
            "策略 %s 初始化: fast=%d, slow=%d, am.size=%d",
            self.strategy_name, self.fast_window, self.slow_window, self.am.size
        )

        # 参数校验
        if self.fast_window >= self.slow_window:
            self.write_log(f"警告: fast_window({self.fast_window}) >= slow_window({self.slow_window})")

        # 加载历史数据（用于预热 ArrayManager）
        self.load_bar(10)

    def on_start(self) -> None:
        """策略启动回调."""
        self.write_log("策略启动")
        self.bar_count = 0

    def on_stop(self) -> None:
        """策略停止回调."""
        self.write_log(f"策略停止: 共处理 {self.bar_count} 根 K 线")

    def on_bar(self, bar: BarData) -> None:
        """K 线数据更新回调."""
        self.bar_count += 1

        # 更新 K 线数据
        self.am.update_bar(bar)

        # 等待足够数据初始化 ArrayManager
        if not self.am.inited:
            # 诊断：记录首次初始化的时刻
            if self.bar_count == self.am.size:
                logger.info(
                    "策略 %s: bar %d, am.inited 即将变为 True (size=%d)",
                    self.strategy_name, self.bar_count, self.am.size
                )
            return

        # 首次初始化时记录
        if self.bar_count == self.am.size:
            self.write_log(f"ArrayManager 初始化完成: bar_count={self.bar_count}")

        # 计算均线
        self.fast_ma = self.am.sma(self.fast_window)
        self.slow_ma = self.am.sma(self.slow_window)

        # 判断趋势
        if self.fast_ma > self.slow_ma:
            new_trend = 1
        elif self.fast_ma < self.slow_ma:
            new_trend = -1
        else:
            new_trend = 0

        # 趋势变化时交易
        if new_trend != self.ma_trend and new_trend != 0:
            trend_name = "多头" if new_trend == 1 else "空头"
            self.write_log(
                f"趋势变化: {self.ma_trend} -> {new_trend} ({trend_name}), "
                f"fast_ma={self.fast_ma:.2f}, slow_ma={self.slow_ma:.2f}"
            )

            if new_trend == 1:
                # 做多
                if self.pos < 0:
                    self.write_log(f"平空仓: {abs(self.pos)} 手 @ {bar.close_price}")
                    self.cover(bar.close_price, abs(self.pos))
                self.write_log(f"开多仓: {self.fixed_size} 手 @ {bar.close_price}")
                self.buy(bar.close_price, self.fixed_size)
            elif new_trend == -1:
                # 做空
                if self.pos > 0:
                    self.write_log(f"平多仓: {self.pos} 手 @ {bar.close_price}")
                    self.sell(bar.close_price, self.pos)
                self.write_log(f"开空仓: {self.fixed_size} 手 @ {bar.close_price}")
                self.short(bar.close_price, self.fixed_size)

        self.ma_trend = new_trend
        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        """成交回调."""
        self.write_log(
            f"成交: {trade.direction.value} {trade.offset.value} "
            f"{trade.volume} 手 @ {trade.price}"
        )
        logger.info(
            "策略 %s 成交: %s %s %.0f 手 @ %.2f",
            self.strategy_name, trade.direction.value, trade.offset.value,
            trade.volume, trade.price
        )
        self.put_event()
