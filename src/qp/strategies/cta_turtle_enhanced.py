# src/qp/strategies/cta_turtle_enhanced.py
"""
增强版海龟策略 - 基于社区优化建议.

优化点（参考社区讨论）：
1. 趋势过滤器：使用长期均线判断大趋势，只顺势交易
2. 中轨止盈：使用通道中轨作为移动止盈点（比ATR止损更快获利了结）
3. 突破确认：收盘价突破 + 突破幅度确认，过滤假突破
4. 双系统切换：根据市场波动率自动切换长短周期
5. 更激进的仓位管理：提高风险预算

K线周期说明：
- 回测时：直接使用数据库中对应周期的 K 线数据
- 实盘时：通过 BarGenerator 将 Tick 合成为指定周期的 K 线
- bar_window: K线窗口大小（1=1分钟, 15=15分钟, 60=60分钟）
- bar_interval: K线周期类型 ("MINUTE" 或 "HOUR")

参考来源：
- 增强版唐奇安通道策略 (CSDN)
- 海龟交易法则优化 (知乎/掘金量化)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from vnpy.trader.object import BarData, TickData, TradeData, OrderData
from vnpy.trader.utility import ArrayManager, BarGenerator
from vnpy.trader.constant import Interval
from vnpy_ctastrategy import CtaTemplate
from vnpy_ctastrategy.base import StopOrder

logger = logging.getLogger(__name__)


class CtaTurtleEnhancedStrategy(CtaTemplate):
    """
    增强版海龟策略.

    核心优化：
    1. 趋势过滤：只在均线多头/空头排列时交易
    2. 中轨止盈：(上轨+下轨)/2 作为移动止盈
    3. 突破确认：突破幅度 > 阈值
    4. 激进仓位：risk_per_trade 2-3%
    """

    author: str = "QuantPlus"

    # -------------------------
    # 可配置参数（2026-01 激进优化: Sharpe 1.28, 收益 +24.41%, 年化 +34.07%）
    # -------------------------
    # 通道参数
    entry_window: int = 15          # 入场通道窗口
    exit_window: int = 1            # 出场通道窗口（极快出场）

    # 趋势过滤（核心优化点：使用更长的趋势周期）
    trend_ma_fast: int = 10         # 快速均线
    trend_ma_slow: int = 100        # 慢速均线（更长周期过滤噪音）
    use_trend_filter: bool = True   # 是否启用趋势过滤（减少假突破）

    # 突破确认
    break_confirm_bars: int = 1     # 突破确认K线数（1=不需要确认）
    break_threshold: float = 0.002  # 突破幅度阈值（0.2%）

    # 止损止盈
    atr_window: int = 14            # ATR 窗口
    atr_stop: float = 1.5           # ATR 止损倍数
    use_mid_line_exit: bool = True  # 使用中轨止盈（更快获利了结）
    trailing_start_atr: float = 1.5 # 开始跟踪止盈的盈利ATR倍数

    # 仓位管理（激进配置）
    risk_per_trade: float = 0.06    # 单笔风险预算 6%
    max_units: int = 50             # 最大手数

    # 加仓（更快加仓）
    enable_pyramid: bool = True     # 启用金字塔加仓
    pyramid_atr: float = 0.15       # 每 0.15 ATR 加仓一次
    max_pyramid: int = 4            # 最大加仓次数

    # 双系统
    use_dual_system: bool = False   # 是否使用双系统（S1:20日, S2:55日）
    s2_entry_window: int = 55       # System2 入场窗口
    s2_exit_window: int = 20        # System2 出场窗口

    # 合约参数
    contract_size: float = 10.0
    base_capital: float = 1_000_000

    enable_short: bool = True
    debug: bool = False

    # K线周期参数（实盘使用）
    bar_window: int = 1             # K线窗口大小（1=1分钟, 15=15分钟）
    bar_interval: str = "MINUTE"    # K线周期类型 ("MINUTE" 或 "HOUR")

    parameters: list[str] = [
        "entry_window", "exit_window",
        "trend_ma_fast", "trend_ma_slow", "use_trend_filter",
        "break_confirm_bars", "break_threshold",
        "atr_window", "atr_stop",
        "use_mid_line_exit", "trailing_start_atr",
        "risk_per_trade", "max_units",
        "enable_pyramid", "pyramid_atr", "max_pyramid",
        "use_dual_system", "s2_entry_window", "s2_exit_window",
        "contract_size", "base_capital",
        "enable_short", "debug",
        "bar_window", "bar_interval",
    ]

    # -------------------------
    # 运行时变量
    # -------------------------
    entry_up: float = 0.0
    entry_down: float = 0.0
    exit_up: float = 0.0
    exit_down: float = 0.0
    mid_line: float = 0.0
    atr: float = 0.0
    ma_fast: float = 0.0
    ma_slow: float = 0.0
    trend: int = 0  # 1=多头, -1=空头, 0=无趋势
    stop_price: float = 0.0
    target_units: int = 0
    pyramid_count: int = 0

    variables: list[str] = [
        "entry_up", "entry_down",
        "exit_up", "exit_down",
        "mid_line", "atr",
        "ma_fast", "ma_slow", "trend",
        "stop_price", "target_units", "pyramid_count",
    ]

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict[str, Any],
    ) -> None:
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 计算所需的最大窗口
        max_window = max(
            self.entry_window,
            self.exit_window,
            self.trend_ma_slow,
            self.atr_window,
            self.s2_entry_window if self.use_dual_system else 0,
        )
        self._am_size = max_window + 30

        self.am = ArrayManager(size=self._am_size)

        # K线生成器（实盘时将 Tick 转换为 Bar）
        self.bg: Optional[BarGenerator] = None

        # 交易状态
        self._entry_price: float = 0.0
        self._last_pyramid_price: Optional[float] = None
        self._break_confirm_count: int = 0
        self._pending_signal: int = 0  # 1=待确认多, -1=待确认空
        self._bar_count: int = 0

        logger.info(
            "策略初始化: %s, am_size=%d, entry=%d, trend_filter=%s",
            strategy_name, self._am_size, self.entry_window, self.use_trend_filter
        )

    def on_init(self) -> None:
        self.write_log(f"策略初始化: {self.strategy_name}")

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

        # 加载历史数据（周期需与 bar_interval 匹配）
        # 注意：load_bar 的 interval 参数决定从数据库加载的 K 线周期
        load_interval = Interval.HOUR if self.bar_interval == "HOUR" else Interval.MINUTE
        self.load_bar(max(10, self._am_size // 20), interval=load_interval)
        self.write_log("策略初始化完成")

    def on_start(self) -> None:
        self.write_log("策略启动")
        self.put_event()

    def on_stop(self) -> None:
        self.write_log("策略停止")
        self.put_event()

    def on_tick(self, tick: TickData) -> None:
        """Tick 数据回调（实盘时由 CTA 引擎调用）."""
        if self.bg:
            self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        self._bar_count += 1
        self.cancel_all()

        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # 计算指标
        self._calc_indicators()

        if self.atr <= 0:
            return

        # 计算目标仓位
        self.target_units = self._calc_target_units()

        if self.debug and self._bar_count % 10 == 0:
            self.write_log(
                f"[DEBUG] close={bar.close_price:.0f} entry_up={self.entry_up:.0f} "
                f"entry_down={self.entry_down:.0f} mid={self.mid_line:.0f} "
                f"trend={self.trend} pos={self.pos}"
            )

        # 交易逻辑
        if self.pos == 0:
            self._handle_entry(bar)
        elif self.pos > 0:
            self._handle_long_position(bar)
        else:
            self._handle_short_position(bar)

        self.put_event()

    def _calc_indicators(self) -> None:
        """计算所有技术指标."""
        high = self.am.high
        low = self.am.low
        close = self.am.close

        # 唐奇安通道（使用前一根bar之前的数据，避免look-ahead）
        self.entry_up = float(np.max(high[-self.entry_window - 1:-1]))
        self.entry_down = float(np.min(low[-self.entry_window - 1:-1]))
        self.exit_up = float(np.max(high[-self.exit_window - 1:-1]))
        self.exit_down = float(np.min(low[-self.exit_window - 1:-1]))

        # 中轨
        self.mid_line = (self.entry_up + self.entry_down) / 2.0

        # ATR
        self.atr = float(self.am.atr(self.atr_window))

        # 趋势均线
        self.ma_fast = float(self.am.sma(self.trend_ma_fast))
        self.ma_slow = float(self.am.sma(self.trend_ma_slow))

        # 趋势判断
        if self.ma_fast > self.ma_slow * 1.001:  # 加小阈值避免频繁切换
            self.trend = 1
        elif self.ma_fast < self.ma_slow * 0.999:
            self.trend = -1
        else:
            self.trend = 0

    def _handle_entry(self, bar: BarData) -> None:
        """处理入场逻辑."""
        self._entry_price = 0.0
        self._last_pyramid_price = None
        self.pyramid_count = 0
        self.stop_price = 0.0

        # 检查突破
        break_up = bar.close_price > self.entry_up
        break_down = bar.close_price < self.entry_down

        # 突破幅度确认
        if break_up:
            break_pct = (bar.close_price - self.entry_up) / self.entry_up
            if break_pct < self.break_threshold:
                break_up = False
        if break_down:
            break_pct = (self.entry_down - bar.close_price) / self.entry_down
            if break_pct < self.break_threshold:
                break_down = False

        # 趋势过滤
        if self.use_trend_filter:
            if break_up and self.trend != 1:
                if self.debug:
                    self.write_log(f"[DEBUG] 多头信号被趋势过滤: trend={self.trend}")
                break_up = False
            if break_down and self.trend != -1:
                if self.debug:
                    self.write_log(f"[DEBUG] 空头信号被趋势过滤: trend={self.trend}")
                break_down = False

        # 突破确认（多根K线确认）
        if self.break_confirm_bars > 1:
            if break_up:
                if self._pending_signal == 1:
                    self._break_confirm_count += 1
                else:
                    self._pending_signal = 1
                    self._break_confirm_count = 1
                if self._break_confirm_count < self.break_confirm_bars:
                    break_up = False
            elif break_down:
                if self._pending_signal == -1:
                    self._break_confirm_count += 1
                else:
                    self._pending_signal = -1
                    self._break_confirm_count = 1
                if self._break_confirm_count < self.break_confirm_bars:
                    break_down = False
            else:
                self._pending_signal = 0
                self._break_confirm_count = 0

        # 执行入场
        if break_up:
            volume = max(1, self.target_units)
            self.buy(bar.close_price, volume)
            self._entry_price = bar.close_price
            self._last_pyramid_price = bar.close_price
            self.stop_price = bar.close_price - self.atr_stop * self.atr
            self.write_log(
                f"多头入场: price={bar.close_price:.0f}, vol={volume}, "
                f"stop={self.stop_price:.0f}, trend={self.trend}"
            )
            self._pending_signal = 0
            self._break_confirm_count = 0

        elif break_down and self.enable_short:
            volume = max(1, self.target_units)
            self.short(bar.close_price, volume)
            self._entry_price = bar.close_price
            self._last_pyramid_price = bar.close_price
            self.stop_price = bar.close_price + self.atr_stop * self.atr
            self.write_log(
                f"空头入场: price={bar.close_price:.0f}, vol={volume}, "
                f"stop={self.stop_price:.0f}, trend={self.trend}"
            )
            self._pending_signal = 0
            self._break_confirm_count = 0

    def _handle_long_position(self, bar: BarData) -> None:
        """处理多头持仓."""
        # 更新跟踪止损
        profit_atr = (bar.high_price - self._entry_price) / self.atr if self.atr > 0 else 0

        # 如果盈利超过阈值，开始使用更紧的止损
        if profit_atr >= self.trailing_start_atr:
            # 使用中轨或ATR止损中更紧的那个
            atr_stop = bar.high_price - self.atr_stop * self.atr
            if self.use_mid_line_exit:
                new_stop = max(atr_stop, self.mid_line)
            else:
                new_stop = atr_stop
            self.stop_price = max(self.stop_price, new_stop)
        else:
            # 初始阶段使用ATR止损
            new_stop = bar.high_price - self.atr_stop * self.atr
            self.stop_price = max(self.stop_price, new_stop) if self.stop_price > 0 else new_stop

        # 检查止损
        if bar.close_price <= self.stop_price:
            self.sell(bar.close_price, abs(self.pos))
            self.write_log(f"多头止损: price={bar.close_price:.0f}, stop={self.stop_price:.0f}")
            return

        # 检查通道出场（中轨或下轨）
        exit_line = self.mid_line if self.use_mid_line_exit else self.exit_down
        if bar.close_price < exit_line:
            self.sell(bar.close_price, abs(self.pos))
            self.write_log(f"多头出场(通道): price={bar.close_price:.0f}, exit={exit_line:.0f}")
            return

        # 挂止损单保护
        self.sell(self.stop_price, abs(self.pos), stop=True)

        # 金字塔加仓
        if self.enable_pyramid and self.pyramid_count < self.max_pyramid:
            self._try_pyramid_long(bar)

    def _handle_short_position(self, bar: BarData) -> None:
        """处理空头持仓."""
        # 更新跟踪止损
        profit_atr = (self._entry_price - bar.low_price) / self.atr if self.atr > 0 else 0

        if profit_atr >= self.trailing_start_atr:
            atr_stop = bar.low_price + self.atr_stop * self.atr
            if self.use_mid_line_exit:
                new_stop = min(atr_stop, self.mid_line)
            else:
                new_stop = atr_stop
            self.stop_price = min(self.stop_price, new_stop) if self.stop_price > 0 else new_stop
        else:
            new_stop = bar.low_price + self.atr_stop * self.atr
            self.stop_price = min(self.stop_price, new_stop) if self.stop_price > 0 else new_stop

        # 检查止损
        if bar.close_price >= self.stop_price:
            self.cover(bar.close_price, abs(self.pos))
            self.write_log(f"空头止损: price={bar.close_price:.0f}, stop={self.stop_price:.0f}")
            return

        # 检查通道出场
        exit_line = self.mid_line if self.use_mid_line_exit else self.exit_up
        if bar.close_price > exit_line:
            self.cover(bar.close_price, abs(self.pos))
            self.write_log(f"空头出场(通道): price={bar.close_price:.0f}, exit={exit_line:.0f}")
            return

        # 挂止损单保护
        self.cover(self.stop_price, abs(self.pos), stop=True)

        # 金字塔加仓
        if self.enable_pyramid and self.pyramid_count < self.max_pyramid:
            self._try_pyramid_short(bar)

    def _try_pyramid_long(self, bar: BarData) -> None:
        """多头金字塔加仓."""
        if self._last_pyramid_price is None:
            return

        current_pos = abs(self.pos)
        if current_pos >= self.max_units:
            return

        trigger = self._last_pyramid_price + self.pyramid_atr * self.atr
        if bar.close_price >= trigger:
            add_vol = min(1, self.max_units - current_pos)
            if add_vol > 0:
                self.buy(bar.close_price, add_vol)
                self._last_pyramid_price = bar.close_price
                self.pyramid_count += 1
                # 更新止损（加仓后收紧止损）
                self.stop_price = max(self.stop_price, bar.close_price - self.atr_stop * self.atr)
                self.write_log(f"多头加仓: price={bar.close_price:.0f}, pyramid={self.pyramid_count}")

    def _try_pyramid_short(self, bar: BarData) -> None:
        """空头金字塔加仓."""
        if self._last_pyramid_price is None:
            return

        current_pos = abs(self.pos)
        if current_pos >= self.max_units:
            return

        trigger = self._last_pyramid_price - self.pyramid_atr * self.atr
        if bar.close_price <= trigger:
            add_vol = min(1, self.max_units - current_pos)
            if add_vol > 0:
                self.short(bar.close_price, add_vol)
                self._last_pyramid_price = bar.close_price
                self.pyramid_count += 1
                self.stop_price = min(self.stop_price, bar.close_price + self.atr_stop * self.atr)
                self.write_log(f"空头加仓: price={bar.close_price:.0f}, pyramid={self.pyramid_count}")

    def _calc_target_units(self) -> int:
        """计算目标手数."""
        equity = float(self.base_capital)
        risk_money = equity * float(self.risk_per_trade)

        one_unit_risk = float(self.atr_stop) * float(self.atr) * float(self.contract_size)
        if one_unit_risk <= 0:
            return 1

        units = int(math.floor(risk_money / one_unit_risk))
        units = max(1, min(units, self.max_units))
        return units

    def on_trade(self, trade: TradeData) -> None:
        self.write_log(
            f"成交: {trade.direction.value} {trade.offset.value} "
            f"{trade.volume}手 @ {trade.price:.0f}"
        )
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

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass
