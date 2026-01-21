# src/qp/strategies/cta_chan_v1.py
"""
缠论策略 V1 - 基于二买/二卖信号的趋势策略.

核心逻辑（来自 changege 研究脚本）：
1. 使用 5 分钟 K 线识别分型和笔
2. 使用 15 分钟 MACD 作为趋势过滤
3. 二买信号：低点抬高 + MACD 背驰 + 大周期多头
4. 二卖信号：高点降低 + MACD 背驰 + 大周期空头
5. 风控：P1 硬止损 + ATR 移动止损

数据要求：
- 回测时使用 1 分钟 K 线数据
- 策略内部增量合成 5 分钟和 15 分钟 K 线

参考回测结果（原始脚本）：
- Dataset 1 (p2509): 净利润 1455, 交易数 145, 胜率 46.90%, 最大回撤 349
- Dataset 2 (p2601): 净利润 215, 交易数 96, 胜率 42.71%, 最大回撤 400
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from vnpy.trader.object import BarData, TickData, TradeData, OrderData
from vnpy.trader.utility import BarGenerator
from vnpy.trader.constant import Interval
from vnpy_ctastrategy import CtaTemplate
from vnpy_ctastrategy.base import StopOrder

logger = logging.getLogger(__name__)


class CtaChanV1Strategy(CtaTemplate):
    """
    缠论策略 V1（优化版 - 增量计算）.

    核心信号：
    - 二买：低点抬高 + MACD 背驰 + 大周期多头趋势
    - 二卖：高点降低 + MACD 背驰 + 大周期空头趋势

    风控系统：
    - P1 硬止损：前一个同向笔端点
    - ATR 移动止损：浮盈 > 1.5 ATR 后激活
    """

    author: str = "QuantPlus"

    # -------------------------
    # 可配置参数
    # -------------------------
    # MACD 参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ATR 参数
    atr_window: int = 14
    atr_trailing_mult: float = 3.0   # ATR 移动止损倍数
    atr_activate_mult: float = 1.5   # 激活移动止损的浮盈 ATR 倍数

    # 笔构建参数
    min_bi_gap: int = 5              # 笔端点最小间隔（5 分钟 K 线数）

    # 合约参数
    fixed_volume: int = 1            # 固定手数

    # 调试
    debug: bool = False

    parameters: list[str] = [
        "macd_fast", "macd_slow", "macd_signal",
        "atr_window", "atr_trailing_mult", "atr_activate_mult",
        "min_bi_gap", "fixed_volume", "debug",
    ]

    # -------------------------
    # 运行时变量
    # -------------------------
    bar_count: int = 0
    bi_count: int = 0
    signal: str = ""

    variables: list[str] = [
        "bar_count", "bi_count", "signal",
    ]

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict[str, Any],
    ) -> None:
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 实盘用 BarGenerator (Tick -> 1m Bar)
        self.bg: Optional[BarGenerator] = None

        # K 线合成缓存（增量方式）
        self._window_bar_5m: Optional[dict] = None
        self._last_window_end_5m: Optional[datetime] = None
        self._window_bar_15m: Optional[dict] = None
        self._last_window_end_15m: Optional[datetime] = None

        # 5m K 线数据（仅保留必要数量）
        self._bars_5m: deque = deque(maxlen=200)
        self._high_5m: deque = deque(maxlen=10)
        self._low_5m: deque = deque(maxlen=10)
        self._diff_5m_history: deque = deque(maxlen=500)

        # MACD 增量计算缓存
        self._ema_fast_5m: float = 0.0
        self._ema_slow_5m: float = 0.0
        self._ema_signal_5m: float = 0.0
        self._ema_fast_15m: float = 0.0
        self._ema_slow_15m: float = 0.0
        self._ema_signal_15m: float = 0.0
        self._macd_inited_5m: bool = False
        self._macd_inited_15m: bool = False

        self.diff_5m: float = 0.0
        self.dea_5m: float = 0.0
        self.diff_15m: float = 0.0
        self.dea_15m: float = 0.0

        # 用于 shift(1) 效果的延迟 MACD
        self._prev_diff_15m: float = 0.0
        self._prev_dea_15m: float = 0.0

        # ATR 增量计算缓存
        self._tr_values: deque = deque(maxlen=14)
        self._prev_close_5m: float = 0.0
        self.atr: float = 0.0

        # 笔端点列表
        self._bi_points: list[dict] = []

        # 交易状态
        self._position: int = 0
        self._entry_price: float = 0.0
        self._stop_price: float = 0.0
        self._trailing_active: bool = False

        # 5m bar 计数
        self._bar_5m_count: int = 0

        logger.info("策略初始化: %s", strategy_name)

    def on_init(self) -> None:
        self.write_log(f"策略初始化: {self.strategy_name}")

        # 创建 BarGenerator（实盘 Tick -> 1m Bar）
        self.bg = BarGenerator(self._on_1m_bar)

        # 加载历史数据（1 分钟）
        self.load_bar(60)

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
        """
        1 分钟 K 线回调.

        使用增量方式合成 5m/15m K 线并计算指标。
        """
        self.bar_count += 1

        # 转换为 dict 格式
        bar_dict = {
            'datetime': bar.datetime,
            'open': bar.open_price,
            'high': bar.high_price,
            'low': bar.low_price,
            'close': bar.close_price,
            'volume': bar.volume,
        }

        # 增量更新 15m K 线
        bar_15m = self._update_15m_bar(bar_dict)
        if bar_15m:
            self._on_15m_bar(bar_15m)

        # 增量更新 5m K 线
        bar_5m = self._update_5m_bar(bar_dict)
        if bar_5m:
            self._on_5m_bar(bar_5m)

        self.put_event()

    def _on_1m_bar(self, bar: BarData) -> None:
        """1 分钟 K 线回调（来自 BarGenerator）."""
        self.on_bar(bar)

    def _get_window_end(self, dt: datetime, window: int) -> datetime:
        """
        计算窗口结束时间.

        与 pandas resample('Nmin', label='right', closed='right') 完全一致。
        """
        total_minutes = dt.hour * 60 + dt.minute
        window_end_minutes = math.ceil(total_minutes / window) * window

        hours = window_end_minutes // 60
        minutes = window_end_minutes % 60

        result = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        result += pd.Timedelta(hours=hours, minutes=minutes)

        return result

    def _update_5m_bar(self, bar: dict) -> Optional[dict]:
        """增量更新 5 分钟 K 线."""
        window_end = self._get_window_end(bar['datetime'], 5)

        result = None
        if self._window_bar_5m is not None:
            if window_end != self._last_window_end_5m:
                result = self._window_bar_5m.copy()
                self._window_bar_5m = None

        if self._window_bar_5m is None:
            self._window_bar_5m = {
                'datetime': window_end,
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume']
            }
        else:
            self._window_bar_5m['high'] = max(self._window_bar_5m['high'], bar['high'])
            self._window_bar_5m['low'] = min(self._window_bar_5m['low'], bar['low'])
            self._window_bar_5m['close'] = bar['close']
            self._window_bar_5m['volume'] += bar['volume']

        self._last_window_end_5m = window_end
        return result

    def _update_15m_bar(self, bar: dict) -> Optional[dict]:
        """增量更新 15 分钟 K 线."""
        window_end = self._get_window_end(bar['datetime'], 15)

        result = None
        if self._window_bar_15m is not None:
            if window_end != self._last_window_end_15m:
                result = self._window_bar_15m.copy()
                self._window_bar_15m = None

        if self._window_bar_15m is None:
            self._window_bar_15m = {
                'datetime': window_end,
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume']
            }
        else:
            self._window_bar_15m['high'] = max(self._window_bar_15m['high'], bar['high'])
            self._window_bar_15m['low'] = min(self._window_bar_15m['low'], bar['low'])
            self._window_bar_15m['close'] = bar['close']
            self._window_bar_15m['volume'] += bar['volume']

        self._last_window_end_15m = window_end
        return result

    def _update_macd_5m(self, close: float) -> None:
        """增量更新 5 分钟 MACD."""
        alpha_fast = 2.0 / (self.macd_fast + 1)
        alpha_slow = 2.0 / (self.macd_slow + 1)
        alpha_signal = 2.0 / (self.macd_signal + 1)

        if not self._macd_inited_5m:
            self._ema_fast_5m = close
            self._ema_slow_5m = close
            self._macd_inited_5m = True
            diff = 0.0
            self._ema_signal_5m = 0.0
        else:
            self._ema_fast_5m = alpha_fast * close + (1 - alpha_fast) * self._ema_fast_5m
            self._ema_slow_5m = alpha_slow * close + (1 - alpha_slow) * self._ema_slow_5m
            diff = self._ema_fast_5m - self._ema_slow_5m
            self._ema_signal_5m = alpha_signal * diff + (1 - alpha_signal) * self._ema_signal_5m

        self.diff_5m = diff
        self.dea_5m = self._ema_signal_5m

    def _update_macd_15m(self, close: float) -> None:
        """增量更新 15 分钟 MACD."""
        alpha_fast = 2.0 / (self.macd_fast + 1)
        alpha_slow = 2.0 / (self.macd_slow + 1)
        alpha_signal = 2.0 / (self.macd_signal + 1)

        if not self._macd_inited_15m:
            self._ema_fast_15m = close
            self._ema_slow_15m = close
            self._macd_inited_15m = True
            diff = 0.0
            self._ema_signal_15m = 0.0
        else:
            self._ema_fast_15m = alpha_fast * close + (1 - alpha_fast) * self._ema_fast_15m
            self._ema_slow_15m = alpha_slow * close + (1 - alpha_slow) * self._ema_slow_15m
            diff = self._ema_fast_15m - self._ema_slow_15m
            self._ema_signal_15m = alpha_signal * diff + (1 - alpha_signal) * self._ema_signal_15m

        self.diff_15m = diff
        self.dea_15m = self._ema_signal_15m

    def _update_atr(self, bar: dict) -> None:
        """增量更新 ATR."""
        if self._prev_close_5m == 0.0:
            self._prev_close_5m = bar['close']
            return

        high_low = bar['high'] - bar['low']
        high_close = abs(bar['high'] - self._prev_close_5m)
        low_close = abs(bar['low'] - self._prev_close_5m)
        tr = max(high_low, high_close, low_close)

        self._tr_values.append(tr)
        self._prev_close_5m = bar['close']

        if len(self._tr_values) >= self.atr_window:
            self.atr = sum(self._tr_values) / len(self._tr_values)

    def _on_15m_bar(self, bar: dict) -> None:
        """15 分钟 K 线回调."""
        # 实现 shift(1) 效果：先保存当前值作为 prev，再更新
        self._prev_diff_15m = self.diff_15m
        self._prev_dea_15m = self.dea_15m

        self._update_macd_15m(bar['close'])

    def _on_5m_bar(self, bar: dict) -> None:
        """5 分钟 K 线回调 - 核心交易逻辑."""
        self._bars_5m.append(bar)
        self._high_5m.append(bar['high'])
        self._low_5m.append(bar['low'])

        self._bar_5m_count += 1
        bar_idx = self._bar_5m_count - 1

        self._update_macd_5m(bar['close'])
        self._diff_5m_history.append(self.diff_5m)
        self._update_atr(bar)

        if bar_idx < 5:
            return

        # 检查止损
        if self._position != 0:
            if self._check_stop_loss(bar):
                return

        # 更新笔结构
        new_bi = self._update_bi_structure(bar_idx)

        # 只在新笔形成时检查入场信号
        if new_bi:
            self._check_entry_signal(bar)

        # 更新移动止损
        if self._position != 0:
            self._update_trailing_stop(bar)

    def _update_bi_structure(self, bar_idx: int) -> bool:
        """更新笔结构."""
        if len(self._high_5m) < 3:
            return False

        curr_high = self._high_5m[-1]
        curr_low = self._low_5m[-1]
        prev_high = self._high_5m[-2]
        prev_low = self._low_5m[-2]
        prev2_high = self._high_5m[-3]
        prev2_low = self._low_5m[-3]

        is_top = (prev_high > prev2_high) and (prev_high > curr_high)
        is_bot = (prev_low < prev2_low) and (prev_low < curr_low)

        fractal_idx = bar_idx - 1
        fractal_diff = self._diff_5m_history[-2] if len(self._diff_5m_history) >= 2 else self.diff_5m

        new_bi = False
        if is_top:
            candidate = {
                'idx': fractal_idx,
                'type': 'top',
                'price': prev_high,
                'diff': fractal_diff
            }
            new_bi = self._process_bi_candidate(candidate)
        elif is_bot:
            candidate = {
                'idx': fractal_idx,
                'type': 'bottom',
                'price': prev_low,
                'diff': fractal_diff
            }
            new_bi = self._process_bi_candidate(candidate)

        self.bi_count = len(self._bi_points)
        return new_bi

    def _process_bi_candidate(self, candidate: dict) -> bool:
        """处理笔端点候选."""
        if not self._bi_points:
            self._bi_points.append(candidate)
            return False

        last_bi = self._bi_points[-1]

        if last_bi['type'] == candidate['type']:
            # 同向延伸
            if candidate['type'] == 'top' and candidate['price'] > last_bi['price']:
                self._bi_points[-1] = candidate
            elif candidate['type'] == 'bottom' and candidate['price'] < last_bi['price']:
                self._bi_points[-1] = candidate
            return False
        else:
            # 异向成笔
            if candidate['idx'] - last_bi['idx'] >= self.min_bi_gap:
                self._bi_points.append(candidate)
                return True
            return False

    def _check_entry_signal(self, bar: dict) -> None:
        """检查入场信号."""
        if len(self._bi_points) < 5:
            return

        p_now = self._bi_points[-1]
        p_last = self._bi_points[-2]
        p_prev = self._bi_points[-3]
        p_prev2 = self._bi_points[-5]

        # 使用延迟的 15m MACD（与原始脚本的 shift(1)+ffill 一致）
        is_bull_trend = self._prev_diff_15m > self._prev_dea_15m
        is_bear_trend = self._prev_diff_15m < self._prev_dea_15m

        # 二买信号（允许新建仓和反向开仓，禁止同向重复开仓）
        if p_now['type'] == 'bottom':
            is_structure_ok = p_now['price'] > p_prev['price']
            is_divergence = p_prev['diff'] > p_prev2['diff']

            if is_structure_ok and is_divergence and is_bull_trend:
                if self._position != 1:
                    self._execute_buy(bar, p_prev['price'])

        # 二卖信号（允许新建仓和反向开仓，禁止同向重复开仓）
        elif p_now['type'] == 'top':
            is_structure_ok = p_now['price'] < p_prev['price']
            is_divergence = p_prev['diff'] < p_prev2['diff']

            if is_structure_ok and is_divergence and is_bear_trend:
                if self._position != -1:
                    self._execute_short(bar, p_prev['price'])

    def _execute_buy(self, bar: dict, p1_price: float) -> None:
        """执行买入."""
        if self.pos < 0:
            self.cover(bar['close'], abs(self.pos))
            self.write_log(f"平空: price={bar['close']:.0f}")

        self.buy(bar['close'], self.fixed_volume)

        self._position = 1
        self._entry_price = bar['close']
        self._stop_price = p1_price - 1
        self._trailing_active = False
        self.signal = "二买"

        self.write_log(f"开多: price={bar['close']:.0f}, stop={self._stop_price:.0f}")

    def _execute_short(self, bar: dict, p1_price: float) -> None:
        """执行卖出."""
        if self.pos > 0:
            self.sell(bar['close'], abs(self.pos))
            self.write_log(f"平多: price={bar['close']:.0f}")

        self.short(bar['close'], self.fixed_volume)

        self._position = -1
        self._entry_price = bar['close']
        self._stop_price = p1_price + 1
        self._trailing_active = False
        self.signal = "二卖"

        self.write_log(f"开空: price={bar['close']:.0f}, stop={self._stop_price:.0f}")

    def _check_stop_loss(self, bar: dict) -> bool:
        """检查止损."""
        sl_hit = False
        exit_price = 0.0

        if self._position == 1:
            if bar['low'] <= self._stop_price:
                sl_hit = True
                exit_price = self._stop_price
        elif self._position == -1:
            if bar['high'] >= self._stop_price:
                sl_hit = True
                exit_price = self._stop_price

        if sl_hit:
            if self._position == 1:
                self.sell(exit_price, abs(self.pos))
                self.write_log(f"多头止损: price={exit_price:.0f}")
            else:
                self.cover(exit_price, abs(self.pos))
                self.write_log(f"空头止损: price={exit_price:.0f}")

            self._position = 0
            self._trailing_active = False
            self.signal = "止损"
            return True

        return False

    def _update_trailing_stop(self, bar: dict) -> None:
        """更新移动止损."""
        if self.atr <= 0:
            return

        if self._position == 1:
            float_pnl = bar['close'] - self._entry_price
        else:
            float_pnl = self._entry_price - bar['close']

        if not self._trailing_active:
            if float_pnl > (self.atr * self.atr_activate_mult):
                self._trailing_active = True

        if self._trailing_active:
            if self._position == 1:
                new_stop = bar['high'] - (self.atr * self.atr_trailing_mult)
                if new_stop > self._stop_price:
                    self._stop_price = new_stop
            else:
                new_stop = bar['low'] + (self.atr * self.atr_trailing_mult)
                if new_stop < self._stop_price:
                    self._stop_price = new_stop

    def on_trade(self, trade: TradeData) -> None:
        """成交回调."""
        self.write_log(
            f"成交: {trade.direction.value} {trade.offset.value} "
            f"{trade.volume}手 @ {trade.price:.0f}"
        )
        self.sync_data()
        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """订单状态更新回调."""
        pass

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """停止单回调."""
        pass
