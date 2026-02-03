# src/qp/strategies/cta_chan_pivot.py
"""
缠论中枢策略 - 基于中枢(Pivot/ZhongShu)的3B/3S信号.

核心逻辑（来自 chan0121Pivot 研究脚本）：
1. 使用 5 分钟 K 线，进行包含处理后识别分型和严格笔
2. 基于3笔重叠区域识别中枢(ZhongShu)
3. 3B信号：向上离开中枢 + 回踩不破中枢高点(ZG)
4. 3S信号：向下离开中枢 + 回抽不破中枢低点(ZD)
5. 2B/2S作为辅助信号（趋势延续补充）
6. 使用 15 分钟 MACD 作为趋势过滤
7. 风控：P1 硬止损 + ATR 移动止损

数据要求：
- 回测时使用 1 分钟 K 线数据
- 策略内部增量合成 5 分钟和 15 分钟 K 线
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

from qp.utils.chan_debugger import ChanDebugger

logger = logging.getLogger(__name__)


class CtaChanPivotStrategy(CtaTemplate):
    """
    缠论中枢策略（增量计算版）.

    核心信号：
    - 3B买点：向上离开中枢 + 回踩不破中枢高点(ZG) + 大周期多头
    - 3S卖点：向下离开中枢 + 回抽不破中枢低点(ZD) + 大周期空头
    - 2B/2S：辅助信号（低点抬高/高点降低 + MACD背驰）

    风控系统：
    - P1 硬止损：回踩/回抽点
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
    atr_trailing_mult: float = 2.0   # ATR 移动止损倍数 (S6: 2.0)
    atr_activate_mult: float = 1.5   # 激活移动止损的浮盈 ATR 倍数
    atr_entry_filter: float = 1.5    # 入场过滤：触发价与止损距离不超过 N 倍 ATR (S6: 1.5)

    # 笔构建参数
    min_bi_gap: int = 5              # 严格笔端点最小间隔 (S6: 5)

    # 中枢参数
    pivot_valid_range: int = 6       # 中枢有效范围（笔端点数）

    # 合约参数
    fixed_volume: int = 1            # 固定手数

    # --- S6 (Iter3-6) 新增参数 ---
    trend_filter: bool = True        # Iter3: 5m MACD方向一致性过滤
    disable_3s_short: bool = True    # Iter3: 禁用3S做空(改为平多退出)
    bi_amp_filter: bool = True       # Iter4: 笔幅度过滤
    bi_amp_min_atr: float = 1.5      # Iter4: 最小笔幅度(ATR倍数)
    macd_consistency: int = 3        # Iter4: MACD方向连续一致K线数
    macd15_mag_cap_atr: float = 3.0  # Iter6: 15m MACD幅度上限(ATR倍数)

    # 调试
    debug: bool = False
    debug_enabled: bool = True      # 是否启用Debug记录
    debug_log_console: bool = True  # 是否输出到控制台

    parameters: list[str] = [
        "macd_fast", "macd_slow", "macd_signal",
        "atr_window", "atr_trailing_mult", "atr_activate_mult", "atr_entry_filter",
        "min_bi_gap", "pivot_valid_range", "fixed_volume",
        "trend_filter", "disable_3s_short",
        "bi_amp_filter", "bi_amp_min_atr", "macd_consistency",
        "macd15_mag_cap_atr",
        "debug", "debug_enabled", "debug_log_console",
    ]

    # -------------------------
    # 运行时变量
    # -------------------------
    bar_count: int = 0
    bi_count: int = 0
    pivot_count: int = 0
    signal: str = ""

    variables: list[str] = [
        "bar_count", "bi_count", "pivot_count", "signal",
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

        # 包含处理后的 K 线
        self._k_lines: list[dict] = []
        self._inclusion_dir: int = 0  # 包含处理方向：1=向上, -1=向下, 0=未定

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

        # S6: 最近 N 根 5m MACD diff (用于 macd_consistency 检查)
        self._recent_diffs: list[float] = []

        # ATR 增量计算缓存
        self._tr_values: deque = deque(maxlen=14)
        self._prev_close_5m: float = 0.0
        self.atr: float = 0.0

        # 笔端点列表（严格笔）
        self._bi_points: list[dict] = []

        # 中枢列表
        self._pivots: list[dict] = []

        # 待触发信号
        self._pending_signal: Optional[dict] = None

        # 交易状态
        self._position: int = 0
        self._entry_price: float = 0.0
        self._stop_price: float = 0.0
        self._trailing_active: bool = False

        # 5m bar 计数
        self._bar_5m_count: int = 0

        # Debug工具
        self._debugger: Optional[ChanDebugger] = None
        self._signal_type: str = ""  # 当前信号类型(用于debug)

        logger.info("策略初始化: %s", strategy_name)

    def on_init(self) -> None:
        self.write_log(f"策略初始化: {self.strategy_name}")

        # 创建 BarGenerator（实盘 Tick -> 1m Bar）
        self.bg = BarGenerator(self._on_1m_bar)

        # 初始化Debug工具
        if self.debug_enabled:
            self._debugger = ChanDebugger(
                strategy_name=self.strategy_name,
                base_dir="data/debug",
                enabled=True,
                log_level="DEBUG",
                log_console=self.debug_log_console
            )
            # 保存配置
            self._debugger.save_config({
                "strategy_name": self.strategy_name,
                "vt_symbol": self.vt_symbol,
                "macd_fast": self.macd_fast,
                "macd_slow": self.macd_slow,
                "macd_signal": self.macd_signal,
                "atr_window": self.atr_window,
                "atr_trailing_mult": self.atr_trailing_mult,
                "atr_activate_mult": self.atr_activate_mult,
                "atr_entry_filter": self.atr_entry_filter,
                "min_bi_gap": self.min_bi_gap,
                "pivot_valid_range": self.pivot_valid_range,
                "fixed_volume": self.fixed_volume,
            })

        # 加载历史数据（1 分钟）
        self.load_bar(60)

        self.write_log("策略初始化完成")

    def on_start(self) -> None:
        self.write_log("策略启动")
        self.put_event()

    def on_stop(self) -> None:
        self.write_log("策略停止")
        # 保存debug摘要
        if self._debugger:
            self._debugger.close()
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

        # Debug: 记录1分钟K线（仅实盘模式）
        if self._debugger and self.trading:
            self._debugger.log_kline_1m(bar_dict)

        # 1. 持仓管理：检查止损（1分钟级别，仅实盘模式）
        if self.trading and self._position != 0:
            if self._check_stop_loss_1m(bar_dict):
                self.put_event()
                return

        # 2. 进场管理：检查待触发信号（1分钟级别，仅实盘模式）
        if self.trading and self._position == 0 and self._pending_signal:
            self._check_entry_1m(bar_dict)

        # 3. 增量更新 15m K 线
        bar_15m = self._update_15m_bar(bar_dict)
        if bar_15m:
            self._on_15m_bar(bar_15m)

        # 4. 增量更新 5m K 线
        bar_5m = self._update_5m_bar(bar_dict)
        if bar_5m:
            self._on_5m_bar(bar_5m)

        self.put_event()

    def _on_1m_bar(self, bar: BarData) -> None:
        """1 分钟 K 线回调（来自 BarGenerator）."""
        self.on_bar(bar)

    def _get_window_end(self, dt: datetime, window: int) -> datetime:
        """计算窗口结束时间."""
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

        # 更新当前窗口 bar
        if self._window_bar_5m is None or window_end != self._last_window_end_5m:
            # 新窗口开始
            self._window_bar_5m = {
                'datetime': window_end,
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume']
            }
        else:
            # 同一窗口，更新
            self._window_bar_5m['high'] = max(self._window_bar_5m['high'], bar['high'])
            self._window_bar_5m['low'] = min(self._window_bar_5m['low'], bar['low'])
            self._window_bar_5m['close'] = bar['close']
            self._window_bar_5m['volume'] += bar['volume']

        self._last_window_end_5m = window_end

        # 在边界分钟（与 pandas resample closed='right' 一致）输出完成的 bar
        if bar['datetime'].minute % 5 == 0:
            return self._window_bar_5m.copy()
        return None

    def _update_15m_bar(self, bar: dict) -> Optional[dict]:
        """增量更新 15 分钟 K 线."""
        window_end = self._get_window_end(bar['datetime'], 15)

        # 更新当前窗口 bar
        if self._window_bar_15m is None or window_end != self._last_window_end_15m:
            # 新窗口开始
            self._window_bar_15m = {
                'datetime': window_end,
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume']
            }
        else:
            # 同一窗口，更新
            self._window_bar_15m['high'] = max(self._window_bar_15m['high'], bar['high'])
            self._window_bar_15m['low'] = min(self._window_bar_15m['low'], bar['low'])
            self._window_bar_15m['close'] = bar['close']
            self._window_bar_15m['volume'] += bar['volume']

        self._last_window_end_15m = window_end

        # 在边界分钟输出完成的 bar
        if bar['datetime'].minute % 15 == 0:
            return self._window_bar_15m.copy()
        return None

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
        self._bar_5m_count += 1

        # 更新指标
        self._update_macd_5m(bar['close'])
        self._update_atr(bar)

        # Debug: 记录5分钟K线和指标（仅实盘模式）
        if self._debugger and self.trading:
            self._debugger.log_kline_5m(
                bar,
                diff_5m=self.diff_5m,
                dea_5m=self.dea_5m,
                atr=self.atr,
                diff_15m=self._prev_diff_15m,
                dea_15m=self._prev_dea_15m
            )

        # S6: 记录最近 MACD diff 用于 macd_consistency
        self._recent_diffs.append(self.diff_5m)
        if len(self._recent_diffs) > 10:
            self._recent_diffs.pop(0)

        # 构建当前 bar 数据（包含指标）
        bar_data = {
            'datetime': bar['datetime'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'diff': self.diff_5m,
            'atr': self.atr,
            'diff_15m': self._prev_diff_15m,
            'dea_15m': self._prev_dea_15m,
        }

        # 1. 包含处理
        self._process_inclusion(bar_data)

        # 2. 严格笔处理
        new_bi = self._process_bi()

        # 3. 更新移动止损（5分钟级别，仅实盘模式）
        if self.trading and self._position != 0:
            self._update_trailing_stop(bar_data)

        # 4. 信号检查（只在新笔形成时，仅实盘模式）
        if self.trading and new_bi:
            self._check_signal(bar_data, new_bi)

        # 更新显示变量
        self.bi_count = len(self._bi_points)
        self.pivot_count = len(self._pivots)

        # Debug: 记录缠论状态（仅实盘模式）
        if self._debugger and self.trading:
            self._debugger.log_chan_state(
                k_lines_count=len(self._k_lines),
                bi_count=len(self._bi_points),
                pivot_count=len(self._pivots),
                bi_points=self._bi_points,
                pivots=self._pivots
            )
            # 记录持仓状态
            if self._position != 0:
                self._debugger.log_position_status(
                    position=self._position,
                    entry_price=self._entry_price,
                    stop_price=self._stop_price,
                    current_price=bar_data['close'],
                    trailing_active=self._trailing_active
                )

    def _process_inclusion(self, new_bar: dict) -> None:
        """包含处理."""
        if not self._k_lines:
            self._k_lines.append(new_bar)
            return

        last = self._k_lines[-1]

        # 检查包含关系
        in_last = new_bar['high'] <= last['high'] and new_bar['low'] >= last['low']
        in_new = last['high'] <= new_bar['high'] and last['low'] >= new_bar['low']

        if in_last or in_new:
            # 存在包含关系，进行合并
            if self._inclusion_dir == 0:
                self._inclusion_dir = 1  # 默认向上

            merged = last.copy()
            merged['datetime'] = new_bar['datetime']
            merged['diff'] = new_bar['diff']
            merged['atr'] = new_bar['atr']
            merged['diff_15m'] = new_bar['diff_15m']
            merged['dea_15m'] = new_bar['dea_15m']

            if self._inclusion_dir == 1:  # 向上包含
                merged['high'] = max(last['high'], new_bar['high'])
                merged['low'] = max(last['low'], new_bar['low'])
            else:  # 向下包含
                merged['high'] = min(last['high'], new_bar['high'])
                merged['low'] = min(last['low'], new_bar['low'])

            # Debug: 记录包含处理（仅实盘模式）
            if self._debugger and self.trading:
                self._debugger.log_inclusion(
                    before_high=last['high'],
                    before_low=last['low'],
                    after_high=merged['high'],
                    after_low=merged['low'],
                    direction=self._inclusion_dir
                )

            self._k_lines[-1] = merged
        else:
            # 无包含关系，更新方向并添加
            if new_bar['high'] > last['high'] and new_bar['low'] > last['low']:
                self._inclusion_dir = 1  # 向上
            elif new_bar['high'] < last['high'] and new_bar['low'] < last['low']:
                self._inclusion_dir = -1  # 向下

            self._k_lines.append(new_bar)

    def _process_bi(self) -> Optional[dict]:
        """严格笔处理."""
        if len(self._k_lines) < 3:
            return None

        curr = self._k_lines[-1]
        mid = self._k_lines[-2]
        left = self._k_lines[-3]

        is_top = mid['high'] > left['high'] and mid['high'] > curr['high']
        is_bot = mid['low'] < left['low'] and mid['low'] < curr['low']

        cand = None
        if is_top:
            cand = {
                'type': 'top',
                'price': mid['high'],
                'idx': len(self._k_lines) - 2,
                'data': mid
            }
        elif is_bot:
            cand = {
                'type': 'bottom',
                'price': mid['low'],
                'idx': len(self._k_lines) - 2,
                'data': mid
            }

        if not cand:
            return None

        if not self._bi_points:
            self._bi_points.append(cand)
            return None

        last = self._bi_points[-1]

        if last['type'] == cand['type']:
            # 同向延伸
            if last['type'] == 'top' and cand['price'] > last['price']:
                self._bi_points[-1] = cand
            elif last['type'] == 'bottom' and cand['price'] < last['price']:
                self._bi_points[-1] = cand
            return None
        else:
            # S6/Iter4: 笔幅度过滤
            if self.bi_amp_filter and self.atr > 0:
                amp = abs(cand['price'] - last['price'])
                if amp < self.bi_amp_min_atr * self.atr:
                    return None

            # 异向成笔（严格笔要求间隔 >= min_bi_gap）
            if cand['idx'] - last['idx'] >= self.min_bi_gap:
                self._bi_points.append(cand)
                # Debug: 记录新笔（仅实盘模式）
                if self._debugger and self.trading:
                    self._debugger.log_bi(
                        cand,
                        bi_idx=len(self._bi_points),
                        k_lines_count=len(self._k_lines)
                    )
                return cand
            return None

    def _update_pivots(self) -> None:
        """中枢识别：基于最近4个笔端点（即3个笔）的重叠区."""
        if len(self._bi_points) < 4:
            return

        # 取最近4个点 b0->b1->b2->b3
        b0 = self._bi_points[-4]
        b1 = self._bi_points[-3]
        b2 = self._bi_points[-2]
        b3 = self._bi_points[-1]

        # 计算每一笔的价格区间 (Range)
        r1 = (min(b0['price'], b1['price']), max(b0['price'], b1['price']))
        r2 = (min(b1['price'], b2['price']), max(b1['price'], b2['price']))
        r3 = (min(b2['price'], b3['price']), max(b2['price'], b3['price']))

        # 计算三笔重叠区域 (Intersection)
        zg = min(r1[1], r2[1], r3[1])  # 重叠区高点
        zd = max(r1[0], r2[0], r3[0])  # 重叠区低点

        if zg > zd:  # 存在有效重叠，确认为中枢
            new_pivot = {
                'zg': zg,
                'zd': zd,
                'start_bi_idx': len(self._bi_points) - 4,
                'end_bi_idx': len(self._bi_points) - 1
            }
            self._pivots.append(new_pivot)
            # Debug: 记录新中枢（仅实盘模式）
            if self._debugger and self.trading:
                self._debugger.log_pivot(
                    new_pivot,
                    pivot_idx=len(self._pivots),
                    status="confirmed"
                )

    def _check_signal(self, curr_bar: dict, new_bi: dict) -> None:
        """检查交易信号."""
        # 1. 尝试更新中枢
        self._update_pivots()

        if len(self._bi_points) < 5:
            return

        # 获取笔端点
        p_now = self._bi_points[-1]   # 当前回踩点
        p_last = self._bi_points[-2]  # 前一极值点（离开段端点）
        p_prev = self._bi_points[-3]  # 再前一点（背驰比较用）

        is_bull = self._prev_diff_15m > self._prev_dea_15m
        is_bear = self._prev_diff_15m < self._prev_dea_15m

        # S6/Iter3: 趋势过滤 — 要求 5m MACD 方向与信号一致
        if self.trend_filter:
            if self.diff_5m <= 0:
                is_bull = False
            if self.diff_5m >= 0:
                is_bear = False

        # S6/Iter4: MACD 一致性 — 要求最近 N 根 MACD 方向连续一致
        if self.macd_consistency > 0 and len(self._recent_diffs) >= self.macd_consistency:
            recent = self._recent_diffs[-self.macd_consistency:]
            if is_bull and not all(d > 0 for d in recent):
                is_bull = False
            if is_bear and not all(d < 0 for d in recent):
                is_bear = False

        sig = None
        sig_type = None
        stop_base = 0.0
        trigger_price = 0.0
        last_pivot = self._pivots[-1] if self._pivots else None

        # --------------------------------------------------------
        # 核心逻辑：Pivot 3B/3S
        # --------------------------------------------------------
        if last_pivot:

            # --- 3B 买点 ---
            # 场景：向上离开中枢 + 回踩不破中枢高点 (ZG)
            if p_now['type'] == 'bottom':
                # 1. 回踩点必须在中枢之上
                if p_now['price'] > last_pivot['zg']:
                    # 2. 离开段必须也曾高于中枢
                    if p_last['price'] > last_pivot['zg']:
                        # 3. 中枢必须是"最近"的
                        if last_pivot['end_bi_idx'] >= len(self._bi_points) - self.pivot_valid_range:
                            if is_bull:
                                sig = 'Buy'
                                sig_type = '3B'
                                trigger_price = p_now['data']['high']
                                stop_base = p_now['price']

            # --- 3S 卖点 ---
            # 场景：向下离开中枢 + 回抽不破中枢低点 (ZD)
            elif p_now['type'] == 'top':
                if p_now['price'] < last_pivot['zd']:
                    if p_last['price'] < last_pivot['zd']:
                        if last_pivot['end_bi_idx'] >= len(self._bi_points) - self.pivot_valid_range:
                            if is_bear:
                                # S6/Iter3: 禁用 3S 做空，改为平多退出
                                if self.disable_3s_short:
                                    if self._position == 1:
                                        exit_px = p_now['data']['low']
                                        pnl = exit_px - self._entry_price
                                        if self.trading:
                                            self.sell(exit_px, abs(self.pos))
                                            self.write_log(
                                                f"3S平多退出: price={exit_px:.0f}, pnl={pnl:.0f}"
                                            )
                                        self._position = 0
                                        self._trailing_active = False
                                        self.signal = "3S平多"
                                else:
                                    sig = 'Sell'
                                    sig_type = '3S'
                                    trigger_price = p_now['data']['low']
                                    stop_base = p_now['price']

        # --------------------------------------------------------
        # 辅助逻辑：保留 2B/2S（作为趋势延续补充）
        # --------------------------------------------------------
        if not sig:
            if p_now['type'] == 'bottom':
                div = p_now['data']['diff'] > p_prev['data']['diff']
                if p_now['price'] > p_prev['price'] and div and is_bull:
                    sig = 'Buy'  # 2B
                    sig_type = '2B'
                    trigger_price = p_now['data']['high']
                    stop_base = p_now['price']

            elif p_now['type'] == 'top':
                div = p_now['data']['diff'] < p_prev['data']['diff']
                if p_now['price'] < p_prev['price'] and div and is_bear:
                    sig = 'Sell'  # 2S
                    sig_type = '2S'
                    trigger_price = p_now['data']['low']
                    stop_base = p_now['price']

        # S6/Iter6: 15m MACD 幅度上限过滤（过滤趋势过热末端的信号）
        if sig and self.macd15_mag_cap_atr > 0 and self.atr > 0:
            mag = abs(self._prev_diff_15m) / self.atr
            if mag > self.macd15_mag_cap_atr:
                sig = None  # 过滤掉

        # --------------------------------------------------------
        # 信号过滤与设置
        # --------------------------------------------------------
        if sig and self.atr > 0:
            # 入场过滤：触发价与止损距离不超过 N 倍 ATR
            distance = abs(trigger_price - stop_base)
            if distance < self.atr_entry_filter * self.atr:
                # 使用已确定的信号类型
                self._signal_type = sig_type or ("2B" if sig == 'Buy' else "2S")
                reason = f"{self._signal_type}: 信号确认"

                self._pending_signal = {
                    'type': sig,
                    'trigger_price': trigger_price,
                    'stop_base': stop_base
                }
                self.signal = f"待触发{sig}"

                # Debug: 记录信号（仅实盘模式）
                if self._debugger and self.trading:
                    self._debugger.log_signal(
                        signal_type=self._signal_type,
                        direction=sig,
                        trigger_price=trigger_price,
                        stop_price=stop_base,
                        atr=self.atr,
                        reason=reason
                    )

                if self.debug:
                    self.write_log(f"[DEBUG] 信号生成: {sig}, trigger={trigger_price:.0f}, stop={stop_base:.0f}")

    def _check_entry_1m(self, bar: dict) -> None:
        """1分钟级别检查入场（待触发信号）."""
        signal = self._pending_signal
        if not signal:
            return

        if signal['type'] == 'Buy':
            # 检查是否已经破止损（信号失效）
            if bar['low'] < signal['stop_base']:
                self._pending_signal = None
                self.signal = ""
                return
            # 检查是否触发入场
            if bar['high'] > signal['trigger_price']:
                fill_price = max(signal['trigger_price'], bar['open'])
                if fill_price > bar['high']:
                    fill_price = bar['close']
                self._open_position(1, fill_price, signal['stop_base'])

        elif signal['type'] == 'Sell':
            # 检查是否已经破止损（信号失效）
            if bar['high'] > signal['stop_base']:
                self._pending_signal = None
                self.signal = ""
                return
            # 检查是否触发入场
            if bar['low'] < signal['trigger_price']:
                fill_price = min(signal['trigger_price'], bar['open'])
                if fill_price < bar['low']:
                    fill_price = bar['close']
                self._open_position(-1, fill_price, signal['stop_base'])

    def _open_position(self, direction: int, price: float, stop_base: float) -> None:
        """开仓."""
        if direction == 1:
            self.buy(price, self.fixed_volume)
            self._stop_price = stop_base - 1
            self.signal = "3B/2B买入"
            self.write_log(f"开多: price={price:.0f}, stop={self._stop_price:.0f}")
            action = "OPEN_LONG"
        else:
            self.short(price, self.fixed_volume)
            self._stop_price = stop_base + 1
            self.signal = "3S/2S卖出"
            self.write_log(f"开空: price={price:.0f}, stop={self._stop_price:.0f}")
            action = "OPEN_SHORT"

        self._position = direction
        self._entry_price = price
        self._trailing_active = False
        self._pending_signal = None

        # Debug: 记录开仓（仅实盘模式）
        if self._debugger and self.trading:
            self._debugger.log_trade(
                action=action,
                price=price,
                volume=self.fixed_volume,
                position=direction,
                pnl=0,
                signal_type=self._signal_type
            )

    def _check_stop_loss_1m(self, bar: dict) -> bool:
        """1分钟级别检查止损."""
        sl_hit = False
        exit_price = 0.0

        if self._position == 1:
            if bar['low'] <= self._stop_price:
                sl_hit = True
                exit_price = bar['open'] if bar['open'] < self._stop_price else self._stop_price
        elif self._position == -1:
            if bar['high'] >= self._stop_price:
                sl_hit = True
                exit_price = bar['open'] if bar['open'] > self._stop_price else self._stop_price

        if sl_hit:
            # 计算盈亏
            if self._position == 1:
                pnl = exit_price - self._entry_price
                self.sell(exit_price, abs(self.pos))
                self.write_log(f"多头止损: price={exit_price:.0f}, pnl={pnl:.0f}")
                action = "CLOSE_LONG"
            else:
                pnl = self._entry_price - exit_price
                self.cover(exit_price, abs(self.pos))
                self.write_log(f"空头止损: price={exit_price:.0f}, pnl={pnl:.0f}")
                action = "CLOSE_SHORT"

            # Debug: 记录平仓（仅实盘模式）
            if self._debugger and self.trading:
                self._debugger.log_trade(
                    action=action,
                    price=exit_price,
                    volume=self.fixed_volume,
                    position=0,
                    pnl=pnl,
                    signal_type=self._signal_type
                )

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
