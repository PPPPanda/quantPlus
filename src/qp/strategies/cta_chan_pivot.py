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

from qp.datafeed.normalizer import (
    PALM_OIL_SESSIONS, compute_window_end, get_session_key, _get_session_for_time,
)
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
    atr_trailing_mult: float = 3.0   # ATR 移动止损倍数
    atr_activate_mult: float = 2.5   # 激活移动止损的浮盈 ATR 倍数
    atr_entry_filter: float = 2.0    # 入场过滤：触发价与止损距离不超过 N 倍 ATR

    # 笔构建参数
    min_bi_gap: int = 4              # 严格笔端点最小间隔（包含处理后的K线数）

    # 中枢参数
    pivot_valid_range: int = 6       # 中枢有效范围（笔端点数）

    # 合约参数
    fixed_volume: int = 1            # 固定手数

    # =========================================================================
    # iter14 基线参数（已验证：TOTAL=13667.9 pts, 12/13合约通过）
    # =========================================================================

    # B02/R2: 分层连亏断路器
    # - L1（轻度冷却）：连亏 cooldown_losses 次后，暂停 cooldown_bars 根 5m bar
    # - L2（硬断路器）：连亏 circuit_breaker_losses 次后，暂停 circuit_breaker_bars 根 5m bar
    cooldown_losses: int = 2         # L1 连亏阈值（0=禁用）
    cooldown_bars: int = 20          # L1 冷却期（5m bar 数）
    circuit_breaker_losses: int = 7  # L2 连亏阈值（iter14 基线：7）
    circuit_breaker_bars: int = 70   # L2 冷却期（iter14 基线：70，约 350 分钟）

    # R3: 两段式出场 — 锁盈参数（iter14 禁用）
    lock_profit_atr: float = 0.0       # 浮盈≥1R时锁盈: 止损抬到 entry + N*ATR（0=禁用）

    # S5: 最小持仓保护
    # - 开仓后前 N 根 5m bar 止损距离加宽（×2），避免噪音止损
    min_hold_bars: int = 2             # 保护期长度（0=禁用）

    # S4: 3B回踩深度过滤
    # - 3B 信号要求回踩点与中枢高点 ZG 的距离 < N×ATR
    # - 过滤远离中枢的虚假回踩信号
    max_pullback_atr: float = 3.2      # iter14 基线：3.2

    # S7: 结构 trailing — 激活后用笔低点作为止损参考
    use_bi_trailing: bool = True       # 优先用最近 bottom - buffer 作为 trailing stop

    # B03: 止损 buffer
    # - 止损价 = 计算止损 - buffer
    # - buffer = max(pricetick, atr × stop_buffer_atr_pct)
    stop_buffer_atr_pct: float = 0.02  # ATR 比例（2%）

    # R1: 入场去重（价格区域+时间窗口）
    max_pivot_entries: int = 2         # 同一中枢最多入场次数（0=禁用）
    pivot_reentry_atr: float = 0.6     # 超限后需突破中枢上沿 + N×ATR 才放行
    dedup_bars: int = 0                # 时间去重窗口（0=禁用）
    dedup_atr_mult: float = 1.5        # 价格去重距离（ATR 倍数）

    # B10: 背驰模式参数
    # - div_mode: 0=仅 diff, 1=OR(diff 或面积), 2=AND, 3=仅面积
    # - div_threshold: 面积背驰阈值（当前笔面积 < 前笔 × threshold）
    div_mode: int = 1                  # OR 模式
    div_threshold: float = 0.39        # iter14 基线：0.39

    # =========================================================================
    # iter14 后实验性功能（已验证无效或有副作用，保留代码但禁用）
    # =========================================================================

    # S8: 走势段背驰（iter8 验证无效）
    seg_enabled: bool = False          # False=禁用
    seg_div_ratio: float = 0.82
    seg_div_ratio_sell: float = 0.78
    seg_min_bi: int = 3
    seg_price_tol: float = 0.003

    # S9: histogram 确认门（未验证）
    hist_gate: int = 0                 # 0=禁用

    # S26/S26b: 跳空安全网（iter17 添加，效果有限）
    gap_extreme_atr: float = 0.0       # 0=禁用（原值 1.5）
    gap_cooldown_bars: int = 6
    gap_tier1_atr: float = 10.0
    gap_tier2_atr: float = 30.0

    # S27: 跳空重置包含方向（iter20 验证灾难性失败）
    gap_reset_inclusion: bool = False  # 必须 False

    # B28: Bridge Bar（iter21 添加，+214pts 但波动大）
    bridge_bar_enabled: bool = False   # 禁用回到 iter14 基线
    bridge_gap_threshold: float = 1.5

    # 调试
    debug: bool = False
    debug_enabled: bool = True      # 是否启用Debug记录
    debug_log_console: bool = True  # 是否输出到控制台

    parameters: list[str] = [
        "macd_fast", "macd_slow", "macd_signal",
        "atr_window", "atr_trailing_mult", "atr_activate_mult", "atr_entry_filter",
        "min_bi_gap", "pivot_valid_range", "fixed_volume",
        "cooldown_losses", "cooldown_bars",
        "circuit_breaker_losses", "circuit_breaker_bars",
        "lock_profit_atr",
        "min_hold_bars",
        "max_pullback_atr",
        "use_bi_trailing",
        "stop_buffer_atr_pct",
        "max_pivot_entries", "pivot_reentry_atr",
        "dedup_bars", "dedup_atr_mult",
        "div_mode", "div_threshold",
        "seg_enabled", "seg_div_ratio", "seg_div_ratio_sell", "seg_min_bi", "seg_price_tol",
        "hist_gate",
        "gap_extreme_atr", "gap_cooldown_bars", "gap_tier1_atr", "gap_tier2_atr", "gap_reset_inclusion",
        "bridge_bar_enabled", "bridge_gap_threshold",
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

        # K 线合成缓存（session-aware 增量方式）
        self._sessions = PALM_OIL_SESSIONS
        self._window_bar_5m: Optional[dict] = None
        self._last_window_end_5m: Optional[datetime] = None
        self._last_session_key_5m: Optional[tuple] = None
        self._window_bar_15m: Optional[dict] = None
        self._last_window_end_15m: Optional[datetime] = None
        self._last_session_key_15m: Optional[tuple] = None

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

        # ATR 增量计算缓存
        self._tr_values: deque = deque(maxlen=14)
        self._prev_close_5m: float = 0.0
        self.atr: float = 0.0

        # 笔端点列表（严格笔）
        self._bi_points: list[dict] = []

        # 中枢列表
        self._pivots: list[dict] = []

        # R2: 活跃中枢状态机
        # state: None, 'forming', 'active', 'left_up', 'left_down'
        self._active_pivot: Optional[dict] = None

        # 待触发信号
        self._pending_signal: Optional[dict] = None

        # S3: 3B两步确认缓冲
        # 第一步：回踩bottom形成且>ZG → 存入_pending_3b_confirm
        # 第二步：下一笔向上确认(high > 回踩bar high) → 转为_pending_signal
        self._pending_3b_confirm: Optional[dict] = None

        # 交易状态
        self._position: int = 0
        self._entry_price: float = 0.0
        self._stop_price: float = 0.0
        self._initial_stop: float = 0.0   # R3: 记录初始止损价（计算 1R 用）
        self._trailing_active: bool = False
        self._bars_since_entry: int = 0  # S5: 开仓后经过的5m bar数

        # B02: 连亏冷却状态
        self._consecutive_losses: int = 0
        self._cooldown_remaining: int = 0  # 剩余冷却 5m bar 数

        # S26: 极端跳空安全网状态
        self._gap_cooldown_remaining: int = 0  # 极端跳空后剩余冷却 5m bar 数

        # 5m bar 计数
        self._bar_5m_count: int = 0

        # R1: 入场去重追踪
        self._recent_entries: list[dict] = []     # [{price, bar_5m_count}]

        # B10: MACD 面积背驰追踪
        self._current_bi_macd_area: float = 0.0  # 当前笔段累积 |histogram|
        self._bi_macd_areas: list[float] = []     # 每笔完成时的面积记录

        # S8: 走势段(segment)缓存
        self._segments: list[dict] = []           # 已完成的走势段列表
        self._seg_last_bi_count: int = 0          # 上次构建segments时的笔数

        # B28: Bridge Bar 状态
        self._last_5m_bar: Optional[dict] = None  # 上一根5分钟K线（用于检测跳空）

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
        # 非交易时段 bar 直接跳过（兼容未归一化的原始数据）
        dt = bar.datetime
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        if _get_session_for_time(dt.time(), self._sessions) is None:
            return

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

        # 2. 信号管理：检查待触发信号（入场/平仓，仅实盘模式）
        if self.trading and self._pending_signal:
            # Buy 信号仅在无仓时触发；CloseLong 信号仅在持多仓时触发
            sig_type = self._pending_signal.get('type', '')
            if (sig_type == 'Buy' and self._position == 0) or \
               (sig_type == 'CloseLong' and self._position == 1) or \
               (sig_type == 'Sell' and self._position == 0):
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

    def _update_5m_bar(self, bar: dict) -> Optional[dict]:
        """Session-aware 5 分钟 K 线合成（窗口切换时 emit 上一个窗口）."""
        dt = bar['datetime']
        window_end = compute_window_end(dt, self._sessions, 5)
        if window_end is None:
            return None

        session_key = get_session_key(dt, self._sessions)
        result = None

        # session 或窗口变了 → emit 上一个已累积的 bar
        if self._window_bar_5m is not None:
            session_changed = (self._last_session_key_5m != session_key)
            window_changed = (self._last_window_end_5m != window_end)
            if session_changed or window_changed:
                result = self._window_bar_5m.copy()
                self._window_bar_5m = None

        self._last_session_key_5m = session_key

        # 初始化或更新当前窗口
        if self._window_bar_5m is None:
            self._window_bar_5m = {
                'datetime': window_end,
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume'],
            }
            self._last_window_end_5m = window_end
        else:
            wb = self._window_bar_5m
            wb['high'] = max(wb['high'], bar['high'])
            wb['low'] = min(wb['low'], bar['low'])
            wb['close'] = bar['close']
            wb['volume'] += bar['volume']

        return result

    def _update_15m_bar(self, bar: dict) -> Optional[dict]:
        """Session-aware 15 分钟 K 线合成（窗口切换时 emit）."""
        dt = bar['datetime']
        window_end = compute_window_end(dt, self._sessions, 15)
        if window_end is None:
            return None

        session_key = get_session_key(dt, self._sessions)
        result = None

        if self._window_bar_15m is not None:
            session_changed = (self._last_session_key_15m != session_key)
            window_changed = (self._last_window_end_15m != window_end)
            if session_changed or window_changed:
                result = self._window_bar_15m.copy()
                self._window_bar_15m = None

        self._last_session_key_15m = session_key

        if self._window_bar_15m is None:
            self._window_bar_15m = {
                'datetime': window_end,
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume'],
            }
            self._last_window_end_15m = window_end
        else:
            wb = self._window_bar_15m
            wb['high'] = max(wb['high'], bar['high'])
            wb['low'] = min(wb['low'], bar['low'])
            wb['close'] = bar['close']
            wb['volume'] += bar['volume']

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
        self._bar_5m_count += 1

        # S5: 计数开仓后的bar数
        if self._position != 0:
            self._bars_since_entry += 1

        # R1: 清理过老的入场记录
        if self._recent_entries and self.dedup_bars > 0:
            cutoff = self._bar_5m_count - self.dedup_bars * 2  # 保留 2x 窗口
            while self._recent_entries and self._recent_entries[0]['bar_5m_count'] < cutoff:
                self._recent_entries.pop(0)

        # B02: 冷却计数器递减
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # S26/S26b: 跳空检测 + 分级冷却
        if self._gap_cooldown_remaining > 0:
            self._gap_cooldown_remaining -= 1
        # 检测跳空（使用上一根 5m bar 的收盘价）
        if self._prev_close_5m > 0 and self.atr > 0:
            gap = abs(bar['open'] - self._prev_close_5m)
            gap_atr = gap / self.atr
            # S26b 分级冷却：中等跳空最危险
            if gap_atr >= self.gap_extreme_atr:
                if gap_atr < self.gap_tier1_atr:
                    # 中等跳空(1.5-10x)：最危险，长冷却
                    cooldown = self.gap_cooldown_bars  # 默认6
                elif gap_atr < self.gap_tier2_atr:
                    # 大跳空(10-30x)：短冷却
                    cooldown = max(3, self.gap_cooldown_bars // 2)
                else:
                    # 极大跳空(>30x)：反而安全，无冷却
                    cooldown = 0
                if cooldown > 0:
                    self._gap_cooldown_remaining = cooldown
                    self.write_log(f"[S26c] 跳空: gap={gap:.0f} ({gap_atr:.1f}x ATR), 冷却{cooldown}根bar")
                    # S27: 跳空后重置包含方向（避免方向污染）
                    if self.gap_reset_inclusion:
                        self._inclusion_dir = 0
                        self.write_log(f"[S27] 重置包含方向")

        # 更新指标
        self._update_macd_5m(bar['close'])
        self._update_atr(bar)

        # B10: 累积当前笔段的 MACD histogram 面积
        histogram = self.diff_5m - self.dea_5m
        self._current_bi_macd_area += abs(histogram)

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

        # B28: Bridge Bar 插入（把跳空显式建模为最低级别走势连接）
        if self.bridge_bar_enabled and self._last_5m_bar is not None and self.atr > 0:
            last_close = self._last_5m_bar['close']
            gap = bar['open'] - last_close
            gap_abs = abs(gap)
            gap_atr = gap_abs / self.atr
            
            if gap_atr >= self.bridge_gap_threshold:
                # 构建 bridge bar
                bridge_data = {
                    'datetime': bar['datetime'],  # 使用当前时间
                    'high': max(last_close, bar['open']),
                    'low': min(last_close, bar['open']),
                    'close': bar['open'],
                    'diff': self.diff_5m,
                    'atr': self.atr,
                    'diff_15m': self._prev_diff_15m,
                    'dea_15m': self._prev_dea_15m,
                    'is_bridge': True,
                    'bridge_gap_atr': gap_atr,
                    'bridge_direction': 'up' if gap > 0 else 'down',
                }
                # 先处理 bridge bar
                self._process_inclusion(bridge_data)
                if self.debug_log_console:
                    self.write_log(f"[B28] Bridge Bar: gap={gap:.0f} ({gap_atr:.1f}x ATR), dir={bridge_data['bridge_direction']}")

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
            'is_bridge': False,
        }

        # 1. 包含处理
        self._process_inclusion(bar_data)

        # 2. 严格笔处理
        new_bi = self._process_bi()
        
        # 保存当前bar用于下次跳空检测
        self._last_5m_bar = bar

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
        """包含处理.

        R1修正: dir==0时不强制默认向上，而是跳过合并直接append，
        等第一次出现非包含K线时从价格关系推导方向。
        """
        if not self._k_lines:
            self._k_lines.append(new_bar)
            return

        last = self._k_lines[-1]

        # 检查包含关系
        in_last = new_bar['high'] <= last['high'] and new_bar['low'] >= last['low']
        in_new = last['high'] <= new_bar['high'] and last['low'] >= new_bar['low']

        if in_last or in_new:
            # 存在包含关系
            if self._inclusion_dir == 0:
                # R1: 方向未定时不合并，直接append（等待方向确定）
                self._k_lines.append(new_bar)
                return

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

    def _count_real_bars(self, start_idx: int, end_idx: int) -> int:
        """B28: 计算两个索引之间的真实bar数量（排除bridge bar）."""
        count = 0
        for i in range(start_idx, min(end_idx + 1, len(self._k_lines))):
            if not self._k_lines[i].get('is_bridge', False):
                count += 1
        return count

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
            # 异向成笔（严格笔要求间隔 >= min_bi_gap）
            # B28: 使用真实bar计数（排除bridge bar）而非简单index距离
            if self.bridge_bar_enabled:
                real_gap = self._count_real_bars(last['idx'], cand['idx']) - 1
            else:
                real_gap = cand['idx'] - last['idx']
            
            if real_gap >= self.min_bi_gap:
                self._bi_points.append(cand)
                # B10: 保存当前笔段面积并重置
                self._bi_macd_areas.append(self._current_bi_macd_area)
                self._current_bi_macd_area = 0.0
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
        """R2: 中枢状态机.

        状态流转:
        - None → forming: 检测到3笔重叠
        - forming → active: 第4笔仍在中枢范围内（延伸）
        - active → active: 后续笔仍在中枢范围内
        - forming/active → left_up: 某笔 low > ZG（向上离开）
        - forming/active → left_down: 某笔 high < ZD（向下离开）
        - left_* → None: 离开后又形成新中枢（旧中枢归档）

        信号只在 left_up/left_down 状态下生成。
        """
        if len(self._bi_points) < 4:
            return

        latest_bi = self._bi_points[-1]
        ap = self._active_pivot

        # --- 如果有活跃中枢，检查是否延伸或离开 ---
        if ap is not None:
            zg = ap['zg']
            zd = ap['zd']
            state = ap['state']

            if state in ('forming', 'active'):
                # 检查最新笔是否离开中枢
                bi_low = min(self._bi_points[-1]['price'], self._bi_points[-2]['price'])
                bi_high = max(self._bi_points[-1]['price'], self._bi_points[-2]['price'])

                if bi_low > zg:
                    # 向上离开：整笔都在中枢上方
                    ap['state'] = 'left_up'
                    ap['leave_bi_idx'] = len(self._bi_points) - 1
                    ap['leave_price'] = latest_bi['price']
                    if self._debugger and self.trading:
                        self._debugger.log_pivot(ap, pivot_idx=len(self._pivots), status="left_up")
                    return
                elif bi_high < zd:
                    # 向下离开：整笔都在中枢下方
                    ap['state'] = 'left_down'
                    ap['leave_bi_idx'] = len(self._bi_points) - 1
                    ap['leave_price'] = latest_bi['price']
                    if self._debugger and self.trading:
                        self._debugger.log_pivot(ap, pivot_idx=len(self._pivots), status="left_down")
                    return
                else:
                    # 仍在中枢范围内，延伸
                    ap['state'] = 'active'
                    ap['end_bi_idx'] = len(self._bi_points) - 1
                    # 入场计数不变
                    return

            elif state in ('left_up', 'left_down'):
                # 已离开，检查是否形成新中枢（归档旧的）
                # 继续往下尝试检测新中枢
                pass

        # --- 尝试检测新中枢 ---
        b0 = self._bi_points[-4]
        b1 = self._bi_points[-3]
        b2 = self._bi_points[-2]
        b3 = self._bi_points[-1]

        r1 = (min(b0['price'], b1['price']), max(b0['price'], b1['price']))
        r2 = (min(b1['price'], b2['price']), max(b1['price'], b2['price']))
        r3 = (min(b2['price'], b3['price']), max(b2['price'], b3['price']))

        zg = min(r1[1], r2[1], r3[1])
        zd = max(r1[0], r2[0], r3[0])

        if zg > zd:
            # 归档旧中枢
            if ap is not None:
                self._pivots.append(ap)

            new_pivot = {
                'zg': zg,
                'zd': zd,
                'start_bi_idx': len(self._bi_points) - 4,
                'end_bi_idx': len(self._bi_points) - 1,
                'state': 'forming',
                'leave_bi_idx': None,
                'leave_price': None,
                'entry_count': 0,  # R6: 同中枢入场计数
            }
            self._active_pivot = new_pivot

            if self._debugger and self.trading:
                self._debugger.log_pivot(
                    new_pivot,
                    pivot_idx=len(self._pivots) + 1,
                    status="forming"
                )

    def _has_area_divergence(self) -> Optional[bool]:
        """
        B10: 检查面积背驰.

        比较当前笔段面积与前同向笔段（间隔2笔）面积。
        返回 True=有背驰, False=无背驰, None=数据不足无法判断。
        """
        n = len(self._bi_macd_areas)
        if n < 3:
            return None
        curr = self._bi_macd_areas[-1]
        prev_same = self._bi_macd_areas[-3]  # 同向笔段（交替上下，间隔2）
        if prev_same <= 0:
            return None
        return curr < prev_same * self.div_threshold

    def _eval_div_condition(self, diff_ok: bool) -> bool:
        """
        B10: 根据 div_mode 评估 2B/2S 背驰条件.

        mode 0: 仅 diff 比较（baseline）
        mode 1: diff OR 面积背驰
        mode 2: diff AND 面积背驰（数据不足时 fallback 到 diff）
        mode 3: 仅面积背驰
        """
        if self.div_mode == 0:
            return diff_ok

        div = self._has_area_divergence()

        if self.div_mode == 1:
            return diff_ok or (div is True)
        elif self.div_mode == 2:
            return (diff_ok and div) if div is not None else diff_ok
        elif self.div_mode == 3:
            return div is True
        else:
            return diff_ok

    # ------------------------------------------------------------------
    # S8: 走势段(Segment)构建与背驰判定
    # ------------------------------------------------------------------

    def _build_segments(self) -> list[dict]:
        """
        S8: 从 bi_points + bi_macd_areas 构建走势段.

        段定义（简化工程版，非严格缠论线段）：
        - down 段：从某个 top 开始，回抽 top 不创新高（top 逐步走低）
        - up 段：从某个 bottom 开始，回抽 bottom 不创新低
        - 段切换：出现更高的 top（down→up）或更低的 bottom（up→down）
        """
        bp = self._bi_points
        areas = self._bi_macd_areas
        n_bp = len(bp)

        if n_bp < 5:
            return []

        segments: list[dict] = []

        def bi_dir(i: int) -> str:
            """笔 i 连接 bp[i] -> bp[i+1] 的方向."""
            return 'up' if bp[i + 1]['price'] > bp[i]['price'] else 'down'

        def seg_area(seg_dir: str, bi_start: int, bi_end: int) -> float:
            """计算段内同向笔的 MACD 面积之和."""
            total = 0.0
            for k in range(bi_start, min(bi_end, len(areas))):
                if bi_dir(k) == seg_dir:
                    total += abs(areas[k]) if k < len(areas) else 0.0
            return total

        # 初始化当前段
        cur_dir = bi_dir(0)
        cur_start = 0

        # 追踪 **前一个** 同类端点价格（不是全局极值）
        # down 段：当新 top > prev_top 时切换
        # up 段：当新 bottom < prev_bottom 时切换
        prev_top_price = None
        prev_bottom_price = None

        for j in range(min(2, n_bp)):
            if bp[j]['type'] == 'top':
                prev_top_price = bp[j]['price']
            elif bp[j]['type'] == 'bottom':
                prev_bottom_price = bp[j]['price']

        def _close_segment(end_i: int) -> None:
            bi_end = end_i
            if bi_end > cur_start:
                segments.append({
                    'dir': cur_dir,
                    'start_i': cur_start,
                    'end_i': end_i,
                    'start_price': bp[cur_start]['price'],
                    'end_price': bp[end_i]['price'],
                    'bi_start': cur_start,
                    'bi_end': bi_end,
                    'macd_area': seg_area(cur_dir, cur_start, bi_end),
                    'bi_count': bi_end - cur_start,
                })

        for i in range(2, n_bp):
            p = bp[i]

            if p['type'] == 'top':
                if cur_dir == 'down' and prev_top_price is not None:
                    if p['price'] > prev_top_price:
                        # top 创新高 → down 段结束
                        end_i = i - 1
                        _close_segment(end_i)
                        cur_dir = 'up'
                        cur_start = end_i
                        prev_bottom_price = bp[end_i]['price'] if bp[end_i]['type'] == 'bottom' else prev_bottom_price
                # 始终更新为当前 top 价格（跟踪前一个，而非全局最大）
                prev_top_price = p['price']

            elif p['type'] == 'bottom':
                if cur_dir == 'up' and prev_bottom_price is not None:
                    if p['price'] < prev_bottom_price:
                        # bottom 创新低 → up 段结束
                        end_i = i - 1
                        _close_segment(end_i)
                        cur_dir = 'down'
                        cur_start = end_i
                        prev_top_price = bp[end_i]['price'] if bp[end_i]['type'] == 'top' else prev_top_price
                prev_bottom_price = p['price']

        # 关闭最后一个未完成段
        end_i = n_bp - 1
        if end_i > cur_start:
            segments.append({
                'dir': cur_dir,
                'start_i': cur_start,
                'end_i': end_i,
                'start_price': bp[cur_start]['price'],
                'end_price': bp[end_i]['price'],
                'bi_start': cur_start,
                'bi_end': end_i,
                'macd_area': seg_area(cur_dir, cur_start, end_i),
                'bi_count': end_i - cur_start,
            })

        return segments

    def _get_segments(self) -> list[dict]:
        """获取走势段（带缓存，笔数变化时重建）."""
        n = len(self._bi_points)
        if n != self._seg_last_bi_count:
            self._segments = self._build_segments()
            self._seg_last_bi_count = n
        return self._segments

    def _check_seg_divergence(self, direction: str) -> bool:
        """
        S8: 检查最近两个同向走势段是否存在背驰.

        direction: 'down' → 检查底背驰(2B买入)
                   'up'   → 检查顶背驰(2S平多)

        返回 True 表示存在背驰（价格创新极值但力度衰减）。
        """
        segs = self._get_segments()
        # 取最近两个同向段
        same_dir = [s for s in segs if s['dir'] == direction and s['bi_count'] >= self.seg_min_bi]
        if len(same_dir) < 2:
            return False

        s1 = same_dir[-2]  # 前一段
        s2 = same_dir[-1]  # 当前段

        # 能量过滤：前一段要"真用力"
        if s1['macd_area'] <= 0:
            return False

        # 价格条件
        tol = self.seg_price_tol
        if direction == 'down':
            # 底背驰：当前段终点 <= 前一段终点 * (1+tol)（创新低/等低）
            price_ok = s2['end_price'] <= s1['end_price'] * (1 + tol)
        else:
            # 顶背驰：当前段终点 >= 前一段终点 * (1-tol)（创新高/等高）
            price_ok = s2['end_price'] >= s1['end_price'] * (1 - tol)

        if not price_ok:
            return False

        # 力度衰减
        ratio_th = self.seg_div_ratio if direction == 'down' else self.seg_div_ratio_sell
        ratio = s2['macd_area'] / (s1['macd_area'] + 1e-9)
        return ratio <= ratio_th

    def _check_signal(self, curr_bar: dict, new_bi: dict) -> None:
        """R2: 检查交易信号（使用中枢状态机）."""
        # 1. 尝试更新中枢
        self._update_pivots()

        if len(self._bi_points) < 5:
            return

        # B02: 冷却期间不生成新信号
        if self._cooldown_remaining > 0:
            return

        # S26: 极端跳空冷却期间不生成新信号
        if self._gap_cooldown_remaining > 0:
            return

        # 获取笔端点
        p_now = self._bi_points[-1]   # 当前回踩点
        p_last = self._bi_points[-2]  # 前一极值点（离开段端点）
        p_prev = self._bi_points[-3]  # 再前一点（背驰比较用）

        is_bull = self._prev_diff_15m > self._prev_dea_15m
        is_bear = self._prev_diff_15m < self._prev_dea_15m

        sig = None
        stop_base = 0.0
        trigger_price = 0.0

        # R2: 使用活跃中枢状态机
        ap = self._active_pivot
        # 同时保留对历史中枢的 fallback（兼容性）
        last_pivot = ap if ap else (self._pivots[-1] if self._pivots else None)

        # --------------------------------------------------------
        # 核心逻辑：Pivot 3B/3S（R2: 状态机驱动）
        # --------------------------------------------------------
        if ap and ap['state'] in ('left_up', 'left_down'):
            # --- 3B 买点 ---
            # R2: 向上离开中枢 + 回踩不破 ZG（状态机确认离开段）
            if ap['state'] == 'left_up' and p_now['type'] == 'bottom':
                # 回踩点在 ZG 之上 = 回踩不破中枢高点
                # S4: 且回踩点不能离ZG太远（确保是真正的"回踩"而非追高）
                pullback_ok = True
                if self.max_pullback_atr > 0 and self.atr > 0:
                    pullback_ok = p_now['price'] < ap['zg'] + self.max_pullback_atr * self.atr
                if p_now['price'] > ap['zg'] and pullback_ok:
                    if is_bull:
                        # R1: 同中枢入场去重 — 检查是否超限
                        if self.max_pivot_entries > 0 and ap['entry_count'] >= self.max_pivot_entries:
                            # 超限后需要更强突破: 触发价 > ZG + pivot_reentry_atr * ATR
                            breakout_threshold = ap['zg'] + self.pivot_reentry_atr * self.atr
                            if p_now['data']['high'] <= breakout_threshold:
                                pass  # 不满足强突破条件，跳过
                            else:
                                sig = 'Buy'
                                trigger_price = p_now['data']['high']
                                stop_base = p_now['price']
                        else:
                            sig = 'Buy'
                            trigger_price = p_now['data']['high']
                            stop_base = p_now['price']

            # --- 3S 卖点 ---
            # R2: 向下离开中枢 + 回抽不破 ZD
            elif ap['state'] == 'left_down' and p_now['type'] == 'top':
                if p_now['price'] < ap['zd']:
                    if is_bear:
                        sig = 'CloseLong'
                        trigger_price = p_now['data']['low']
                        stop_base = p_now['price']

        elif last_pivot:
            # Fallback: 中枢处于 forming/active 时，使用旧逻辑（价格阈值）
            if p_now['type'] == 'bottom':
                if p_now['price'] > last_pivot['zg']:
                    if p_last['price'] > last_pivot['zg']:
                        age = len(self._bi_points) - 1 - last_pivot.get('end_bi_idx', 0)
                        if age <= self.pivot_valid_range:
                            if is_bull:
                                # R1: 同中枢入场去重
                                if self.max_pivot_entries > 0 and last_pivot.get('entry_count', 0) >= self.max_pivot_entries:
                                    breakout_threshold = last_pivot['zg'] + self.pivot_reentry_atr * self.atr
                                    if p_now['data']['high'] <= breakout_threshold:
                                        pass  # 跳过
                                    else:
                                        sig = 'Buy'
                                        trigger_price = p_now['data']['high']
                                        stop_base = p_now['price']
                                else:
                                    sig = 'Buy'
                                    trigger_price = p_now['data']['high']
                                    stop_base = p_now['price']
            elif p_now['type'] == 'top':
                if p_now['price'] < last_pivot['zd']:
                    if p_last['price'] < last_pivot['zd']:
                        age = len(self._bi_points) - 1 - last_pivot.get('end_bi_idx', 0)
                        if age <= self.pivot_valid_range:
                            if is_bear:
                                sig = 'CloseLong'
                                trigger_price = p_now['data']['low']
                                stop_base = p_now['price']

        # --------------------------------------------------------
        # 辅助逻辑：2B/2S（趋势延续）
        # S2: 结构闸门 — 需要活跃中枢且处于 active/forming
        # B10: 支持面积背驰混合模式
        # --------------------------------------------------------
        has_active_structure = (ap is not None and ap['state'] in ('forming', 'active'))
        if not sig and has_active_structure:
            if p_now['type'] == 'bottom':
                # 原始2B：低点抬高 + 动能增强/面积背驰（趋势延续）
                diff_ok = p_now['data']['diff'] > p_prev['data']['diff']
                price_ok = p_now['price'] > p_prev['price']
                cond = self._eval_div_condition(diff_ok)
                if price_ok and cond and is_bull:
                    # S9: histogram 确认门
                    hist_ok = True
                    if self.hist_gate > 0:
                        hist_now = self.diff_5m - self.dea_5m
                        if self.hist_gate == 1:
                            hist_ok = hist_now > 0  # histogram > 0
                        elif self.hist_gate == 2:
                            hist_prev_val = p_prev['data']['diff'] - p_prev['data'].get('dea', self.dea_5m)
                            hist_ok = hist_now > hist_prev_val  # histogram 上拐
                    if hist_ok:
                        sig = 'Buy'  # 2B
                        trigger_price = p_now['data']['high']
                        stop_base = p_now['price']

            elif p_now['type'] == 'top':
                diff_ok = p_now['data']['diff'] < p_prev['data']['diff']
                price_ok = p_now['price'] < p_prev['price']
                cond = self._eval_div_condition(diff_ok)
                if price_ok and cond and is_bear:
                    sig = 'CloseLong'  # 2S
                    trigger_price = p_now['data']['low']
                    stop_base = p_now['price']

        # (S8: 段背驰入场信号已验证无效，已移除)

        # --------------------------------------------------------
        # S6: 单笔面积背驰补充路径（与S8共存）
        # --------------------------------------------------------
        if not sig and has_active_structure and len(self._bi_macd_areas) >= 3:
            if p_now['type'] == 'bottom':
                price_lower = p_now['price'] <= p_prev['price']
                area_div = self._has_area_divergence()
                if price_lower and area_div is True and is_bull:
                    sig = 'Buy'  # 底背驰2B
                    trigger_price = p_now['data']['high']
                    stop_base = p_now['price']

        # --------------------------------------------------------
        # 信号过滤与设置
        # --------------------------------------------------------
        if sig and self.atr > 0:
            # R1: 入场去重 — 检查近期是否在相同价格区域入过场
            if sig == 'Buy' and self.dedup_bars > 0 and self._recent_entries:
                dedup_dist = self.dedup_atr_mult * self.atr
                for entry in reversed(self._recent_entries):
                    bars_ago = self._bar_5m_count - entry['bar_5m_count']
                    if bars_ago > self.dedup_bars:
                        break  # 超出时间窗口
                    if abs(trigger_price - entry['price']) < dedup_dist:
                        sig = None  # 太近，去重
                        break

            # B09: CloseLong 信号仅在持有多仓时有效
            if sig == 'CloseLong':
                if self._position == 1:
                    # 有多仓，设置平仓信号
                    self._signal_type = "3S" if (self._pivots and last_pivot and
                        p_now['price'] < last_pivot['zd']) else "2S"
                    reason = f"{self._signal_type}: 平多信号"
                    self._pending_signal = {
                        'type': 'CloseLong',
                        'trigger_price': trigger_price,
                        'stop_base': stop_base
                    }
                    self.signal = f"待平多({self._signal_type})"
                    if self._debugger and self.trading:
                        self._debugger.log_signal(
                            signal_type=self._signal_type,
                            direction="CloseLong",
                            trigger_price=trigger_price,
                            stop_price=stop_base,
                            atr=self.atr,
                            reason=reason
                        )
                # 无多仓时忽略 CloseLong 信号
                return

            # 入场过滤：触发价与止损距离不超过 N 倍 ATR
            distance = abs(trigger_price - stop_base)
            if distance < self.atr_entry_filter * self.atr:
                # 判断信号类型（只有 Buy 信号能到这里）
                if self._pivots and last_pivot:
                    if p_now['type'] == 'bottom' and p_now['price'] > last_pivot['zg']:
                        self._signal_type = "3B"
                        reason = f"3B买点: 回踩不破中枢ZG={last_pivot['zg']:.0f}"
                    else:
                        self._signal_type = "2B"
                        reason = f"{self._signal_type}: 趋势延续确认"
                else:
                    self._signal_type = "2B"
                    reason = f"{self._signal_type}: 趋势延续确认"

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
        """1分钟级别检查入场/平仓（待触发信号）."""
        signal = self._pending_signal
        if not signal:
            return

        # S1: 冷却期间冻结 Buy 信号（CloseLong 仍允许，避免持仓风险）
        if self._cooldown_remaining > 0 and signal['type'] == 'Buy':
            self._pending_signal = None
            self.signal = ""
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

        elif signal['type'] == 'CloseLong':
            # B09: 平多仓信号
            if self._position != 1:
                # 已无多仓，信号失效
                self._pending_signal = None
                self.signal = ""
                return
            # 检查是否触发平仓（价格跌破触发价）
            if bar['low'] < signal['trigger_price']:
                fill_price = min(signal['trigger_price'], bar['open'])
                if fill_price < bar['low']:
                    fill_price = bar['close']
                # 平多仓
                pnl = fill_price - self._entry_price
                self.sell(fill_price, abs(self.pos))
                self.write_log(f"3S/2S平多: price={fill_price:.0f}, pnl={pnl:.0f}")
                if self._debugger and self.trading:
                    self._debugger.log_trade(
                        action="CLOSE_LONG",
                        price=fill_price,
                        volume=self.fixed_volume,
                        position=0,
                        pnl=pnl,
                        signal_type=self._signal_type
                    )
                self._position = 0
                self._trailing_active = False
                self._pending_signal = None
                self.signal = "平多"
                # B02/R2: 更新连亏计数
                self._update_loss_streak(pnl)

        elif signal['type'] == 'Sell':
            # 保留 Sell 逻辑以防万一（当前不应该被触发）
            if bar['high'] > signal['stop_base']:
                self._pending_signal = None
                self.signal = ""
                return
            if bar['low'] < signal['trigger_price']:
                fill_price = min(signal['trigger_price'], bar['open'])
                if fill_price < bar['low']:
                    fill_price = bar['close']
                self._open_position(-1, fill_price, signal['stop_base'])

    def _calc_stop_buffer(self) -> float:
        """B11: 计算止损 buffer，自适应 ATR."""
        pricetick = 2.0  # 棕榈油 pricetick
        if self.atr > 0:
            return max(pricetick, self.atr * self.stop_buffer_atr_pct)
        return pricetick

    def _open_position(self, direction: int, price: float, stop_base: float) -> None:
        """开仓."""
        buffer = self._calc_stop_buffer()
        if direction == 1:
            self.buy(price, self.fixed_volume)
            self._stop_price = stop_base - buffer
            self.signal = "3B/2B买入"
            self.write_log(f"开多: price={price:.0f}, stop={self._stop_price:.0f}, buffer={buffer:.1f}")
            action = "OPEN_LONG"
        else:
            self.short(price, self.fixed_volume)
            self._stop_price = stop_base + buffer
            self.signal = "3S/2S卖出"
            self.write_log(f"开空: price={price:.0f}, stop={self._stop_price:.0f}, buffer={buffer:.1f}")
            action = "OPEN_SHORT"

        self._position = direction
        self._entry_price = price
        self._initial_stop = self._stop_price  # R3: 记录初始止损
        self._trailing_active = False
        self._bars_since_entry = 0  # S5: 重置持仓bar计数
        self._pending_signal = None

        # R1: 递增当前中枢的入场计数 + 记录近期入场
        if direction == 1:
            if self._active_pivot is not None:
                self._active_pivot['entry_count'] = self._active_pivot.get('entry_count', 0) + 1
            self._recent_entries.append({
                'price': price,
                'bar_5m_count': self._bar_5m_count,
            })

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

        # S5: 最小持仓保护期间加宽止损
        effective_stop = self._stop_price
        if self.min_hold_bars > 0 and self._bars_since_entry <= self.min_hold_bars:
            # 保护期内：止损距离加倍（向远离方向移动）
            stop_dist = abs(self._entry_price - self._stop_price)
            if self._position == 1:
                effective_stop = self._entry_price - stop_dist * 2
            elif self._position == -1:
                effective_stop = self._entry_price + stop_dist * 2

        if self._position == 1:
            if bar['low'] <= effective_stop:
                sl_hit = True
                exit_price = bar['open'] if bar['open'] < effective_stop else effective_stop
        elif self._position == -1:
            if bar['high'] >= effective_stop:
                sl_hit = True
                exit_price = bar['open'] if bar['open'] > effective_stop else effective_stop

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

            # B02/R2: 更新连亏计数
            self._update_loss_streak(pnl)

            return True

        return False

    def _update_loss_streak(self, pnl: float) -> None:
        """B02/R2: 统一更新连亏计数和断路器."""
        if pnl < 0:
            self._consecutive_losses += 1
            # R2: L2 断路器（更严）
            if self.circuit_breaker_losses > 0 and self._consecutive_losses >= self.circuit_breaker_losses:
                self._cooldown_remaining = self.circuit_breaker_bars
                self._consecutive_losses = 0
            # B02: L1 冷却（温和）
            elif self.cooldown_losses > 0 and self._consecutive_losses >= self.cooldown_losses:
                self._cooldown_remaining = self.cooldown_bars
                # 不重置连亏计数，继续累积到 L2
                if self.circuit_breaker_losses == 0:
                    self._consecutive_losses = 0
        else:
            self._consecutive_losses = 0

    def _update_trailing_stop(self, bar: dict) -> None:
        """R3: 两段式出场（分阶段移动止损）.

        Phase 1: 浮盈 < 1R → 初始止损不动
        Phase 2: 浮盈 ≥ 1R → 抬止损到 entry + lock_profit_atr * ATR（锁盈）
        Phase 3: 浮盈 ≥ activate_mult * ATR → 启用 trailing（high - trailing_mult * ATR）
        """
        if self.atr <= 0:
            return

        if self._position == 1:
            float_pnl = bar['close'] - self._entry_price
        elif self._position == -1:
            float_pnl = self._entry_price - bar['close']
        else:
            return

        # 计算 1R = 初始止损距离
        initial_risk = abs(self._entry_price - self._initial_stop)
        if initial_risk <= 0:
            initial_risk = self.atr  # fallback

        # Phase 3: 启用 trailing（优先级最高）
        if not self._trailing_active:
            if float_pnl > (self.atr * self.atr_activate_mult):
                self._trailing_active = True

        if self._trailing_active:
            if self._position == 1:
                # S7: 结构trailing — 用最近笔底点作为止损参考
                new_stop = bar['high'] - (self.atr * self.atr_trailing_mult)
                if self.use_bi_trailing and len(self._bi_points) >= 2:
                    # 找最近的bottom bi point
                    for i in range(len(self._bi_points) - 1, max(len(self._bi_points) - 4, -1), -1):
                        bp = self._bi_points[i]
                        if bp['type'] == 'bottom' and bp['price'] > self._entry_price + self.atr:
                            buffer = self._calc_stop_buffer()
                            bi_stop = bp['price'] - buffer
                            # 取ATR trailing和笔低点中较高者（更紧但仍结构合理）
                            new_stop = max(new_stop, bi_stop)
                            break
                if new_stop > self._stop_price:
                    self._stop_price = new_stop
            else:
                new_stop = bar['low'] + (self.atr * self.atr_trailing_mult)
                if new_stop < self._stop_price:
                    self._stop_price = new_stop
            return

        # Phase 2: 浮盈 ≥ 1R → 锁盈（仅多头）
        if self.lock_profit_atr > 0 and float_pnl >= initial_risk:
            if self._position == 1:
                lock_stop = self._entry_price + self.lock_profit_atr * self.atr
                if lock_stop > self._stop_price:
                    self._stop_price = lock_stop

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
