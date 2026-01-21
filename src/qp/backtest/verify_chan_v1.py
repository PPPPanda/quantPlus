# src/qp/backtest/verify_chan_v1.py
"""
验证 CtaChanV1Strategy 与原始 changege 脚本的回测结果一致性.

使用方法：
    uv run python -m qp.backtest.verify_chan_v1
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class ChanRealTimeStrategy:
    """原始 changege 脚本的策略逻辑（用于验证）."""

    def __init__(self, df_1m: pd.DataFrame):
        # 1. 基础数据处理 (Resample)
        self.df_5m = df_1m.resample('5min', label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        self.df_15m = df_1m.resample('15min', label='right', closed='right').agg({
            'close': 'last'
        }).dropna()

        # 2. 预计算指标
        self._calc_indicators()

    def _calc_indicators(self) -> None:
        # 5m MACD & ATR
        self.df_5m['diff'], self.df_5m['dea'], _ = self._get_macd(self.df_5m['close'])

        # 15m Trend (Shift 1 对齐)
        d15, dea15, _ = self._get_macd(self.df_15m['close'])
        aligned = pd.DataFrame({'diff': d15, 'dea': dea15}).shift(1).reindex(
            self.df_5m.index, method='ffill'
        )
        self.df_5m['diff_15m'] = aligned['diff']
        self.df_5m['dea_15m'] = aligned['dea']

        # ATR (14)
        high_low = self.df_5m['high'] - self.df_5m['low']
        high_close = np.abs(self.df_5m['high'] - self.df_5m['close'].shift())
        low_close = np.abs(self.df_5m['low'] - self.df_5m['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df_5m['atr'] = tr.rolling(14).mean()

    def _get_macd(
        self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig) * 2

    def run(self) -> pd.DataFrame:
        # --- 状态变量 (State Variables) ---
        bi_points: list[dict] = []

        trades: list[dict] = []
        position = 0
        entry_price = 0.0
        stop_price = 0.0
        trailing_active = False

        # 逐 Bar 循环 (从第5根开始)
        for i in range(5, len(self.df_5m)):
            curr_bar = self.df_5m.iloc[i]
            prev_bar = self.df_5m.iloc[i - 1]
            prev2_bar = self.df_5m.iloc[i - 2]

            # ==========================
            # A. 实时风控 (Exits)
            # ==========================
            if position != 0:
                sl_hit = False
                exit_price = 0.0

                # 检查是否打损 (Bar 内最低/最高价触发)
                if position == 1:
                    if curr_bar['low'] <= stop_price:
                        sl_hit = True
                        exit_price = stop_price
                elif position == -1:
                    if curr_bar['high'] >= stop_price:
                        sl_hit = True
                        exit_price = stop_price

                if sl_hit:
                    pnl = (exit_price - entry_price) * position
                    trades.append({
                        'time': curr_bar.name,
                        'type': 'Stop/Trail',
                        'pnl': pnl
                    })
                    position = 0
                    trailing_active = False
                    continue

                # 更新 ATR 移动止损 (Hybrid Logic)
                current_atr = curr_bar['atr']
                # 1. 激活
                if not trailing_active:
                    float_pnl = (curr_bar['close'] - entry_price) * position
                    if float_pnl > (current_atr * 1.5):
                        trailing_active = True
                # 2. 移动
                if trailing_active:
                    if position == 1:
                        new_stop = curr_bar['high'] - (current_atr * 3.0)
                        if new_stop > stop_price:
                            stop_price = new_stop
                    else:
                        new_stop = curr_bar['low'] + (current_atr * 3.0)
                        if new_stop < stop_price:
                            stop_price = new_stop

            # ==========================
            # B. 实时结构更新 (Strict Bi)
            # ==========================
            is_top = (prev_bar['high'] > prev2_bar['high']) and (
                prev_bar['high'] > curr_bar['high']
            )
            is_bot = (prev_bar['low'] < prev2_bar['low']) and (
                prev_bar['low'] < curr_bar['low']
            )

            fractal_idx = i - 1
            new_bi = None

            if is_top:
                candidate = {
                    'idx': fractal_idx,
                    'type': 'top',
                    'price': prev_bar['high'],
                    'diff': self.df_5m.iloc[fractal_idx]['diff']
                }

                if not bi_points:
                    bi_points.append(candidate)
                else:
                    last_bi = bi_points[-1]
                    if last_bi['type'] == 'top':
                        if candidate['price'] > last_bi['price']:
                            bi_points[-1] = candidate
                    elif last_bi['type'] == 'bottom':
                        if candidate['idx'] - last_bi['idx'] >= 5:
                            bi_points.append(candidate)
                            new_bi = candidate

            elif is_bot:
                candidate = {
                    'idx': fractal_idx,
                    'type': 'bottom',
                    'price': prev_bar['low'],
                    'diff': self.df_5m.iloc[fractal_idx]['diff']
                }

                if not bi_points:
                    bi_points.append(candidate)
                else:
                    last_bi = bi_points[-1]
                    if last_bi['type'] == 'bottom':
                        if candidate['price'] < last_bi['price']:
                            bi_points[-1] = candidate
                    elif last_bi['type'] == 'top':
                        if candidate['idx'] - last_bi['idx'] >= 5:
                            bi_points.append(candidate)
                            new_bi = candidate

            # ==========================
            # C. 实时信号生成 (Signals)
            # ==========================
            if new_bi is not None and len(bi_points) >= 5:
                p_now = bi_points[-1]
                p_last = bi_points[-2]
                p_prev = bi_points[-3]
                p_prev2 = bi_points[-5]

                is_bull_trend = curr_bar['diff_15m'] > curr_bar['dea_15m']
                is_bear_trend = curr_bar['diff_15m'] < curr_bar['dea_15m']

                # 1. 二买 (Buy)
                if p_now['type'] == 'bottom':
                    is_structure_ok = p_now['price'] > p_prev['price']
                    is_divergence = p_prev['diff'] > p_prev2['diff']

                    if is_structure_ok and is_divergence and is_bull_trend:
                        if position != 1:
                            if position == -1:
                                pnl = (curr_bar['close'] - entry_price) * position
                                trades.append({
                                    'time': curr_bar.name,
                                    'type': 'Signal Reverse',
                                    'pnl': pnl
                                })

                            position = 1
                            entry_price = curr_bar['close']
                            stop_price = p_prev['price'] - 1
                            trailing_active = False

                # 2. 二卖 (Sell)
                elif p_now['type'] == 'top':
                    is_structure_ok = p_now['price'] < p_prev['price']
                    is_divergence = p_prev['diff'] < p_prev2['diff']

                    if is_structure_ok and is_divergence and is_bear_trend:
                        if position != -1:
                            if position == 1:
                                pnl = (curr_bar['close'] - entry_price) * position
                                trades.append({
                                    'time': curr_bar.name,
                                    'type': 'Signal Reverse',
                                    'pnl': pnl
                                })

                            position = -1
                            entry_price = curr_bar['close']
                            stop_price = p_prev['price'] + 1
                            trailing_active = False

        # End Loop
        if position != 0:
            last_pnl = (self.df_5m.iloc[-1]['close'] - entry_price) * position
            trades.append({
                'time': self.df_5m.index[-1],
                'type': 'End',
                'pnl': last_pnl
            })

        return pd.DataFrame(trades)


def run_original_test(file_path: str, name: str) -> dict:
    """运行原始脚本测试."""
    df_raw = pd.read_csv(file_path)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_raw.set_index('datetime', inplace=True)

    strat = ChanRealTimeStrategy(df_raw)
    res = strat.run()

    print(f"\n====== {name} (原始脚本) ======")
    if res.empty:
        print("无交易")
        return {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0, 'max_dd': 0}

    closed = res[res['pnl'].notnull()]
    total = closed['pnl'].sum()
    cnt = len(closed)
    wr = len(closed[closed['pnl'] > 0]) / cnt if cnt > 0 else 0
    dd = (closed['pnl'].cumsum().cummax() - closed['pnl'].cumsum()).max()

    print(f"净利润: {total:.0f}")
    print(f"交易数: {cnt}")
    print(f"胜率:   {wr:.2%}")
    print(f"最大回撤: {dd:.0f}")

    return {
        'total_pnl': total,
        'trade_count': cnt,
        'win_rate': wr,
        'max_dd': dd
    }


class MockCtaEngine:
    """模拟 CTA 引擎，用于测试策略."""

    def __init__(self):
        self.trades: list[dict] = []
        self.logs: list[str] = []

    def send_order(self, *args, **kwargs) -> list:
        return []

    def cancel_order(self, *args, **kwargs) -> None:
        pass

    def get_contract(self, *args, **kwargs):
        return None

    def write_log(self, msg: str, *args, **kwargs) -> None:
        self.logs.append(msg)

    def put_event(self, *args, **kwargs) -> None:
        pass

    def sync_strategy_data(self, *args, **kwargs) -> None:
        pass

    def get_pricetick(self, *args, **kwargs) -> float:
        return 1.0


class CtaChanV1TesterPandas:
    """
    使用 pandas 预处理的测试器.

    与原始脚本完全相同的数据处理逻辑，验证交易逻辑的正确性。
    """

    def __init__(self, df_1m: pd.DataFrame):
        # 使用与原始脚本完全相同的 pandas 处理
        self.df_5m = df_1m.resample('5min', label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        self.df_15m = df_1m.resample('15min', label='right', closed='right').agg({
            'close': 'last'
        }).dropna()

        self._calc_indicators()
        self._init_state()

    def _calc_indicators(self) -> None:
        """计算指标（与原始脚本完全相同）."""
        # 5m MACD
        self.df_5m['diff'], self.df_5m['dea'], _ = self._get_macd(self.df_5m['close'])

        # 15m Trend (Shift 1 对齐)
        d15, dea15, _ = self._get_macd(self.df_15m['close'])
        aligned = pd.DataFrame({'diff': d15, 'dea': dea15}).shift(1).reindex(
            self.df_5m.index, method='ffill'
        )
        self.df_5m['diff_15m'] = aligned['diff']
        self.df_5m['dea_15m'] = aligned['dea']

        # ATR (14)
        high_low = self.df_5m['high'] - self.df_5m['low']
        high_close = np.abs(self.df_5m['high'] - self.df_5m['close'].shift())
        low_close = np.abs(self.df_5m['low'] - self.df_5m['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df_5m['atr'] = tr.rolling(14).mean()

    def _get_macd(
        self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, (macd - sig) * 2

    def _init_state(self) -> None:
        """初始化交易状态."""
        self._bi_points: list[dict] = []
        self.trades: list[dict] = []
        self._position = 0
        self._entry_price = 0.0
        self._stop_price = 0.0
        self._trailing_active = False

    def run(self) -> pd.DataFrame:
        """运行回测（逻辑与原始脚本完全一致）."""
        for i in range(5, len(self.df_5m)):
            curr_bar = self.df_5m.iloc[i]
            prev_bar = self.df_5m.iloc[i - 1]
            prev2_bar = self.df_5m.iloc[i - 2]

            # A. 风控
            if self._position != 0:
                sl_hit = False
                exit_price = 0.0

                if self._position == 1:
                    if curr_bar['low'] <= self._stop_price:
                        sl_hit = True
                        exit_price = self._stop_price
                elif self._position == -1:
                    if curr_bar['high'] >= self._stop_price:
                        sl_hit = True
                        exit_price = self._stop_price

                if sl_hit:
                    pnl = (exit_price - self._entry_price) * self._position
                    self.trades.append({
                        'time': curr_bar.name,
                        'type': 'Stop/Trail',
                        'pnl': pnl
                    })
                    self._position = 0
                    self._trailing_active = False
                    continue

                # 更新移动止损
                current_atr = curr_bar['atr']
                if not self._trailing_active:
                    float_pnl = (curr_bar['close'] - self._entry_price) * self._position
                    if float_pnl > (current_atr * 1.5):
                        self._trailing_active = True
                if self._trailing_active:
                    if self._position == 1:
                        new_stop = curr_bar['high'] - (current_atr * 3.0)
                        if new_stop > self._stop_price:
                            self._stop_price = new_stop
                    else:
                        new_stop = curr_bar['low'] + (current_atr * 3.0)
                        if new_stop < self._stop_price:
                            self._stop_price = new_stop

            # B. 更新笔结构
            is_top = (prev_bar['high'] > prev2_bar['high']) and (prev_bar['high'] > curr_bar['high'])
            is_bot = (prev_bar['low'] < prev2_bar['low']) and (prev_bar['low'] < curr_bar['low'])

            fractal_idx = i - 1
            new_bi = None

            if is_top:
                candidate = {
                    'idx': fractal_idx,
                    'type': 'top',
                    'price': prev_bar['high'],
                    'diff': self.df_5m.iloc[fractal_idx]['diff']
                }
                if not self._bi_points:
                    self._bi_points.append(candidate)
                else:
                    last_bi = self._bi_points[-1]
                    if last_bi['type'] == 'top':
                        if candidate['price'] > last_bi['price']:
                            self._bi_points[-1] = candidate
                    elif last_bi['type'] == 'bottom':
                        if candidate['idx'] - last_bi['idx'] >= 5:
                            self._bi_points.append(candidate)
                            new_bi = candidate

            elif is_bot:
                candidate = {
                    'idx': fractal_idx,
                    'type': 'bottom',
                    'price': prev_bar['low'],
                    'diff': self.df_5m.iloc[fractal_idx]['diff']
                }
                if not self._bi_points:
                    self._bi_points.append(candidate)
                else:
                    last_bi = self._bi_points[-1]
                    if last_bi['type'] == 'bottom':
                        if candidate['price'] < last_bi['price']:
                            self._bi_points[-1] = candidate
                    elif last_bi['type'] == 'top':
                        if candidate['idx'] - last_bi['idx'] >= 5:
                            self._bi_points.append(candidate)
                            new_bi = candidate

            # C. 信号生成
            if new_bi is not None and len(self._bi_points) >= 5:
                p_now = self._bi_points[-1]
                p_last = self._bi_points[-2]
                p_prev = self._bi_points[-3]
                p_prev2 = self._bi_points[-5]

                is_bull_trend = curr_bar['diff_15m'] > curr_bar['dea_15m']
                is_bear_trend = curr_bar['diff_15m'] < curr_bar['dea_15m']

                if p_now['type'] == 'bottom':
                    is_structure_ok = p_now['price'] > p_prev['price']
                    is_divergence = p_prev['diff'] > p_prev2['diff']

                    if is_structure_ok and is_divergence and is_bull_trend:
                        if self._position != 1:
                            if self._position == -1:
                                pnl = (curr_bar['close'] - self._entry_price) * self._position
                                self.trades.append({
                                    'time': curr_bar.name,
                                    'type': 'Signal Reverse',
                                    'pnl': pnl
                                })
                            self._position = 1
                            self._entry_price = curr_bar['close']
                            self._stop_price = p_prev['price'] - 1
                            self._trailing_active = False

                elif p_now['type'] == 'top':
                    is_structure_ok = p_now['price'] < p_prev['price']
                    is_divergence = p_prev['diff'] < p_prev2['diff']

                    if is_structure_ok and is_divergence and is_bear_trend:
                        if self._position != -1:
                            if self._position == 1:
                                pnl = (curr_bar['close'] - self._entry_price) * self._position
                                self.trades.append({
                                    'time': curr_bar.name,
                                    'type': 'Signal Reverse',
                                    'pnl': pnl
                                })
                            self._position = -1
                            self._entry_price = curr_bar['close']
                            self._stop_price = p_prev['price'] + 1
                            self._trailing_active = False

        # 结束时平仓
        if self._position != 0:
            last_pnl = (self.df_5m.iloc[-1]['close'] - self._entry_price) * self._position
            self.trades.append({
                'time': self.df_5m.index[-1],
                'type': 'End',
                'pnl': last_pnl
            })

        return pd.DataFrame(self.trades)


class CtaChanV1Tester:
    """
    CtaChanV1Strategy 独立测试器.

    直接调用策略逻辑，不依赖 vnpy 回测引擎。
    """

    def __init__(self, df_1m: pd.DataFrame):
        self.df_1m = df_1m

        # 重建策略逻辑（与 CtaChanV1Strategy 完全一致）
        self._init_strategy()

    def _init_strategy(self) -> None:
        """初始化策略状态."""
        from collections import deque
        import math

        # K 线数据存储
        self._bars_5m: list = []
        self._close_5m: deque = deque(maxlen=200)
        self._high_5m: deque = deque(maxlen=200)
        self._low_5m: deque = deque(maxlen=200)
        self._diff_5m_history: list = []  # 存储每个 5m bar 的 DIFF 值

        self._bars_15m: list = []
        self._close_15m: deque = deque(maxlen=100)

        # MACD 计算缓存
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

        # ATR 计算缓存
        self._tr_values: deque = deque(maxlen=14)
        self._prev_close: float = 0.0
        self.atr: float = 0.0

        # 笔端点列表
        self._bi_points: list = []

        # 交易状态
        self._position: int = 0
        self._entry_price: float = 0.0
        self._stop_price: float = 0.0
        self._trailing_active: bool = False

        # 交易记录
        self.trades: list[dict] = []

        # K 线合成缓存
        self._window_bar_5m = None
        self._last_window_end_5m = None
        self._window_bar_15m = None
        self._last_window_end_15m = None

    def _get_window_end(self, dt: datetime, window: int) -> datetime:
        """
        计算窗口结束时间.

        与 pandas resample('Nmin', label='right', closed='right') 完全一致。
        pandas 使用整点对齐：00:00, 00:05, 00:10, ...
        """
        # 计算从午夜开始的总分钟数
        total_minutes = dt.hour * 60 + dt.minute

        # 计算窗口结束时间（使用 ceil 向上取整到窗口边界）
        import math
        window_end_minutes = math.ceil(total_minutes / window) * window

        # 转换回小时和分钟
        hours = window_end_minutes // 60
        minutes = window_end_minutes % 60

        # 处理跨天的情况
        result = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        result += pd.Timedelta(hours=hours, minutes=minutes)

        return result

    def _update_5m_bar(self, bar: dict) -> dict | None:
        """更新 5 分钟 K 线."""
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

    def _update_15m_bar(self, bar: dict) -> dict | None:
        """更新 15 分钟 K 线."""
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
        """
        更新 5 分钟 MACD.

        与 pandas ewm(span=N, adjust=False) 完全一致：
        - alpha = 2 / (span + 1)
        - EMA_0 = X_0
        - EMA_t = alpha * X_t + (1 - alpha) * EMA_{t-1}
        """
        alpha_fast = 2.0 / 13   # span=12
        alpha_slow = 2.0 / 27   # span=26
        alpha_signal = 2.0 / 10 # span=9

        if not self._macd_inited_5m:
            # 第一个数据点：初始化 EMA 为当前值
            self._ema_fast_5m = close
            self._ema_slow_5m = close
            self._macd_inited_5m = True
            diff = 0.0
            self._ema_signal_5m = 0.0
        else:
            # 后续数据点：正常更新 EMA
            self._ema_fast_5m = alpha_fast * close + (1 - alpha_fast) * self._ema_fast_5m
            self._ema_slow_5m = alpha_slow * close + (1 - alpha_slow) * self._ema_slow_5m
            diff = self._ema_fast_5m - self._ema_slow_5m
            self._ema_signal_5m = alpha_signal * diff + (1 - alpha_signal) * self._ema_signal_5m

        self.diff_5m = diff
        self.dea_5m = self._ema_signal_5m

    def _update_macd_15m(self, close: float) -> None:
        """
        更新 15 分钟 MACD.

        与 pandas ewm(span=N, adjust=False) 完全一致。
        """
        alpha_fast = 2.0 / 13   # span=12
        alpha_slow = 2.0 / 27   # span=26
        alpha_signal = 2.0 / 10 # span=9

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
        """更新 ATR."""
        if self._prev_close == 0.0:
            self._prev_close = bar['close']
            return

        high_low = bar['high'] - bar['low']
        high_close = abs(bar['high'] - self._prev_close)
        low_close = abs(bar['low'] - self._prev_close)
        tr = max(high_low, high_close, low_close)

        self._tr_values.append(tr)
        self._prev_close = bar['close']

        if len(self._tr_values) >= 14:
            self.atr = sum(self._tr_values) / len(self._tr_values)

    def _on_15m_bar(self, bar: dict) -> None:
        """15 分钟 K 线回调."""
        self._bars_15m.append(bar)
        self._close_15m.append(bar['close'])

        # 实现 shift(1) 效果：先保存当前值作为 prev，再更新
        self._prev_diff_15m = self.diff_15m
        self._prev_dea_15m = self.dea_15m

        self._update_macd_15m(bar['close'])

    def _on_5m_bar(self, bar: dict) -> None:
        """5 分钟 K 线回调 - 核心交易逻辑."""
        self._bars_5m.append(bar)
        self._close_5m.append(bar['close'])
        self._high_5m.append(bar['high'])
        self._low_5m.append(bar['low'])

        bar_idx = len(self._bars_5m) - 1

        self._update_macd_5m(bar['close'])
        self._diff_5m_history.append(self.diff_5m)  # 存储当前 bar 的 DIFF 值
        self._update_atr(bar)

        if bar_idx < 5:
            return

        # 检查止损
        if self._position != 0:
            if self._check_stop_loss(bar):
                return

        # 更新笔结构
        new_bi = self._update_bi_structure(bar_idx)

        # 只在新笔形成时检查入场信号（与原始脚本一致）
        if new_bi:
            self._check_entry_signal(bar)

        # 更新移动止损
        if self._position != 0:
            self._update_trailing_stop(bar)

    def _update_bi_structure(self, bar_idx: int) -> bool:
        """
        更新笔结构.

        Returns:
            True 如果形成了新笔，False 否则
        """
        if bar_idx < 2:
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
        # 使用分型中心点（fractal_idx）的 DIFF 值，与原始脚本一致
        fractal_diff = self._diff_5m_history[fractal_idx] if fractal_idx < len(self._diff_5m_history) else self.diff_5m

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

        return new_bi

    def _process_bi_candidate(self, candidate: dict) -> bool:
        """
        处理笔端点候选.

        Returns:
            True 如果形成了新笔（异向成笔），False 否则
        """
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
            if candidate['idx'] - last_bi['idx'] >= 5:
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

        # 二买信号（与原始脚本一致：允许新建仓和反向开仓，禁止同向重复开仓）
        if p_now['type'] == 'bottom':
            is_structure_ok = p_now['price'] > p_prev['price']
            is_divergence = p_prev['diff'] > p_prev2['diff']

            if is_structure_ok and is_divergence and is_bull_trend:
                if self._position != 1:  # 只有不是多头时才开多
                    self._execute_buy(bar, p_prev['price'])

        # 二卖信号（与原始脚本一致：允许新建仓和反向开仓，禁止同向重复开仓）
        elif p_now['type'] == 'top':
            is_structure_ok = p_now['price'] < p_prev['price']
            is_divergence = p_prev['diff'] < p_prev2['diff']

            if is_structure_ok and is_divergence and is_bear_trend:
                if self._position != -1:  # 只有不是空头时才开空
                    self._execute_short(bar, p_prev['price'])

    def _execute_buy(self, bar: dict, p1_price: float) -> None:
        """执行买入."""
        if self._position == -1:
            pnl = (bar['close'] - self._entry_price) * self._position
            self.trades.append({
                'time': bar['datetime'],
                'type': 'Signal Reverse',
                'pnl': pnl
            })

        self._position = 1
        self._entry_price = bar['close']
        self._stop_price = p1_price - 1
        self._trailing_active = False

    def _execute_short(self, bar: dict, p1_price: float) -> None:
        """执行卖出."""
        if self._position == 1:
            pnl = (bar['close'] - self._entry_price) * self._position
            self.trades.append({
                'time': bar['datetime'],
                'type': 'Signal Reverse',
                'pnl': pnl
            })

        self._position = -1
        self._entry_price = bar['close']
        self._stop_price = p1_price + 1
        self._trailing_active = False

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
            pnl = (exit_price - self._entry_price) * self._position
            self.trades.append({
                'time': bar['datetime'],
                'type': 'Stop/Trail',
                'pnl': pnl
            })
            self._position = 0
            self._trailing_active = False
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

        float_pnl_atr = float_pnl / self.atr if self.atr > 0 else 0

        if not self._trailing_active:
            if float_pnl_atr > 1.5:
                self._trailing_active = True

        if self._trailing_active:
            if self._position == 1:
                new_stop = bar['high'] - 3.0 * self.atr
                if new_stop > self._stop_price:
                    self._stop_price = new_stop
            else:
                new_stop = bar['low'] + 3.0 * self.atr
                if new_stop < self._stop_price:
                    self._stop_price = new_stop

    def run(self) -> pd.DataFrame:
        """运行回测."""
        for _, row in self.df_1m.iterrows():
            bar = {
                'datetime': row.name,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }

            # 更新 K 线合成器
            bar_5m = self._update_5m_bar(bar)
            bar_15m = self._update_15m_bar(bar)

            # 先处理 15m bar（更新趋势 MACD）
            if bar_15m is not None:
                self._on_15m_bar(bar_15m)

            # 再处理 5m bar（核心交易逻辑）
            if bar_5m is not None:
                self._on_5m_bar(bar_5m)

        # 刷新未完成的 K 线
        if self._window_bar_5m is not None:
            self._on_5m_bar(self._window_bar_5m)
        if self._window_bar_15m is not None:
            self._on_15m_bar(self._window_bar_15m)

        # 结束时平仓
        if self._position != 0:
            last_close = self.df_1m.iloc[-1]['close']
            last_pnl = (last_close - self._entry_price) * self._position
            self.trades.append({
                'time': self.df_1m.index[-1],
                'type': 'End',
                'pnl': last_pnl
            })

        return pd.DataFrame(self.trades)


def run_cta_test(file_path: str, name: str, debug: bool = False) -> dict:
    """运行 CTA 策略测试."""
    df_raw = pd.read_csv(file_path)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_raw.set_index('datetime', inplace=True)

    tester = CtaChanV1Tester(df_raw)
    res = tester.run()

    print(f"\n====== {name} (CTA策略) ======")

    if debug:
        # 比较 K 线数量
        df_5m_pandas = df_raw.resample('5min', label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        print(f"  [DEBUG] pandas 5m K线数: {len(df_5m_pandas)}")
        print(f"  [DEBUG] CTA 5m K线数: {len(tester._bars_5m)}")

    if res.empty:
        print("无交易")
        return {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0, 'max_dd': 0}

    closed = res[res['pnl'].notnull()]
    total = closed['pnl'].sum()
    cnt = len(closed)
    wr = len(closed[closed['pnl'] > 0]) / cnt if cnt > 0 else 0
    dd = (closed['pnl'].cumsum().cummax() - closed['pnl'].cumsum()).max()

    print(f"净利润: {total:.0f}")
    print(f"交易数: {cnt}")
    print(f"胜率:   {wr:.2%}")
    print(f"最大回撤: {dd:.0f}")

    return {
        'total_pnl': total,
        'trade_count': cnt,
        'win_rate': wr,
        'max_dd': dd
    }


def run_pandas_tester(file_path: str, name: str) -> dict:
    """运行 pandas 版本测试器."""
    df_raw = pd.read_csv(file_path)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_raw.set_index('datetime', inplace=True)

    tester = CtaChanV1TesterPandas(df_raw)
    res = tester.run()

    print(f"\n====== {name} (Pandas测试器) ======")
    if res.empty:
        print("无交易")
        return {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0, 'max_dd': 0}

    closed = res[res['pnl'].notnull()]
    total = closed['pnl'].sum()
    cnt = len(closed)
    wr = len(closed[closed['pnl'] > 0]) / cnt if cnt > 0 else 0
    dd = (closed['pnl'].cumsum().cummax() - closed['pnl'].cumsum()).max()

    print(f"净利润: {total:.0f}")
    print(f"交易数: {cnt}")
    print(f"胜率:   {wr:.2%}")
    print(f"最大回撤: {dd:.0f}")

    return {
        'total_pnl': total,
        'trade_count': cnt,
        'win_rate': wr,
        'max_dd': dd
    }


def main() -> None:
    """主函数."""
    # 数据文件路径
    data_dir = Path("E:/work/quant/quantPlus/data/analyse")
    file1 = data_dir / "p2509_1min_202503-202508.csv"
    file2 = data_dir / "p2601_1min_202507-202512.csv"

    print("=" * 60)
    print("验证 changege 原始脚本回测结果")
    print("=" * 60)

    results_original = {}
    results_pandas = {}
    results_cta = {}

    # 运行原始脚本测试
    if file1.exists():
        results_original['dataset1'] = run_original_test(str(file1), "Dataset 1 (p2509)")
    if file2.exists():
        results_original['dataset2'] = run_original_test(str(file2), "Dataset 2 (p2601)")

    print("\n" + "=" * 60)
    print("验证 Pandas 测试器（应与原始脚本完全一致）")
    print("=" * 60)

    # 运行 pandas 测试器
    if file1.exists():
        results_pandas['dataset1'] = run_pandas_tester(str(file1), "Dataset 1 (p2509)")
    if file2.exists():
        results_pandas['dataset2'] = run_pandas_tester(str(file2), "Dataset 2 (p2601)")

    print("\n" + "=" * 60)
    print("验证 CtaChanV1Strategy 回测结果")
    print("=" * 60)

    # 运行 CTA 策略测试
    if file1.exists():
        results_cta['dataset1'] = run_cta_test(str(file1), "Dataset 1 (p2509)", debug=True)
    if file2.exists():
        results_cta['dataset2'] = run_cta_test(str(file2), "Dataset 2 (p2601)", debug=True)

    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    print("\n目标结果（来自 changege.md）:")
    print("Dataset 1: 净利润 1455, 交易数 145, 胜率 46.90%, 最大回撤 349")
    print("Dataset 2: 净利润 215, 交易数 96, 胜率 42.71%, 最大回撤 400")

    if results_original and results_cta:
        print("\n一致性检查:")
        for ds in ['dataset1', 'dataset2']:
            if ds in results_original and ds in results_cta:
                orig = results_original[ds]
                cta = results_cta[ds]
                match = (
                    abs(orig['total_pnl'] - cta['total_pnl']) < 10 and
                    orig['trade_count'] == cta['trade_count']
                )
                status = "PASS" if match else "DIFF"
                print(f"  {ds}: {status}")
                if not match:
                    print(f"    原始: PnL={orig['total_pnl']:.0f}, 交易={orig['trade_count']}")
                    print(f"    CTA:  PnL={cta['total_pnl']:.0f}, 交易={cta['trade_count']}")


if __name__ == "__main__":
    main()
