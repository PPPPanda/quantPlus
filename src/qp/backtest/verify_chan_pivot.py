"""
验证 CtaChanPivotStrategy 与原始脚本结果一致性.

运行方式：
    uv run python src/qp/backtest/verify_chan_pivot.py
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 原始脚本测试器（Pandas 批量处理）
# =============================================================================
class ChanPivotTesterPandas:
    """使用原始脚本逻辑的测试器."""

    def __init__(self, df_1m: pd.DataFrame):
        self.df_1m = df_1m.reset_index(drop=True)
        self.trades = []
        self.position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.trailing_active = False
        self.ACTIVATE_ATR = 1.5
        self.TRAIL_ATR = 3.0

        # 包含处理后的K线
        self.k_lines = []
        self.inclusion_dir = 0
        self.bi_points = []
        self.pivots = []
        self.pending_signal = None

        # 预计算数据
        df_1m_idx = self.df_1m.set_index('datetime')
        df_1m_idx.index = pd.to_datetime(df_1m_idx.index)

        self.df_5m = df_1m_idx.resample('5min', label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        self._calc_indicators()

    def _calc_indicators(self):
        """计算指标."""
        df = self.df_5m

        # 5m MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        sig = macd.ewm(span=9, adjust=False).mean()
        df['diff'] = macd
        df['dea'] = sig

        # 15m MACD
        df_15m = df.resample('15min', closed='right', label='right').agg({'close': 'last'}).dropna()
        e1 = df_15m['close'].ewm(span=12, adjust=False).mean()
        e2 = df_15m['close'].ewm(span=26, adjust=False).mean()
        m = e1 - e2
        s = m.ewm(span=9, adjust=False).mean()
        aligned = pd.DataFrame({'diff': m, 'dea': s}).shift(1).reindex(df.index, method='ffill')
        df['diff_15m'] = aligned['diff']
        df['dea_15m'] = aligned['dea']

        # ATR
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

    def run(self) -> pd.DataFrame:
        """运行回测."""
        for i, row in self.df_1m.iterrows():
            current_time = pd.to_datetime(row['datetime'])

            # 1. 检查止损
            if self.position != 0:
                self._check_exit(row)

            # 2. 检查入场
            if self.position == 0 and self.pending_signal:
                self._check_entry(row)

            # 3. 5分钟更新
            if current_time.minute % 5 == 0:
                if current_time in self.df_5m.index:
                    bar_5m = self.df_5m.loc[current_time]
                    self._on_bar_close(bar_5m)

                    if self.position != 0:
                        self._update_trailing_stop(bar_5m)

        return pd.DataFrame(self.trades)

    def _check_entry(self, row):
        signal = self.pending_signal
        if not signal:
            return

        if signal['type'] == 'Buy':
            if row['low'] < signal['stop_base']:
                self.pending_signal = None
                return
            if row['high'] > signal['trigger_price']:
                fill = max(signal['trigger_price'], row['open'])
                if fill > row['high']:
                    fill = row['close']
                self._open_position(1, fill, row['datetime'], signal['stop_base'])

        elif signal['type'] == 'Sell':
            if row['high'] > signal['stop_base']:
                self.pending_signal = None
                return
            if row['low'] < signal['trigger_price']:
                fill = min(signal['trigger_price'], row['open'])
                if fill < row['low']:
                    fill = row['close']
                self._open_position(-1, fill, row['datetime'], signal['stop_base'])

    def _open_position(self, direction, price, time, stop_base):
        self.position = direction
        self.entry_price = price
        self.stop_price = stop_base - 1 if direction == 1 else stop_base + 1
        self.pending_signal = None
        self.trailing_active = False

    def _check_exit(self, row):
        hit = False
        exit_px = 0
        if self.position == 1:
            if row['low'] <= self.stop_price:
                hit = True
                exit_px = row['open'] if row['open'] < self.stop_price else self.stop_price
        elif self.position == -1:
            if row['high'] >= self.stop_price:
                hit = True
                exit_px = row['open'] if row['open'] > self.stop_price else self.stop_price
        if hit:
            pnl = (exit_px - self.entry_price) * self.position
            self.trades.append({
                'time': row['datetime'],
                'type': 'Stop/Trail',
                'pnl': pnl
            })
            self.position = 0

    def _update_trailing_stop(self, curr_bar):
        atr = curr_bar['atr'] if not np.isnan(curr_bar['atr']) else 0
        pnl = (curr_bar['close'] - self.entry_price) * self.position
        if not self.trailing_active and pnl > self.ACTIVATE_ATR * atr:
            self.trailing_active = True
        if self.trailing_active:
            if self.position == 1:
                new = curr_bar['high'] - self.TRAIL_ATR * atr
                if new > self.stop_price:
                    self.stop_price = new
            else:
                new = curr_bar['low'] + self.TRAIL_ATR * atr
                if new < self.stop_price:
                    self.stop_price = new

    def _on_bar_close(self, curr_bar):
        bar = {
            'high': curr_bar['high'],
            'low': curr_bar['low'],
            'time': curr_bar.name,
            'diff': curr_bar['diff'],
            'atr': curr_bar['atr'],
            'diff_15m': curr_bar['diff_15m'],
            'dea_15m': curr_bar['dea_15m']
        }

        self._process_inclusion(bar)
        new_bi = self._process_bi()

        if new_bi:
            self._check_signal(curr_bar, new_bi)

    def _process_inclusion(self, new_bar):
        if not self.k_lines:
            self.k_lines.append(new_bar)
            return
        last = self.k_lines[-1]
        in_last = new_bar['high'] <= last['high'] and new_bar['low'] >= last['low']
        in_new = last['high'] <= new_bar['high'] and last['low'] >= new_bar['low']

        if in_last or in_new:
            if self.inclusion_dir == 0:
                self.inclusion_dir = 1
            merged = last.copy()
            merged['time'] = new_bar['time']
            merged['diff'] = new_bar['diff']
            merged['atr'] = new_bar['atr']
            merged['diff_15m'] = new_bar['diff_15m']
            merged['dea_15m'] = new_bar['dea_15m']
            if self.inclusion_dir == 1:
                merged['high'] = max(last['high'], new_bar['high'])
                merged['low'] = max(last['low'], new_bar['low'])
            else:
                merged['high'] = min(last['high'], new_bar['high'])
                merged['low'] = min(last['low'], new_bar['low'])
            self.k_lines[-1] = merged
        else:
            if new_bar['high'] > last['high'] and new_bar['low'] > last['low']:
                self.inclusion_dir = 1
            elif new_bar['high'] < last['high'] and new_bar['low'] < last['low']:
                self.inclusion_dir = -1
            self.k_lines.append(new_bar)

    def _process_bi(self):
        if len(self.k_lines) < 3:
            return None
        curr = self.k_lines[-1]
        mid = self.k_lines[-2]
        left = self.k_lines[-3]
        is_top = mid['high'] > left['high'] and mid['high'] > curr['high']
        is_bot = mid['low'] < left['low'] and mid['low'] < curr['low']
        cand = None
        if is_top:
            cand = {'type': 'top', 'price': mid['high'], 'idx': len(self.k_lines) - 2, 'data': mid}
        elif is_bot:
            cand = {'type': 'bottom', 'price': mid['low'], 'idx': len(self.k_lines) - 2, 'data': mid}

        if not cand:
            return None
        if not self.bi_points:
            self.bi_points.append(cand)
            return None
        last = self.bi_points[-1]
        if last['type'] == cand['type']:
            if last['type'] == 'top' and cand['price'] > last['price']:
                self.bi_points[-1] = cand
            elif last['type'] == 'bottom' and cand['price'] < last['price']:
                self.bi_points[-1] = cand
        else:
            if cand['idx'] - last['idx'] >= 4:
                self.bi_points.append(cand)
                return cand
        return None

    def _update_pivots(self):
        if len(self.bi_points) < 4:
            return

        b0 = self.bi_points[-4]
        b1 = self.bi_points[-3]
        b2 = self.bi_points[-2]
        b3 = self.bi_points[-1]

        r1 = (min(b0['price'], b1['price']), max(b0['price'], b1['price']))
        r2 = (min(b1['price'], b2['price']), max(b1['price'], b2['price']))
        r3 = (min(b2['price'], b3['price']), max(b2['price'], b3['price']))

        zg = min(r1[1], r2[1], r3[1])
        zd = max(r1[0], r2[0], r3[0])

        if zg > zd:
            new_p = {
                'zg': zg,
                'zd': zd,
                'start_bi_idx': len(self.bi_points) - 4,
                'end_bi_idx': len(self.bi_points) - 1
            }
            self.pivots.append(new_p)

    def _check_signal(self, curr_bar, new_bi):
        self._update_pivots()

        if len(self.bi_points) < 5:
            return

        p_now = self.bi_points[-1]
        p_last = self.bi_points[-2]
        p_prev = self.bi_points[-3]

        is_bull = curr_bar['diff_15m'] > curr_bar['dea_15m']
        is_bear = curr_bar['diff_15m'] < curr_bar['dea_15m']
        sig = None

        if self.pivots:
            last_pivot = self.pivots[-1]

            if p_now['type'] == 'bottom':
                if p_now['price'] > last_pivot['zg']:
                    if p_last['price'] > last_pivot['zg']:
                        if last_pivot['end_bi_idx'] >= len(self.bi_points) - 6:
                            if is_bull:
                                sig = 'Buy'

            elif p_now['type'] == 'top':
                if p_now['price'] < last_pivot['zd']:
                    if p_last['price'] < last_pivot['zd']:
                        if last_pivot['end_bi_idx'] >= len(self.bi_points) - 6:
                            if is_bear:
                                sig = 'Sell'

        if not sig:
            if p_now['type'] == 'bottom':
                div = p_now['data']['diff'] > p_prev['data']['diff']
                if p_now['price'] > p_prev['price'] and div and is_bull:
                    sig = 'Buy'
            elif p_now['type'] == 'top':
                div = p_now['data']['diff'] < p_prev['data']['diff']
                if p_now['price'] < p_prev['price'] and div and is_bear:
                    sig = 'Sell'

        atr = curr_bar['atr']
        if sig == 'Buy':
            trig = p_now['data']['high']
            if (trig - p_now['price']) < 2.0 * atr:
                self.pending_signal = {'type': 'Buy', 'trigger_price': trig, 'stop_base': p_now['price']}
        elif sig == 'Sell':
            trig = p_now['data']['low']
            if (p_now['price'] - trig) < 2.0 * atr:
                self.pending_signal = {'type': 'Sell', 'trigger_price': trig, 'stop_base': p_now['price']}


# =============================================================================
# CTA 策略测试器（增量计算）
# =============================================================================
class ChanPivotTesterCTA:
    """模拟 CtaChanPivotStrategy 的增量计算测试器."""

    def __init__(self, df_1m: pd.DataFrame):
        self.df_1m = df_1m.reset_index(drop=True)
        self.trades = []

        # K线合成
        self._window_bar_5m = None
        self._last_window_end_5m = None
        self._window_bar_15m = None
        self._last_window_end_15m = None

        # 包含处理
        self._k_lines = []
        self._inclusion_dir = 0

        # MACD
        self._ema_fast_5m = 0.0
        self._ema_slow_5m = 0.0
        self._ema_signal_5m = 0.0
        self._ema_fast_15m = 0.0
        self._ema_slow_15m = 0.0
        self._ema_signal_15m = 0.0
        self._macd_inited_5m = False
        self._macd_inited_15m = False

        self.diff_5m = 0.0
        self.dea_5m = 0.0
        self.diff_15m = 0.0
        self.dea_15m = 0.0
        self._prev_diff_15m = 0.0
        self._prev_dea_15m = 0.0

        # ATR
        self._tr_values = deque(maxlen=14)
        self._prev_close_5m = 0.0
        self.atr = 0.0

        # 笔和中枢
        self._bi_points = []
        self._pivots = []

        # 信号
        self._pending_signal = None

        # 交易状态
        self._position = 0
        self._entry_price = 0.0
        self._stop_price = 0.0
        self._trailing_active = False

        self.ACTIVATE_ATR = 1.5
        self.TRAIL_ATR = 3.0

    def run(self) -> pd.DataFrame:
        """运行回测."""
        for i, row in self.df_1m.iterrows():
            bar_dict = {
                'datetime': pd.to_datetime(row['datetime']),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
            }

            # 1. 检查止损
            if self._position != 0:
                if self._check_stop_loss_1m(bar_dict):
                    continue

            # 2. 检查入场
            if self._position == 0 and self._pending_signal:
                self._check_entry_1m(bar_dict)

            # 3. 更新 15m
            bar_15m = self._update_15m_bar(bar_dict)
            if bar_15m:
                self._on_15m_bar(bar_15m)

            # 4. 更新 5m
            bar_5m = self._update_5m_bar(bar_dict)
            if bar_5m:
                self._on_5m_bar(bar_5m)

        return pd.DataFrame(self.trades)

    def _get_window_end(self, dt, window):
        total_minutes = dt.hour * 60 + dt.minute
        window_end_minutes = math.ceil(total_minutes / window) * window
        hours = window_end_minutes // 60
        minutes = window_end_minutes % 60
        result = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        result += pd.Timedelta(hours=hours, minutes=minutes)
        return result

    def _update_5m_bar(self, bar):
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

    def _update_15m_bar(self, bar):
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

    def _update_macd_5m(self, close):
        alpha_fast = 2.0 / 13
        alpha_slow = 2.0 / 27
        alpha_signal = 2.0 / 10

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

    def _update_macd_15m(self, close):
        alpha_fast = 2.0 / 13
        alpha_slow = 2.0 / 27
        alpha_signal = 2.0 / 10

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

    def _update_atr(self, bar):
        if self._prev_close_5m == 0.0:
            self._prev_close_5m = bar['close']
            return

        high_low = bar['high'] - bar['low']
        high_close = abs(bar['high'] - self._prev_close_5m)
        low_close = abs(bar['low'] - self._prev_close_5m)
        tr = max(high_low, high_close, low_close)

        self._tr_values.append(tr)
        self._prev_close_5m = bar['close']

        if len(self._tr_values) >= 14:
            self.atr = sum(self._tr_values) / len(self._tr_values)

    def _on_15m_bar(self, bar):
        self._prev_diff_15m = self.diff_15m
        self._prev_dea_15m = self.dea_15m
        self._update_macd_15m(bar['close'])

    def _on_5m_bar(self, bar):
        self._update_macd_5m(bar['close'])
        self._update_atr(bar)

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

        self._process_inclusion(bar_data)
        new_bi = self._process_bi()

        if self._position != 0:
            self._update_trailing_stop(bar_data)

        if new_bi:
            self._check_signal(bar_data, new_bi)

    def _process_inclusion(self, new_bar):
        if not self._k_lines:
            self._k_lines.append(new_bar)
            return

        last = self._k_lines[-1]
        in_last = new_bar['high'] <= last['high'] and new_bar['low'] >= last['low']
        in_new = last['high'] <= new_bar['high'] and last['low'] >= new_bar['low']

        if in_last or in_new:
            if self._inclusion_dir == 0:
                self._inclusion_dir = 1

            merged = last.copy()
            merged['datetime'] = new_bar['datetime']
            merged['diff'] = new_bar['diff']
            merged['atr'] = new_bar['atr']
            merged['diff_15m'] = new_bar['diff_15m']
            merged['dea_15m'] = new_bar['dea_15m']

            if self._inclusion_dir == 1:
                merged['high'] = max(last['high'], new_bar['high'])
                merged['low'] = max(last['low'], new_bar['low'])
            else:
                merged['high'] = min(last['high'], new_bar['high'])
                merged['low'] = min(last['low'], new_bar['low'])

            self._k_lines[-1] = merged
        else:
            if new_bar['high'] > last['high'] and new_bar['low'] > last['low']:
                self._inclusion_dir = 1
            elif new_bar['high'] < last['high'] and new_bar['low'] < last['low']:
                self._inclusion_dir = -1

            self._k_lines.append(new_bar)

    def _process_bi(self):
        if len(self._k_lines) < 3:
            return None

        curr = self._k_lines[-1]
        mid = self._k_lines[-2]
        left = self._k_lines[-3]

        is_top = mid['high'] > left['high'] and mid['high'] > curr['high']
        is_bot = mid['low'] < left['low'] and mid['low'] < curr['low']

        cand = None
        if is_top:
            cand = {'type': 'top', 'price': mid['high'], 'idx': len(self._k_lines) - 2, 'data': mid}
        elif is_bot:
            cand = {'type': 'bottom', 'price': mid['low'], 'idx': len(self._k_lines) - 2, 'data': mid}

        if not cand:
            return None

        if not self._bi_points:
            self._bi_points.append(cand)
            return None

        last = self._bi_points[-1]

        if last['type'] == cand['type']:
            if last['type'] == 'top' and cand['price'] > last['price']:
                self._bi_points[-1] = cand
            elif last['type'] == 'bottom' and cand['price'] < last['price']:
                self._bi_points[-1] = cand
            return None
        else:
            if cand['idx'] - last['idx'] >= 4:
                self._bi_points.append(cand)
                return cand
            return None

    def _update_pivots(self):
        if len(self._bi_points) < 4:
            return

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
            new_pivot = {
                'zg': zg,
                'zd': zd,
                'start_bi_idx': len(self._bi_points) - 4,
                'end_bi_idx': len(self._bi_points) - 1
            }
            self._pivots.append(new_pivot)

    def _check_signal(self, curr_bar, new_bi):
        self._update_pivots()

        if len(self._bi_points) < 5:
            return

        p_now = self._bi_points[-1]
        p_last = self._bi_points[-2]
        p_prev = self._bi_points[-3]

        is_bull = self._prev_diff_15m > self._prev_dea_15m
        is_bear = self._prev_diff_15m < self._prev_dea_15m

        sig = None
        stop_base = 0.0
        trigger_price = 0.0

        if self._pivots:
            last_pivot = self._pivots[-1]

            if p_now['type'] == 'bottom':
                if p_now['price'] > last_pivot['zg']:
                    if p_last['price'] > last_pivot['zg']:
                        if last_pivot['end_bi_idx'] >= len(self._bi_points) - 6:
                            if is_bull:
                                sig = 'Buy'
                                trigger_price = p_now['data']['high']
                                stop_base = p_now['price']

            elif p_now['type'] == 'top':
                if p_now['price'] < last_pivot['zd']:
                    if p_last['price'] < last_pivot['zd']:
                        if last_pivot['end_bi_idx'] >= len(self._bi_points) - 6:
                            if is_bear:
                                sig = 'Sell'
                                trigger_price = p_now['data']['low']
                                stop_base = p_now['price']

        if not sig:
            if p_now['type'] == 'bottom':
                div = p_now['data']['diff'] > p_prev['data']['diff']
                if p_now['price'] > p_prev['price'] and div and is_bull:
                    sig = 'Buy'
                    trigger_price = p_now['data']['high']
                    stop_base = p_now['price']

            elif p_now['type'] == 'top':
                div = p_now['data']['diff'] < p_prev['data']['diff']
                if p_now['price'] < p_prev['price'] and div and is_bear:
                    sig = 'Sell'
                    trigger_price = p_now['data']['low']
                    stop_base = p_now['price']

        if sig and self.atr > 0:
            distance = abs(trigger_price - stop_base)
            if distance < 2.0 * self.atr:
                self._pending_signal = {
                    'type': sig,
                    'trigger_price': trigger_price,
                    'stop_base': stop_base
                }

    def _check_entry_1m(self, bar):
        signal = self._pending_signal
        if not signal:
            return

        if signal['type'] == 'Buy':
            if bar['low'] < signal['stop_base']:
                self._pending_signal = None
                return
            if bar['high'] > signal['trigger_price']:
                fill_price = max(signal['trigger_price'], bar['open'])
                if fill_price > bar['high']:
                    fill_price = bar['close']
                self._open_position(1, fill_price, signal['stop_base'])

        elif signal['type'] == 'Sell':
            if bar['high'] > signal['stop_base']:
                self._pending_signal = None
                return
            if bar['low'] < signal['trigger_price']:
                fill_price = min(signal['trigger_price'], bar['open'])
                if fill_price < bar['low']:
                    fill_price = bar['close']
                self._open_position(-1, fill_price, signal['stop_base'])

    def _open_position(self, direction, price, stop_base):
        self._position = direction
        self._entry_price = price
        self._stop_price = stop_base - 1 if direction == 1 else stop_base + 1
        self._trailing_active = False
        self._pending_signal = None

    def _check_stop_loss_1m(self, bar):
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

    def _update_trailing_stop(self, bar):
        if self.atr <= 0:
            return

        if self._position == 1:
            float_pnl = bar['close'] - self._entry_price
        else:
            float_pnl = self._entry_price - bar['close']

        if not self._trailing_active:
            if float_pnl > (self.atr * self.ACTIVATE_ATR):
                self._trailing_active = True

        if self._trailing_active:
            if self._position == 1:
                new_stop = bar['high'] - (self.atr * self.TRAIL_ATR)
                if new_stop > self._stop_price:
                    self._stop_price = new_stop
            else:
                new_stop = bar['low'] + (self.atr * self.TRAIL_ATR)
                if new_stop < self._stop_price:
                    self._stop_price = new_stop


# =============================================================================
# 主程序
# =============================================================================
def run_test(file_path: str, name: str) -> dict:
    """运行测试并返回结果."""
    df_raw = pd.read_csv(file_path)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

    # Pandas 测试器
    print(f"\n运行 Pandas 测试器...")
    pandas_tester = ChanPivotTesterPandas(df_raw)
    pandas_trades = pandas_tester.run()

    # CTA 测试器
    print(f"运行 CTA 测试器...")
    cta_tester = ChanPivotTesterCTA(df_raw)
    cta_trades = cta_tester.run()

    # 统计结果
    def calc_stats(trades_df, bi_count, pivot_count):
        if trades_df.empty:
            return {'pnl': 0, 'trades': 0, 'win_rate': 0, 'max_dd': 0, 'bi': bi_count, 'pivot': pivot_count}
        pnl = trades_df['pnl'].sum()
        total = len(trades_df)
        wins = len(trades_df[trades_df['pnl'] > 0])
        win_rate = wins / total * 100 if total > 0 else 0
        cumsum = trades_df['pnl'].cumsum()
        max_dd = abs((cumsum - cumsum.cummax()).min())
        return {'pnl': pnl, 'trades': total, 'win_rate': win_rate, 'max_dd': max_dd, 'bi': bi_count, 'pivot': pivot_count}

    pandas_stats = calc_stats(pandas_trades, len(pandas_tester.bi_points), len(pandas_tester.pivots))
    cta_stats = calc_stats(cta_trades, len(cta_tester._bi_points), len(cta_tester._pivots))

    print(f"\n{'='*60}")
    print(f"验证结果: {name}")
    print(f"{'='*60}")
    print(f"\n原始脚本 (Pandas):")
    print(f"  净利润: {pandas_stats['pnl']:.0f}")
    print(f"  交易数: {pandas_stats['trades']}")
    print(f"  胜率:   {pandas_stats['win_rate']:.2f}%")
    print(f"  最大回撤: {pandas_stats['max_dd']:.0f}")
    print(f"  笔数: {pandas_stats['bi']}, 中枢数: {pandas_stats['pivot']}")

    print(f"\nCTA 策略 (增量):")
    print(f"  净利润: {cta_stats['pnl']:.0f}")
    print(f"  交易数: {cta_stats['trades']}")
    print(f"  胜率:   {cta_stats['win_rate']:.2f}%")
    print(f"  最大回撤: {cta_stats['max_dd']:.0f}")
    print(f"  笔数: {cta_stats['bi']}, 中枢数: {cta_stats['pivot']}")

    # 一致性检查
    pnl_diff = abs(pandas_stats['pnl'] - cta_stats['pnl'])
    trades_match = pandas_stats['trades'] == cta_stats['trades']

    if pnl_diff < 10 and trades_match:
        print(f"\n[PASS] 结果一致")
        return {'status': 'PASS', 'pandas': pandas_stats, 'cta': cta_stats}
    else:
        print(f"\n[DIFF] 结果有差异 (PnL差: {pnl_diff:.0f}, 交易数: {pandas_stats['trades']} vs {cta_stats['trades']})")
        return {'status': 'DIFF', 'pandas': pandas_stats, 'cta': cta_stats}


if __name__ == "__main__":
    data_dir = Path("E:/work/quant/quantPlus/data/analyse")

    print("=" * 60)
    print("验证 CtaChanPivotStrategy 回测结果")
    print("=" * 60)

    results = {}

    # Dataset 1
    file1 = data_dir / "p2509_1min_202503-202508.csv"
    if file1.exists():
        results['dataset1'] = run_test(str(file1), "Dataset 1 (p2509)")

    # Dataset 2
    file2 = data_dir / "p2601_1min_202507-202512.csv"
    if file2.exists():
        results['dataset2'] = run_test(str(file2), "Dataset 2 (p2601)")

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {name}: {result['status']}")
