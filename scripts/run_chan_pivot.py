"""
运行 chan0121Pivot 原始脚本获取基准结果.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 1. 回测引擎 (Backtest Engine)
# =============================================================================
class BacktestEngine:
    def __init__(self, df_1m, strategy):
        self.df_1m = df_1m.reset_index(drop=True)
        self.strategy = strategy
        self.trades = []

        self.position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0

        df_1m_idx = self.df_1m.set_index('datetime')
        df_1m_idx.index = pd.to_datetime(df_1m_idx.index)

        self.df_5m_full = df_1m_idx.resample('5min', label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        self.strategy.calculate_indicators(self.df_5m_full)

    def run(self):
        for i, row in self.df_1m.iterrows():
            current_time = pd.to_datetime(row['datetime'])

            if self.position != 0:
                self._check_exit(row)

            if self.position == 0 and self.strategy.pending_signal:
                self._check_entry(row)

            if current_time.minute % 5 == 0:
                if current_time in self.df_5m_full.index:
                    bar_5m = self.df_5m_full.loc[current_time]
                    self.strategy.on_bar_close(bar_5m)

                    if self.position != 0:
                        self.strategy.update_trailing_stop(bar_5m, self.position, self.entry_price)
                        self.stop_price = self.strategy.stop_price

    def _check_entry(self, row):
        signal = self.strategy.pending_signal
        if not signal: return

        if signal['type'] == 'Buy':
            if row['low'] < signal['stop_base']:
                self.strategy.pending_signal = None; return
            if row['high'] > signal['trigger_price']:
                fill = max(signal['trigger_price'], row['open'])
                if fill > row['high']: fill = row['close']
                self._open_position(1, fill, row['datetime'], signal['stop_base'])

        elif signal['type'] == 'Sell':
            if row['high'] > signal['stop_base']:
                self.strategy.pending_signal = None; return
            if row['low'] < signal['trigger_price']:
                fill = min(signal['trigger_price'], row['open'])
                if fill < row['low']: fill = row['close']
                self._open_position(-1, fill, row['datetime'], signal['stop_base'])

    def _open_position(self, direction, price, time, stop_base):
        self.position = direction
        self.entry_price = price
        self.stop_price = stop_base - 1 if direction == 1 else stop_base + 1
        self.strategy.stop_price = self.stop_price
        self.strategy.pending_signal = None
        self.strategy.trailing_active = False

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
            self.trades.append({'entry': self.entry_price, 'exit': exit_px, 'time': row['datetime'], 'pnl': pnl})
            self.position = 0
            self.strategy.position = 0

# =============================================================================
# 2. 策略基类
# =============================================================================
class ChanBaseStrategy:
    def __init__(self):
        self.name = "Base"
        self.pending_signal = None
        self.stop_price = 0
        self.trailing_active = False
        self.ACTIVATE_ATR = 1.5
        self.TRAIL_ATR = 3.0

    def calculate_indicators(self, df):
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        sig = macd.ewm(span=9, adjust=False).mean()
        df['diff'] = macd
        df['dea'] = sig

        df_15m = df.resample('15min', closed='right', label='right').agg({'close':'last'}).dropna()
        e1 = df_15m['close'].ewm(span=12, adjust=False).mean()
        e2 = df_15m['close'].ewm(span=26, adjust=False).mean()
        m = e1 - e2
        s = m.ewm(span=9, adjust=False).mean()
        aligned = pd.DataFrame({'diff': m, 'dea': s}).shift(1).reindex(df.index, method='ffill')
        df['diff_15m'] = aligned['diff']
        df['dea_15m'] = aligned['dea']

        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

    def update_trailing_stop(self, curr_bar, position, entry_price):
        atr = curr_bar['atr'] if not np.isnan(curr_bar['atr']) else 0
        pnl = (curr_bar['close'] - entry_price) * position
        if not self.trailing_active and pnl > self.ACTIVATE_ATR * atr:
            self.trailing_active = True
        if self.trailing_active:
            if position == 1:
                new = curr_bar['high'] - self.TRAIL_ATR * atr
                if new > self.stop_price: self.stop_price = new
            else:
                new = curr_bar['low'] + self.TRAIL_ATR * atr
                if new < self.stop_price: self.stop_price = new

# =============================================================================
# 3. 严格笔处理
# =============================================================================
class ChanImprovedStrategy(ChanBaseStrategy):
    def __init__(self):
        super().__init__()
        self.k_lines = []
        self.inclusion_dir = 0
        self.bi_points = []

    def on_bar_close(self, curr_bar):
        pass

    def _process_inclusion(self, new_bar):
        if not self.k_lines:
            self.k_lines.append(new_bar)
            return
        last = self.k_lines[-1]
        in_last = new_bar['high'] <= last['high'] and new_bar['low'] >= last['low']
        in_new = last['high'] <= new_bar['high'] and last['low'] >= new_bar['low']

        if in_last or in_new:
            if self.inclusion_dir == 0: self.inclusion_dir = 1
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
        if len(self.k_lines) < 3: return None
        curr = self.k_lines[-1]
        mid = self.k_lines[-2]
        left = self.k_lines[-3]
        is_top = mid['high'] > left['high'] and mid['high'] > curr['high']
        is_bot = mid['low'] < left['low'] and mid['low'] < curr['low']
        cand = None
        if is_top:
            cand = {'type':'top', 'price': mid['high'], 'idx': len(self.k_lines)-2, 'data': mid}
        elif is_bot:
            cand = {'type':'bottom', 'price': mid['low'], 'idx': len(self.k_lines)-2, 'data': mid}

        if not cand: return None
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

# =============================================================================
# 4. 中枢策略
# =============================================================================
class ChanPivotStrategy(ChanImprovedStrategy):
    def __init__(self):
        super().__init__()
        self.name = "Chan_Pivot_ZhongShu"
        self.pivots = []

    def on_bar_close(self, curr_bar):
        bar = {
            'high': curr_bar['high'], 'low': curr_bar['low'], 'time': curr_bar.name,
            'diff': curr_bar['diff'], 'atr': curr_bar['atr'],
            'diff_15m': curr_bar['diff_15m'], 'dea_15m': curr_bar['dea_15m']
        }

        self._process_inclusion(bar)
        new_bi = self._process_bi()

        if new_bi:
            self._check_signal(curr_bar, new_bi)

    def _update_pivots(self):
        if len(self.bi_points) < 4: return

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
                'start_bi_idx': len(self.bi_points)-4,
                'end_bi_idx': len(self.bi_points)-1
            }
            self.pivots.append(new_p)

    def _check_signal(self, curr_bar, new_bi):
        self._update_pivots()

        if len(self.bi_points) < 5: return

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
                            if is_bull: sig = 'Buy'

            elif p_now['type'] == 'top':
                if p_now['price'] < last_pivot['zd']:
                    if p_last['price'] < last_pivot['zd']:
                        if last_pivot['end_bi_idx'] >= len(self.bi_points) - 6:
                            if is_bear: sig = 'Sell'

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
                self.pending_signal = {'type':'Buy', 'trigger_price': trig, 'stop_base': p_now['price']}
        elif sig == 'Sell':
            trig = p_now['data']['low']
            if (p_now['price'] - trig) < 2.0 * atr:
                self.pending_signal = {'type':'Sell', 'trigger_price': trig, 'stop_base': p_now['price']}

# =============================================================================
# 5. 主程序
# =============================================================================
def run_backtest(csv_file: str, name: str):
    """运行回测."""
    df_raw = pd.read_csv(csv_file)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    strategy = ChanPivotStrategy()
    engine = BacktestEngine(df_raw, strategy)
    engine.run()

    trades = pd.DataFrame(engine.trades)

    print(f"\n{'='*60}")
    print(f"Chan Pivot 策略回测结果: {name}")
    print(f"{'='*60}")

    if not trades.empty:
        total_pnl = trades['pnl'].sum()
        win_trades = len(trades[trades['pnl'] > 0])
        total_trades = len(trades)
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

        # 计算最大回撤
        cumsum = trades['pnl'].cumsum()
        max_dd = (cumsum - cumsum.cummax()).min()

        print(f"交易数: {total_trades}")
        print(f"净利润: {total_pnl:.0f}")
        print(f"胜率:   {win_rate:.2f}%")
        print(f"最大回撤: {abs(max_dd):.0f}")
        print(f"笔数: {len(strategy.bi_points)}")
        print(f"中枢数: {len(strategy.pivots)}")

        return {
            'trades': total_trades,
            'pnl': total_pnl,
            'win_rate': win_rate,
            'max_dd': abs(max_dd),
            'bi_count': len(strategy.bi_points),
            'pivot_count': len(strategy.pivots)
        }
    else:
        print("无交易信号")
        return None


if __name__ == "__main__":
    data_dir = Path("E:/work/quant/quantPlus/data/analyse")

    # Dataset 1
    file1 = data_dir / "p2509_1min_202503-202508.csv"
    if file1.exists():
        result1 = run_backtest(str(file1), "Dataset 1 (p2509)")

    # Dataset 2
    file2 = data_dir / "p2601_1min_202507-202512.csv"
    if file2.exists():
        result2 = run_backtest(str(file2), "Dataset 2 (p2601)")
