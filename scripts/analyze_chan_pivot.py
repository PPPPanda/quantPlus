"""
缠论中枢策略分析 - 结合缠论108课进行诊断.

运行方式：
    uv run python scripts/analyze_chan_pivot.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AnalyzeParams:
    """分析参数."""
    atr_trailing_mult: float = 3.0
    atr_activate_mult: float = 1.5
    atr_entry_filter: float = 2.0
    min_bi_gap: int = 4
    pivot_valid_range: int = 6


class ChanPivotAnalyzer:
    """缠论中枢策略分析器 - 带详细日志."""

    def __init__(self, df_1m: pd.DataFrame, params: AnalyzeParams):
        self.df_1m = df_1m.reset_index(drop=True)
        self.params = params
        self.trades = []
        self.signals = []  # 记录所有信号
        self.position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.trailing_active = False

        self.k_lines = []
        self.inclusion_dir = 0
        self.bi_points = []
        self.pivots = []
        self.pending_signal = None

        df_1m_idx = self.df_1m.set_index('datetime')
        df_1m_idx.index = pd.to_datetime(df_1m_idx.index)

        self.df_5m = df_1m_idx.resample('5min', label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        self._calc_indicators()

    def _calc_indicators(self):
        df = self.df_5m
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        sig = macd.ewm(span=9, adjust=False).mean()
        df['diff'] = macd
        df['dea'] = sig

        df_15m = df.resample('15min', closed='right', label='right').agg({'close': 'last'}).dropna()
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

    def run(self) -> dict:
        for i, row in self.df_1m.iterrows():
            current_time = pd.to_datetime(row['datetime'])

            if self.position != 0:
                self._check_exit(row)

            if self.position == 0 and self.pending_signal:
                self._check_entry(row)

            if current_time.minute % 5 == 0:
                if current_time in self.df_5m.index:
                    bar_5m = self.df_5m.loc[current_time]
                    self._on_bar_close(bar_5m)
                    if self.position != 0:
                        self._update_trailing_stop(bar_5m)

        return self._analyze_results()

    def _analyze_results(self) -> dict:
        """详细分析回测结果."""
        result = {
            'trades': [],
            'signals': self.signals,
            'bi_count': len(self.bi_points),
            'pivot_count': len(self.pivots),
            'metrics': {}
        }

        if not self.trades:
            result['metrics'] = {
                'total_trades': 0, 'pnl': 0, 'win_rate': 0, 'max_dd': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0
            }
            return result

        trades_df = pd.DataFrame(self.trades)
        result['trades'] = self.trades

        total_pnl = trades_df['pnl'].sum()
        win_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] <= 0]

        total_trades = len(trades_df)
        win_count = len(win_trades)
        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0

        cumsum = trades_df['pnl'].cumsum()
        max_dd = (cumsum - cumsum.cummax()).min()

        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0

        profits = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
        losses = abs(loss_trades['pnl'].sum()) if len(loss_trades) > 0 else 0
        profit_factor = profits / losses if losses > 0 else float('inf')

        # 信号类型统计
        signal_types = {}
        for sig in self.signals:
            sig_type = sig.get('signal_type', 'unknown')
            if sig_type not in signal_types:
                signal_types[sig_type] = {'count': 0, 'triggered': 0}
            signal_types[sig_type]['count'] += 1
            if sig.get('triggered'):
                signal_types[sig_type]['triggered'] += 1

        result['metrics'] = {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': len(loss_trades),
            'pnl': total_pnl,
            'win_rate': win_rate,
            'max_dd': abs(max_dd),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'signal_types': signal_types
        }

        return result

    def _check_entry(self, row):
        signal = self.pending_signal
        if not signal:
            return
        if signal['type'] == 'Buy':
            if row['low'] < signal['stop_base']:
                # 信号失效
                for s in self.signals:
                    if s.get('time') == signal.get('signal_time'):
                        s['invalidated'] = True
                self.pending_signal = None
                return
            if row['high'] > signal['trigger_price']:
                fill = max(signal['trigger_price'], row['open'])
                if fill > row['high']:
                    fill = row['close']
                self._open_position(1, fill, signal['stop_base'], row['datetime'], signal)
        elif signal['type'] == 'Sell':
            if row['high'] > signal['stop_base']:
                for s in self.signals:
                    if s.get('time') == signal.get('signal_time'):
                        s['invalidated'] = True
                self.pending_signal = None
                return
            if row['low'] < signal['trigger_price']:
                fill = min(signal['trigger_price'], row['open'])
                if fill < row['low']:
                    fill = row['close']
                self._open_position(-1, fill, signal['stop_base'], row['datetime'], signal)

    def _open_position(self, direction, price, stop_base, time, signal):
        self.position = direction
        self.entry_price = price
        self.entry_time = time
        self.stop_price = stop_base - 1 if direction == 1 else stop_base + 1
        self.pending_signal = None
        self.trailing_active = False

        # 更新信号状态
        for s in self.signals:
            if s.get('time') == signal.get('signal_time'):
                s['triggered'] = True
                s['entry_price'] = price
                s['entry_time'] = time

    def _check_exit(self, row):
        hit = False
        exit_px = 0
        exit_type = ''
        if self.position == 1:
            if row['low'] <= self.stop_price:
                hit = True
                exit_px = row['open'] if row['open'] < self.stop_price else self.stop_price
                exit_type = 'trail_stop' if self.trailing_active else 'hard_stop'
        elif self.position == -1:
            if row['high'] >= self.stop_price:
                hit = True
                exit_px = row['open'] if row['open'] > self.stop_price else self.stop_price
                exit_type = 'trail_stop' if self.trailing_active else 'hard_stop'
        if hit:
            pnl = (exit_px - self.entry_price) * self.position
            self.trades.append({
                'entry_time': self.entry_time,
                'exit_time': row['datetime'],
                'direction': 'long' if self.position == 1 else 'short',
                'entry_price': self.entry_price,
                'exit_price': exit_px,
                'pnl': pnl,
                'exit_type': exit_type
            })
            self.position = 0

    def _update_trailing_stop(self, curr_bar):
        atr = curr_bar['atr'] if not np.isnan(curr_bar['atr']) else 0
        pnl = (curr_bar['close'] - self.entry_price) * self.position
        if not self.trailing_active and pnl > self.params.atr_activate_mult * atr:
            self.trailing_active = True
        if self.trailing_active:
            if self.position == 1:
                new = curr_bar['high'] - self.params.atr_trailing_mult * atr
                if new > self.stop_price:
                    self.stop_price = new
            else:
                new = curr_bar['low'] + self.params.atr_trailing_mult * atr
                if new < self.stop_price:
                    self.stop_price = new

    def _on_bar_close(self, curr_bar):
        bar = {
            'high': curr_bar['high'], 'low': curr_bar['low'], 'time': curr_bar.name,
            'diff': curr_bar['diff'], 'atr': curr_bar['atr'],
            'diff_15m': curr_bar['diff_15m'], 'dea_15m': curr_bar['dea_15m']
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

    def _process_bi(self) -> Optional[dict]:
        if len(self.k_lines) < 3:
            return None
        curr, mid, left = self.k_lines[-1], self.k_lines[-2], self.k_lines[-3]
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
            if cand['idx'] - last['idx'] >= self.params.min_bi_gap:
                self.bi_points.append(cand)
                return cand
        return None

    def _update_pivots(self):
        if len(self.bi_points) < 4:
            return
        b0, b1, b2, b3 = self.bi_points[-4], self.bi_points[-3], self.bi_points[-2], self.bi_points[-1]
        r1 = (min(b0['price'], b1['price']), max(b0['price'], b1['price']))
        r2 = (min(b1['price'], b2['price']), max(b1['price'], b2['price']))
        r3 = (min(b2['price'], b3['price']), max(b2['price'], b3['price']))
        zg = min(r1[1], r2[1], r3[1])
        zd = max(r1[0], r2[0], r3[0])
        if zg > zd:
            self.pivots.append({
                'zg': zg, 'zd': zd, 'start_bi_idx': len(self.bi_points) - 4,
                'end_bi_idx': len(self.bi_points) - 1
            })

    def _check_signal(self, curr_bar, new_bi):
        self._update_pivots()
        if len(self.bi_points) < 5:
            return
        p_now, p_last, p_prev = self.bi_points[-1], self.bi_points[-2], self.bi_points[-3]
        is_bull = curr_bar['diff_15m'] > curr_bar['dea_15m']
        is_bear = curr_bar['diff_15m'] < curr_bar['dea_15m']
        sig = None
        signal_type = None

        # 3B/3S 中枢信号
        if self.pivots:
            last_pivot = self.pivots[-1]
            if p_now['type'] == 'bottom':
                if p_now['price'] > last_pivot['zg'] and p_last['price'] > last_pivot['zg']:
                    if last_pivot['end_bi_idx'] >= len(self.bi_points) - self.params.pivot_valid_range:
                        if is_bull:
                            sig = 'Buy'
                            signal_type = '3B'
            elif p_now['type'] == 'top':
                if p_now['price'] < last_pivot['zd'] and p_last['price'] < last_pivot['zd']:
                    if last_pivot['end_bi_idx'] >= len(self.bi_points) - self.params.pivot_valid_range:
                        if is_bear:
                            sig = 'Sell'
                            signal_type = '3S'

        # 2B/2S 辅助信号
        if not sig:
            if p_now['type'] == 'bottom':
                div = p_now['data']['diff'] > p_prev['data']['diff']
                if p_now['price'] > p_prev['price'] and div and is_bull:
                    sig = 'Buy'
                    signal_type = '2B'
            elif p_now['type'] == 'top':
                div = p_now['data']['diff'] < p_prev['data']['diff']
                if p_now['price'] < p_prev['price'] and div and is_bear:
                    sig = 'Sell'
                    signal_type = '2S'

        atr = curr_bar['atr']
        if sig == 'Buy':
            trig = p_now['data']['high']
            if (trig - p_now['price']) < self.params.atr_entry_filter * atr:
                self.pending_signal = {
                    'type': 'Buy', 'trigger_price': trig, 'stop_base': p_now['price'],
                    'signal_time': curr_bar.name
                }
                self.signals.append({
                    'time': curr_bar.name,
                    'signal_type': signal_type,
                    'direction': 'Buy',
                    'trigger_price': trig,
                    'stop_base': p_now['price'],
                    'is_bull': is_bull,
                    'triggered': False
                })
        elif sig == 'Sell':
            trig = p_now['data']['low']
            if (p_now['price'] - trig) < self.params.atr_entry_filter * atr:
                self.pending_signal = {
                    'type': 'Sell', 'trigger_price': trig, 'stop_base': p_now['price'],
                    'signal_time': curr_bar.name
                }
                self.signals.append({
                    'time': curr_bar.name,
                    'signal_type': signal_type,
                    'direction': 'Sell',
                    'trigger_price': trig,
                    'stop_base': p_now['price'],
                    'is_bear': is_bear,
                    'triggered': False
                })


def run_analysis():
    """运行分析."""
    data_dir = Path("E:/work/quant/quantPlus/data/analyse")

    datasets = {
        'p2505': data_dir / "p2505_1min_202501-202504.csv",
        'p2509': data_dir / "p2509_1min_202503-202508.csv",
        'p2601': data_dir / "p2601_1min_202507-202512.csv",
    }

    params = AnalyzeParams()

    print("=" * 80)
    print("缠论中枢策略分析 - 结合缠论108课")
    print("=" * 80)
    print(f"\n当前参数:")
    print(f"  atr_trailing_mult: {params.atr_trailing_mult}")
    print(f"  atr_activate_mult: {params.atr_activate_mult}")
    print(f"  atr_entry_filter:  {params.atr_entry_filter}")
    print(f"  min_bi_gap:        {params.min_bi_gap}")
    print(f"  pivot_valid_range: {params.pivot_valid_range}")

    all_results = {}

    for name, path in datasets.items():
        if not path.exists():
            print(f"\n警告: {name} 数据文件不存在")
            continue

        print(f"\n{'=' * 80}")
        print(f"数据集: {name}")
        print("=" * 80)

        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        print(f"数据量: {len(df)} 条1分钟K线")

        analyzer = ChanPivotAnalyzer(df, params)
        result = analyzer.run()
        all_results[name] = result

        m = result['metrics']
        print(f"\n基础指标:")
        print(f"  笔数量:     {result['bi_count']}")
        print(f"  中枢数量:   {result['pivot_count']}")
        print(f"  信号总数:   {len(result['signals'])}")
        print(f"  成交数量:   {m['total_trades']}")

        print(f"\n收益指标:")
        print(f"  净利润:     {m['pnl']:.0f}")
        print(f"  胜率:       {m['win_rate']:.1f}%")
        print(f"  最大回撤:   {m['max_dd']:.0f}")
        print(f"  盈亏比:     {m['profit_factor']:.2f}")

        if m['total_trades'] > 0:
            print(f"\n交易分析:")
            print(f"  盈利次数:   {m['win_count']}")
            print(f"  亏损次数:   {m['loss_count']}")
            print(f"  平均盈利:   {m['avg_win']:.1f}")
            print(f"  平均亏损:   {m['avg_loss']:.1f}")

        if 'signal_types' in m and m['signal_types']:
            print(f"\n信号类型分布:")
            for sig_type, stats in m['signal_types'].items():
                trigger_rate = stats['triggered'] / stats['count'] * 100 if stats['count'] > 0 else 0
                print(f"  {sig_type}: 生成{stats['count']}次, 触发{stats['triggered']}次 ({trigger_rate:.1f}%)")

        # 出场类型统计
        if result['trades']:
            exit_types = {}
            for t in result['trades']:
                et = t.get('exit_type', 'unknown')
                if et not in exit_types:
                    exit_types[et] = {'count': 0, 'pnl': 0}
                exit_types[et]['count'] += 1
                exit_types[et]['pnl'] += t['pnl']

            print(f"\n出场类型分布:")
            for et, stats in exit_types.items():
                avg = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
                print(f"  {et}: {stats['count']}次, 总PnL={stats['pnl']:.0f}, 均PnL={avg:.1f}")

    # 综合分析
    print("\n" + "=" * 80)
    print("缠论108课视角分析")
    print("=" * 80)

    total_pnl = sum(r['metrics']['pnl'] for r in all_results.values())
    total_trades = sum(r['metrics']['total_trades'] for r in all_results.values())
    avg_win_rate = np.mean([r['metrics']['win_rate'] for r in all_results.values()])

    print(f"\n综合表现:")
    print(f"  三数据集总收益: {total_pnl:.0f}")
    print(f"  总交易次数:     {total_trades}")
    print(f"  平均胜率:       {avg_win_rate:.1f}%")

    print("""
缠论108课核心要点对照分析:

1. 【第17课·中枢定义】
   - 当前实现: 3笔重叠区域作为中枢
   - 问题: 仅检测重叠,未区分中枢级别和延伸
   - 优化方向: 增加中枢级别判断,区分本级别和次级别中枢

2. 【第29课·买卖点定义】
   - 3B买点(离开中枢后第一个不创新低的底分型):
     * 当前实现: p_now > ZG && p_last > ZG
     * 问题: 未严格验证"不创新低"条件
   - 2B买点(趋势背驰):
     * 当前实现: 底抬高 + MACD背驰
     * 问题: 背驰判断仅用DIFF单值,未用面积

3. 【第37课·中枢震荡与突破】
   - 问题: 未区分中枢震荡和真正突破
   - 优化: 增加突破确认机制(如回踩不破)

4. 【第62课·分型确认】
   - 当前实现: 简单三K线高低点比较
   - 问题: 未考虑成交量配合
   - 优化: 增加量能确认

5. 【风控系统分析】
   - 硬止损: 基于P1(分型极值)设置
   - 移动止损: 1.5 ATR激活, 3 ATR跟踪
   - 问题: 止损过于机械,未考虑走势完成度
""")

    # 优化建议
    print("=" * 80)
    print("参数优化建议")
    print("=" * 80)

    # 根据数据分析给出建议
    hard_stop_pnl = sum(
        sum(t['pnl'] for t in r['trades'] if t.get('exit_type') == 'hard_stop')
        for r in all_results.values()
    )
    trail_stop_pnl = sum(
        sum(t['pnl'] for t in r['trades'] if t.get('exit_type') == 'trail_stop')
        for r in all_results.values()
    )

    print(f"\n止损分析:")
    print(f"  硬止损总PnL: {hard_stop_pnl:.0f}")
    print(f"  移动止损总PnL: {trail_stop_pnl:.0f}")

    if hard_stop_pnl < 0 and abs(hard_stop_pnl) > total_pnl * 0.5:
        print("  建议: 硬止损损失过大,考虑:")
        print("    - 增大 min_bi_gap 以过滤噪音笔")
        print("    - 增加入场过滤条件")

    if trail_stop_pnl < hard_stop_pnl:
        print("  建议: 移动止损表现不佳,考虑:")
        print("    - 减小 atr_trailing_mult 以更紧密跟踪")
        print("    - 减小 atr_activate_mult 以更早激活")

    print("""
具体参数调整建议:

1. 若胜率低(<40%):
   - 增大 min_bi_gap: 4 -> 5 (更严格的笔构建)
   - 增大 pivot_valid_range: 6 -> 8 (放宽中枢有效期)
   - 增加趋势过滤强度

2. 若回撤大:
   - 减小 atr_trailing_mult: 3.0 -> 2.5 (更紧止损)
   - 减小 atr_activate_mult: 1.5 -> 1.0 (更早激活)

3. 若交易次数少:
   - 减小 min_bi_gap: 4 -> 3
   - 增大 atr_entry_filter: 2.0 -> 2.5

4. 若盈亏比低:
   - 优化信号选择,优先3B/3S信号
   - 增加趋势强度过滤
""")

    return all_results


if __name__ == "__main__":
    run_analysis()
