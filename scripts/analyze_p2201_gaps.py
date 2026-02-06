#!/usr/bin/env python
"""分析 p2201 的跳空情况，理解为什么 S26 反而伤害了它"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np

# 读取数据
csv_path = ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv"
df = pd.read_csv(csv_path, parse_dates=["datetime"])
df.sort_values("datetime", inplace=True)

# 聚合到 5 分钟
df['bar_5m'] = df['datetime'].dt.floor('5min')
df_5m = df.groupby('bar_5m').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
}).reset_index()
df_5m.rename(columns={'bar_5m': 'datetime'}, inplace=True)

# 计算 ATR
df_5m['prev_close'] = df_5m['close'].shift(1)
df_5m['tr'] = df_5m.apply(
    lambda x: max(
        x['high'] - x['low'],
        abs(x['high'] - x['prev_close']) if pd.notna(x['prev_close']) else 0,
        abs(x['low'] - x['prev_close']) if pd.notna(x['prev_close']) else 0
    ), axis=1
)
df_5m['atr'] = df_5m['tr'].rolling(window=14).mean()

# 计算跳空
df_5m['gap'] = abs(df_5m['open'] - df_5m['prev_close'])
df_5m['gap_atr_ratio'] = df_5m['gap'] / df_5m['atr']

# 找出极端跳空 (> 3 ATR)
extreme_gaps = df_5m[df_5m['gap_atr_ratio'] > 3.0].copy()

print("=" * 80)
print("p2201 极端跳空分析 (gap > 3×ATR)")
print("=" * 80)
print(f"\n数据区间: {df_5m['datetime'].min()} 到 {df_5m['datetime'].max()}")
print(f"总 5m bar 数: {len(df_5m)}")
print(f"极端跳空次数: {len(extreme_gaps)}")
print()

if len(extreme_gaps) > 0:
    print("极端跳空详情:")
    print("-" * 80)
    for idx, row in extreme_gaps.iterrows():
        # 找这个 bar 之后的走势
        future_bars = df_5m.loc[idx:idx+6]  # 当前 + 后 6 根
        if len(future_bars) > 1:
            future_change = future_bars.iloc[-1]['close'] - row['open']
            gap_direction = "up" if row['open'] > row['prev_close'] else "down"
            trend_direction = "up" if future_change > 0 else "down"
            same_direction = gap_direction == trend_direction
            
            print(f"\n{row['datetime']}:")
            print(f"  Gap: {row['gap']:.0f} pts ({row['gap_atr_ratio']:.1f}×ATR)")
            print(f"  Direction: {gap_direction} (open={row['open']:.0f}, prev_close={row['prev_close']:.0f})")
            print(f"  After 30min: {future_change:+.0f} pts ({'continues' if same_direction else 'reverses'})")
            print(f"  ATR: {row['atr']:.1f}")

# 分析跳空后的行情特点
print("\n" + "=" * 80)
print("跳空后行情分析")
print("=" * 80)

# 计算所有跳空（不只是极端的）> 1 ATR 的情况
moderate_gaps = df_5m[(df_5m['gap_atr_ratio'] > 1.0) & (df_5m['gap_atr_ratio'] <= 3.0)].copy()
print(f"\n中等跳空 (1-3×ATR) 次数: {len(moderate_gaps)}")

# 对于这些跳空，看看 S26 暂停信号会不会错过好机会
if len(extreme_gaps) > 0:
    profitable_after_gap = 0
    loss_after_gap = 0
    
    for idx, row in extreme_gaps.iterrows():
        # 检查跳空后 3 根 bar 的价格变化
        future_idx = df_5m.index.get_loc(idx)
        if future_idx + 3 < len(df_5m):
            bars_after = df_5m.iloc[future_idx:future_idx+4]
            # 如果是向上跳空，看做多是否盈利
            # 如果是向下跳空，看做空是否盈利
            gap_up = row['open'] > row['prev_close']
            
            # 假设在跳空后立即入场
            entry = row['open']
            # 看 3 根 bar 后的收盘价
            exit_price = bars_after.iloc[-1]['close']
            
            if gap_up:  # 跳空高开 → 可能的多头信号
                pnl = exit_price - entry
            else:  # 跳空低开 → 可能的空头信号
                pnl = entry - exit_price
            
            if pnl > 0:
                profitable_after_gap += 1
            else:
                loss_after_gap += 1
    
    print(f"\n极端跳空后 3 bar 内:")
    print(f"  顺势持仓盈利次数: {profitable_after_gap}")
    print(f"  顺势持仓亏损次数: {loss_after_gap}")
    if profitable_after_gap + loss_after_gap > 0:
        win_rate = profitable_after_gap / (profitable_after_gap + loss_after_gap) * 100
        print(f"  胜率: {win_rate:.1f}%")
        
        if win_rate > 50:
            print("\n结论: 极端跳空后顺势入场胜率较高，S26 暂停信号会错过好机会！")
        else:
            print("\n结论: 极端跳空后入场胜率较低，S26 暂停信号应该是保护性的")

# 分析 p2201 整体波动性
print("\n" + "=" * 80)
print("p2201 波动性分析")
print("=" * 80)
print(f"平均 ATR: {df_5m['atr'].mean():.1f}")
print(f"ATR 标准差: {df_5m['atr'].std():.1f}")
print(f"平均跳空: {df_5m['gap'].mean():.1f}")
print(f"跳空标准差: {df_5m['gap'].std():.1f}")

# 找出跳空最多的日期
df_5m['date'] = df_5m['datetime'].dt.date
daily_gaps = df_5m.groupby('date')['gap'].sum()
top_gap_days = daily_gaps.nlargest(5)
print("\n跳空最大的 5 天:")
for date, total_gap in top_gap_days.items():
    print(f"  {date}: 累计跳空 {total_gap:.0f} pts")
