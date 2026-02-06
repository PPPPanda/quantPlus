#!/usr/bin/env python3
"""
Debug S27 gap detection - check if it's actually triggering.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

# Load p2209 data
csv_path = ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"
df = pd.read_csv(csv_path)
df_norm = normalize_1m_bars(df, PALM_OIL_SESSIONS)
df_norm['datetime'] = pd.to_datetime(df_norm['datetime'])

# Aggregate to 5m bars
df_norm['dt_5m'] = df_norm['datetime'].dt.floor('5min')
bars_5m = df_norm.groupby('dt_5m').agg(
    open=('open', 'first'),
    high=('high', 'max'),
    low=('low', 'min'),
    close=('close', 'last'),
).reset_index()

print(f"Total 5m bars: {len(bars_5m)}")

# Calculate TR and ATR (14-period)
bars_5m['prev_close'] = bars_5m['close'].shift(1)
bars_5m['high_low'] = bars_5m['high'] - bars_5m['low']
bars_5m['high_close'] = (bars_5m['high'] - bars_5m['prev_close']).abs()
bars_5m['low_close'] = (bars_5m['low'] - bars_5m['prev_close']).abs()
bars_5m['tr'] = bars_5m[['high_low', 'high_close', 'low_close']].max(axis=1)
bars_5m['atr_14'] = bars_5m['tr'].rolling(14).mean()

print(f"\n5m ATR statistics:")
print(f"  Mean ATR: {bars_5m['atr_14'].mean():.1f}")
print(f"  Median ATR: {bars_5m['atr_14'].median():.1f}")
print(f"  Min ATR: {bars_5m['atr_14'].min():.1f}")
print(f"  Max ATR: {bars_5m['atr_14'].max():.1f}")

# Detect session gaps
bars_5m['date'] = bars_5m['dt_5m'].dt.date
bars_5m['prev_date'] = bars_5m['date'].shift(1)
bars_5m['is_new_session'] = bars_5m['date'] != bars_5m['prev_date']

# Get previous session close
prev_session_close = {}
for i, row in bars_5m.iterrows():
    if row['is_new_session'] and i > 0:
        prev_session_close[i] = bars_5m.loc[i-1, 'close']

bars_5m['prev_session_close'] = bars_5m.index.map(lambda x: prev_session_close.get(x, None))

# Calculate gaps
bars_5m['gap'] = (bars_5m['open'] - bars_5m['prev_session_close']).abs()

# Filter new session bars with gaps
new_sessions = bars_5m[bars_5m['is_new_session'] & bars_5m['gap'].notna()].copy()

print(f"\nSession gaps detected: {len(new_sessions)}")

# Check which gaps would trigger S27
new_sessions['atr_at_gap'] = new_sessions['atr_14'].fillna(20)  # default 20 if early
new_sessions['threshold'] = new_sessions['atr_at_gap'] * 1.5
new_sessions['would_trigger'] = new_sessions['gap'] > new_sessions['threshold']

print("\nTop 10 session gaps:")
print("-" * 80)
for _, row in new_sessions.nlargest(10, 'gap').iterrows():
    trigger_mark = "[TRIGGER]" if row['would_trigger'] else ""
    print(f"  {row['date']}: gap={row['gap']:.0f}, ATR={row['atr_at_gap']:.1f}, threshold={row['threshold']:.1f} {trigger_mark}")

triggers = new_sessions[new_sessions['would_trigger']]
print(f"\nTotal S27 triggers: {len(triggers)}")

# Check Labor Day specifically (2022-05-05)
labor_day = new_sessions[new_sessions['date'].astype(str) == '2022-05-05']
if not labor_day.empty:
    row = labor_day.iloc[0]
    print(f"\n[LABOR DAY 2022-05-05]")
    print(f"  Gap: {row['gap']:.0f}")
    print(f"  ATR at gap: {row['atr_at_gap']:.1f}")
    print(f"  Threshold (1.5*ATR): {row['threshold']:.1f}")
    print(f"  Would trigger S27: {row['would_trigger']}")
