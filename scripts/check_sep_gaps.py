import pandas as pd
from pathlib import Path

csv_path = Path('data/analyse/wind/p2201_1min_202108-202112.csv')
df = pd.read_csv(csv_path, parse_dates=['datetime'])
df = df[(df['datetime'] >= '2021-09-01') & (df['datetime'] <= '2021-09-05')]
df['bar_5m'] = df['datetime'].dt.floor('5min')
df_5m = df.groupby('bar_5m').agg({'open': 'first', 'close': 'last'}).reset_index()
df_5m['prev_close'] = df_5m['close'].shift(1)
df_5m['gap'] = abs(df_5m['open'] - df_5m['prev_close'])
df_5m = df_5m.dropna()

# 计算 ATR (简化)
df_5m['atr'] = df_5m['close'].rolling(14).apply(lambda x: x.diff().abs().mean())
df_5m['gap_atr_ratio'] = df_5m['gap'] / df_5m['atr']

print("2021-09-01 ~ 2021-09-05 跳空情况 (>30pts):")
print("-" * 60)
big_gaps = df_5m[df_5m['gap'] > 30]
for _, row in big_gaps.iterrows():
    ratio = row['gap_atr_ratio'] if pd.notna(row['gap_atr_ratio']) else 0
    print(f"{row['bar_5m']} | gap={row['gap']:.0f}pts | ratio={ratio:.1f}xATR")

print("\n2021-09-02 21:00 附近的 5m bars:")
print("-" * 60)
near_signal = df_5m[(df_5m['bar_5m'] >= '2021-09-02 20:00') & (df_5m['bar_5m'] <= '2021-09-02 22:00')]
for _, row in near_signal.iterrows():
    gap = row['gap']
    ratio = row['gap_atr_ratio'] if pd.notna(row['gap_atr_ratio']) else 0
    marker = " <-- S26 blocked signal here" if row['bar_5m'].hour == 21 and row['bar_5m'].minute < 15 else ""
    print(f"{row['bar_5m']} | open={row['open']:.0f} | gap={gap:.0f} | {ratio:.1f}xATR{marker}")
