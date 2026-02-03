import pandas as pd

trades = pd.read_csv(r"E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\experiments\20260203_0150_baseline\trades\p2201_trades.csv")
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
trades["month"] = trades["entry_time"].dt.to_period("M")

print("=== P2201 Monthly PnL ===")
monthly = trades.groupby("month").agg(pnl=("pnl","sum"), n=("pnl","size"), win=("pnl", lambda x: (x>0).mean()*100))
for m, r in monthly.iterrows():
    print(f"  {m}: pnl={r.pnl:.0f} n={int(r.n)} win={r.win:.0f}%")

print("\n=== Direction ===")
for d in [1, -1]:
    sub = trades[trades["direction"]==d]
    print(f"  {'Long' if d==1 else 'Short'}: pnl={sub.pnl.sum():.0f} n={len(sub)} win={(sub.pnl>0).mean()*100:.0f}%")

print("\n=== Long by signal ===")
longs = trades[trades["direction"]==1]
for st, grp in longs.groupby("signal_type"):
    print(f"  {st}: pnl={grp.pnl.sum():.0f} n={len(grp)} win={(grp.pnl>0).mean()*100:.0f}%")

print("\n=== Short by signal ===")
shorts = trades[trades["direction"]==-1]
for st, grp in shorts.groupby("signal_type"):
    print(f"  {st}: pnl={grp.pnl.sum():.0f} n={len(grp)} win={(grp.pnl>0).mean()*100:.0f}%")

# Check P2201 price trend
data = pd.read_csv(r"E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\data\analyse\p2201_1min_202108-202112.csv")
data["datetime"] = pd.to_datetime(data["datetime"])
data["month"] = data["datetime"].dt.to_period("M")
monthly_price = data.groupby("month").agg(open=("open","first"), close=("close","last"))
print("\n=== P2201 Price by Month ===")
for m, r in monthly_price.iterrows():
    change = r["close"] - r["open"]
    print(f"  {m}: {r['open']:.0f} -> {r['close']:.0f} ({change:+.0f})")

# Same for P2401
trades401 = pd.read_csv(r"E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\experiments\20260203_0150_baseline\trades\p2401_trades.csv")
trades401["entry_time"] = pd.to_datetime(trades401["entry_time"])
print("\n=== P2401 Direction ===")
for d in [1, -1]:
    sub = trades401[trades401["direction"]==d]
    print(f"  {'Long' if d==1 else 'Short'}: pnl={sub.pnl.sum():.0f} n={len(sub)} win={(sub.pnl>0).mean()*100:.0f}%")

data401 = pd.read_csv(r"E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus\data\analyse\p2401_1min_202308-202312.csv")
data401["datetime"] = pd.to_datetime(data401["datetime"])
data401["month"] = data401["datetime"].dt.to_period("M")
monthly401 = data401.groupby("month").agg(open=("open","first"), close=("close","last"))
print("\n=== P2401 Price by Month ===")
for m, r in monthly401.iterrows():
    change = r["close"] - r["open"]
    print(f"  {m}: {r['open']:.0f} -> {r['close']:.0f} ({change:+.0f})")
