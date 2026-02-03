"""trim_q02_and_rename.py

对 P2201/P2205/P2401/P2405 的 1min 数据做 q=0.2 首尾分割，输出到 data/analyse/ 并按
p2505_1min_202501-202504.csv 格式命名。

首尾分割逻辑：
  1. 按日统计 volume 总和
  2. threshold = daily_volume.quantile(0.2)
  3. start = 第一个 >= threshold 的日期
  4. end = 最后一个 >= threshold 的日期
  5. 保留 [start, end] 之间所有数据（中间低成交日不剔除）

命名格式：p{合约号}_1min_{YYYYMM起}-{YYYYMM止}.csv
列格式对齐 p2505（含 open_interest/turnover 列，没有则填0）
"""

from pathlib import Path
import pandas as pd

DERIVED_DIR = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/temp/data/derived")
OUT_DIR = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")

CONTRACTS = ["P2201", "P2205", "P2401", "P2405"]

TARGET_COLS = ["datetime", "open", "high", "low", "close", "volume", "open_interest", "turnover"]


def load(contract: str) -> pd.DataFrame:
    fp = DERIVED_DIR / f"{contract}_1min_like_old_ALL.csv"
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def trim_head_tail(df: pd.DataFrame, q: float = 0.2):
    df["date"] = df["datetime"].dt.date
    daily_vol = df.groupby("date")["volume"].sum()
    threshold = daily_vol.quantile(q)

    above = daily_vol[daily_vol >= threshold]
    if above.empty:
        print("  WARNING: no days above threshold, returning all data")
        return df.drop(columns=["date"]), df["date"].min(), df["date"].max(), threshold, 0

    start_date = above.index.min()
    end_date = above.index.max()

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    trimmed = df[mask].copy()

    # count low-volume days inside the active range
    active_daily = daily_vol[(daily_vol.index >= start_date) & (daily_vol.index <= end_date)]
    low_days = int((active_daily < threshold).sum())

    trimmed = trimmed.drop(columns=["date"])
    return trimmed, start_date, end_date, threshold, low_days


def make_filename(contract: str, df_trimmed: pd.DataFrame) -> str:
    start_month = df_trimmed["datetime"].min().strftime("%Y%m")
    end_month = df_trimmed["datetime"].max().strftime("%Y%m")
    return f"{contract.lower()}_1min_{start_month}-{end_month}.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for c in CONTRACTS:
        print(f"\n{'='*60}")
        print(f"Processing {c}")
        df = load(c)
        print(f"  Raw: {len(df)} rows, {df['datetime'].min()} -> {df['datetime'].max()}")

        trimmed, start, end, thresh, low_inside = trim_head_tail(df, q=0.2)
        print(f"  Threshold (q=0.2): {thresh:,.0f}")
        print(f"  Active range: {start} -> {end}")
        print(f"  Low-volume days inside range: {low_inside}")
        print(f"  Trimmed: {len(trimmed)} rows")

        # ensure target columns
        for col in ["open_interest", "turnover"]:
            if col not in trimmed.columns:
                trimmed[col] = 0

        trimmed = trimmed[TARGET_COLS]

        fname = make_filename(c, trimmed)
        out_path = OUT_DIR / fname
        trimmed.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(f"  Filename: {fname}")


if __name__ == "__main__":
    main()
