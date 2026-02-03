"""trim_q02_and_rename_v2.py

从原始 export.csv（含持仓量/成交额）重新处理，q=0.2 首尾分割，
输出到 data/analyse/，命名与 p2505 一致。

原始列（gb18030）：代码,名称,日期,开盘价(元),最高价(元),最低价(元),收盘价(元),结算价,涨跌幅,成交额(百万),成交量,持仓量
"""

from pathlib import Path
import pandas as pd

TEMP_DIR = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/temp/data")
OUT_DIR = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")

CONTRACTS = {
    "P2201": "P2201.DCE._export.csv",
    "P2205": "P2205.DCE._export.csv",
    "P2401": "P2401.DCE._export.csv",
    "P2405": "P2405.DCE._export.csv",
}

TARGET_COLS = ["datetime", "open", "high", "low", "close", "volume", "open_interest", "turnover"]


def load_export(contract: str, filename: str) -> pd.DataFrame:
    fp = TEMP_DIR / filename
    df = pd.read_csv(fp, encoding="gb18030")
    cols = df.columns.tolist()
    # Map Chinese column names to English
    col_map = {}
    for c in cols:
        cl = c.strip()
        if "日期" in cl or "时间" in cl:
            col_map[c] = "datetime"
        elif "开盘" in cl:
            col_map[c] = "open"
        elif "最高" in cl:
            col_map[c] = "high"
        elif "最低" in cl:
            col_map[c] = "low"
        elif "收盘" in cl:
            col_map[c] = "close"
        elif "成交量" in cl:
            col_map[c] = "volume"
        elif "持仓" in cl:
            col_map[c] = "open_interest"
        elif "成交额" in cl:
            col_map[c] = "turnover"

    df = df.rename(columns=col_map)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # turnover: 原始单位是"百万"，转换为元
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce").fillna(0) * 1_000_000

    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Fill missing columns
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = 0

    return df[TARGET_COLS]


def trim_head_tail(df: pd.DataFrame, q: float = 0.2):
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    daily_vol = df.groupby("date")["volume"].sum()
    threshold = daily_vol.quantile(q)

    above = daily_vol[daily_vol >= threshold]
    if above.empty:
        return df.drop(columns=["date"]), df["date"].min(), df["date"].max(), threshold, 0

    start_date = above.index.min()
    end_date = above.index.max()

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    trimmed = df[mask].drop(columns=["date"])

    active_daily = daily_vol[(daily_vol.index >= start_date) & (daily_vol.index <= end_date)]
    low_days = int((active_daily < threshold).sum())

    return trimmed, start_date, end_date, threshold, low_days


def make_filename(contract: str, df_trimmed: pd.DataFrame) -> str:
    start_month = df_trimmed["datetime"].min().strftime("%Y%m")
    end_month = df_trimmed["datetime"].max().strftime("%Y%m")
    return f"{contract.lower()}_1min_{start_month}-{end_month}.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for contract, fname in CONTRACTS.items():
        print(f"\n{'='*60}")
        print(f"Processing {contract} from {fname}")
        df = load_export(contract, fname)
        print(f"  Raw: {len(df)} rows, {df['datetime'].min()} -> {df['datetime'].max()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  open_interest range: {df['open_interest'].min():.0f} - {df['open_interest'].max():.0f}")
        print(f"  turnover range: {df['turnover'].min():.0f} - {df['turnover'].max():.0f}")

        trimmed, start, end, thresh, low_inside = trim_head_tail(df, q=0.2)
        print(f"  Threshold (q=0.2): {thresh:,.0f}")
        print(f"  Active range: {start} -> {end}")
        print(f"  Low-volume days inside: {low_inside}")
        print(f"  Trimmed: {len(trimmed)} rows")

        out_name = make_filename(contract, trimmed)
        out_path = OUT_DIR / out_name
        trimmed.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
