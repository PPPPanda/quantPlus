"""
iter2 Round 2 网格搜索：
  C01: 15m MACD 零轴过滤 (macd_zero_filter)
  C02: atr_entry_filter 参数扫描
  C03: 两者组合

用法：cd quantPlus && .venv/Scripts/python.exe scripts/grid_iter2_r2.py
"""
from __future__ import annotations
import sys, json, time, itertools
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logging
logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

import pandas as pd
from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

# ---------- 全量合约 ----------
WIND_DIR = ROOT / "data" / "analyse" / "wind"
ALL_CONTRACTS = []
for csv_file in sorted(WIND_DIR.glob("p*_1min_*.csv")):
    name = csv_file.stem
    code = name.split("_")[0]
    ALL_CONTRACTS.append({"contract": f"{code}.DCE", "csv": csv_file})
ALL_CONTRACTS.append({
    "contract": "p2601.DCE",
    "csv": ROOT / "data" / "analyse" / "p2601_1min_202507-202512.csv",
})

BT_PARAMS = dict(
    interval=Interval.MINUTE, rate=0.0001, slippage=1.0,
    size=10.0, pricetick=2.0, capital=1_000_000.0,
)

BASE_SETTING = {
    "debug_enabled": True, "debug_log_console": False,
    "atr_activate_mult": 2.5, "atr_trailing_mult": 3.0,
    "cooldown_losses": 4, "cooldown_bars": 30,
}

# 2025 年合约列表
Y25_CONTRACTS = {"p2501.DCE", "p2505.DCE", "p2509.DCE"}


def import_csv_to_db(csv_path, vt_symbol):
    from vnpy.trader.database import get_database
    from vnpy.trader.object import BarData
    from vnpy.trader.constant import Exchange
    from zoneinfo import ZoneInfo
    db = get_database()
    CN_TZ = ZoneInfo("Asia/Shanghai")
    symbol, exchange_str = vt_symbol.split(".")
    exchange = Exchange(exchange_str)
    db.delete_bar_data(symbol, exchange, Interval.MINUTE)
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
    df.sort_values("datetime", inplace=True)
    df.drop_duplicates(subset=["datetime"], keep="first", inplace=True)
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(CN_TZ)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(CN_TZ)
    bars = []
    for _, row in df.iterrows():
        dt = row["datetime"]
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        bar = BarData(
            symbol=symbol, exchange=exchange, datetime=dt,
            interval=Interval.MINUTE,
            volume=float(row.get("volume", 0)),
            turnover=float(row.get("turnover", 0)),
            open_interest=float(row.get("open_interest", 0)),
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            gateway_name="DB",
        )
        bars.append(bar)
    db.save_bar_data(bars)
    start = df["datetime"].min().to_pydatetime()
    end = df["datetime"].max().to_pydatetime()
    return start, end, len(bars)


def run_single(bench, setting):
    vt = bench["contract"]
    start, end, _ = import_csv_to_db(bench["csv"], vt)
    result = run_backtest(
        vt_symbol=vt, start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting, **BT_PARAMS,
    )
    stats = result.stats or {}
    return {
        "contract": vt,
        "pnl": stats.get("total_net_pnl", 0),
        "sharpe": stats.get("sharpe_ratio", 0),
        "trades": stats.get("total_trade_count", 0),
        "commission": stats.get("total_commission", 0),
        "ret%": stats.get("total_return", 0),
    }


# ---------- 搜索空间 ----------
# macd_zero_filter: 0=off (当前), 1=on (diff_15m>0 for bull, <0 for bear)
# atr_entry_filter: 1.0 ~ 3.0
GRID = []
for mzf in [0, 1]:
    for aef in [1.0, 1.5, 2.0, 2.5, 3.0]:
        GRID.append({"macd_zero_filter": mzf, "atr_entry_filter": aef})

print(f"Grid size: {len(GRID)} configs x {len(ALL_CONTRACTS)} contracts = {len(GRID)*len(ALL_CONTRACTS)} runs")
print("="*80)

results_all = []
for gi, cfg in enumerate(GRID):
    setting = dict(BASE_SETTING)
    setting["atr_entry_filter"] = cfg["atr_entry_filter"]
    setting["macd_zero_filter"] = cfg["macd_zero_filter"]

    total_pnl = 0
    y25_pnl = 0
    y25_all_positive = True
    contract_results = {}

    for bench in ALL_CONTRACTS:
        r = run_single(bench, setting)
        total_pnl += r["pnl"]
        contract_results[r["contract"]] = r
        if r["contract"] in Y25_CONTRACTS:
            y25_pnl += r["pnl"]
            if r["pnl"] <= 0:
                y25_all_positive = False

    row = {
        "macd_zero_filter": cfg["macd_zero_filter"],
        "atr_entry_filter": cfg["atr_entry_filter"],
        "total_pnl": round(total_pnl),
        "total_points": round(total_pnl / 10, 1),
        "y25_pnl": round(y25_pnl),
        "y25_all_positive": y25_all_positive,
        "p2501": round(contract_results.get("p2501.DCE", {}).get("pnl", 0)),
        "p2505": round(contract_results.get("p2505.DCE", {}).get("pnl", 0)),
        "p2509": round(contract_results.get("p2509.DCE", {}).get("pnl", 0)),
        "p2601": round(contract_results.get("p2601.DCE", {}).get("pnl", 0)),
        "contracts": contract_results,
    }
    results_all.append(row)

    tag = "Y25-OK" if y25_all_positive else "Y25-NG"
    pts = row["total_points"]
    target = "PASS" if pts >= 10000 else "----"
    print(f"[{gi+1:2d}/{len(GRID)}] mzf={cfg['macd_zero_filter']} aef={cfg['atr_entry_filter']:.1f}  "
          f"total={row['total_pnl']:>8d} pts={pts:>8.1f}  "
          f"y25={row['y25_pnl']:>7d} {tag}  "
          f"p2501={row['p2501']:>7d} p2505={row['p2505']:>7d} p2509={row['p2509']:>7d} p2601={row['p2601']:>7d}  {target}")

# 按 total_pnl 排序输出 Top10
print(f"\n{'='*80}")
print("Top 10 by total PnL (meeting y25 constraint first):")
print(f"{'='*80}")

# 先按 y25_all_positive=True 排，再按 total_pnl 排
sorted_results = sorted(results_all, key=lambda r: (r["y25_all_positive"], r["total_pnl"]), reverse=True)
for i, row in enumerate(sorted_results[:10]):
    tag = "Y25-OK" if row["y25_all_positive"] else "Y25-NG"
    print(f"  #{i+1:2d} mzf={row['macd_zero_filter']} aef={row['atr_entry_filter']:.1f}  "
          f"total={row['total_pnl']:>8d} pts={row['total_points']:>8.1f}  "
          f"y25={row['y25_pnl']:>7d} {tag}  "
          f"p2501={row['p2501']:>7d} p2505={row['p2505']:>7d} p2509={row['p2509']:>7d}")

# 保存
out_dir = ROOT / "experiments" / "iter2" / "r2_grid"
out_dir.mkdir(parents=True, exist_ok=True)

# 去掉 contracts 详情（太大）
save_rows = [{k: v for k, v in r.items() if k != "contracts"} for r in results_all]
with open(out_dir / "grid_results.json", "w") as f:
    json.dump(save_rows, f, indent=2, default=str)
print(f"\nSaved: {out_dir / 'grid_results.json'}")
