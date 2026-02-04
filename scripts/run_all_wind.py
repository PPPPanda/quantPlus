"""
全量 Wind 合约批量回测.

用法：
    cd quantPlus
    .venv/Scripts/python.exe scripts/run_all_wind.py [--setting key=val ...]

默认使用 iter2 B09 最优参数（消除空头 + 冷却）。
输出到 experiments/iter2/full_wind_results.json + .csv
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from vnpy.trader.constant import Interval

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

# ---------- Wind 合约列表（12 个） ----------
WIND_DIR = ROOT / "data" / "analyse" / "wind"
WIND_CONTRACTS = []
for csv_file in sorted(WIND_DIR.glob("p*_1min_*.csv")):
    # 从文件名提取合约代码，如 p2201
    name = csv_file.stem  # p2201_1min_202108-202112
    contract_code = name.split("_")[0]  # p2201
    vt_symbol = f"{contract_code}.DCE"
    WIND_CONTRACTS.append({
        "contract": vt_symbol,
        "csv": csv_file,
        "source": "Wind",
    })

# 加上 XT p2601
XT_DIR = ROOT / "data" / "analyse"
WIND_CONTRACTS.append({
    "contract": "p2601.DCE",
    "csv": XT_DIR / "p2601_1min_202507-202512.csv",
    "source": "XT",
})

# ---------- 回测口径（冻结） ----------
BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001,
    slippage=1.0,
    size=10.0,
    pricetick=2.0,
    capital=1_000_000.0,
)

# B09 最优参数
DEFAULT_SETTING = {
    "debug_enabled": True,
    "debug_log_console": False,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "cooldown_losses": 4,
    "cooldown_bars": 30,
}


def import_csv_to_db(csv_path: Path, vt_symbol: str):
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


def run_single(bench: dict, setting: dict) -> dict:
    vt_symbol = bench["contract"]
    csv_path = bench["csv"]

    t0 = time.time()
    start, end, bar_count = import_csv_to_db(csv_path, vt_symbol)

    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        **BT_PARAMS,
    )
    elapsed = time.time() - t0

    stats = result.stats or {}
    row = {
        "contract": vt_symbol,
        "source": bench["source"],
        "bars": result.history_data_count,
        "start_date": str(stats.get("start_date", "")),
        "end_date": str(stats.get("end_date", "")),
        "total_days": stats.get("total_days", 0),
        "trades": stats.get("total_trade_count", 0),
        "total_return%": stats.get("total_return", 0),
        "annual_return%": stats.get("annual_return", 0),
        "max_dd%": stats.get("max_ddpercent", 0),
        "sharpe": stats.get("sharpe_ratio", 0),
        "total_pnl": stats.get("total_net_pnl", 0),
        "commission": stats.get("total_commission", 0),
        "win_rate%": stats.get("winning_rate", 0),
        "elapsed_s": round(elapsed, 1),
    }

    status = "+" if row["total_pnl"] > 0 else "-"
    print(f"  {status} {vt_symbol:12s}  ret={row['total_return%']:7.2f}%  "
          f"sharpe={row['sharpe']:6.2f}  pnl={row['total_pnl']:>10.0f}  "
          f"trades={row['trades']:>4d}  maxdd={row['max_dd%']:6.2f}%  [{elapsed:.1f}s]")

    return row


def main():
    extra_setting = dict(DEFAULT_SETTING)
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            extra_setting[k] = v

    print(f"策略参数: {extra_setting}")
    print(f"回测口径: slippage={BT_PARAMS['slippage']}, rate={BT_PARAMS['rate']}, size={BT_PARAMS['size']}")
    print(f"合约数: {len(WIND_CONTRACTS)}")
    print(f"{'='*80}")

    results = []
    for bench in WIND_CONTRACTS:
        row = run_single(bench, extra_setting)
        results.append(row)

    # 汇总
    print(f"\n{'='*80}")
    print("汇总")
    print(f"{'='*80}")

    total_pnl = sum(r["total_pnl"] for r in results)
    total_commission = sum(r["commission"] for r in results)
    win_count = sum(1 for r in results if r["total_pnl"] > 0)
    total_count = len(results)

    # 按 PnL 排序
    sorted_results = sorted(results, key=lambda r: r["total_pnl"], reverse=True)
    for r in sorted_results:
        status = "+" if r["total_pnl"] > 0 else "-"
        print(f"  {status} {r['contract']:12s}  pnl={r['total_pnl']:>10.0f}  "
              f"sharpe={r['sharpe']:6.2f}  ret={r['total_return%']:7.2f}%  "
              f"trades={r['trades']:>4d}")

    print(f"\n  盈利合约: {win_count}/{total_count}")
    print(f"  合计 PnL: {total_pnl:.0f}")
    print(f"  合计手续费: {total_commission:.0f}")
    print(f"  点数 PnL: {total_pnl/10:.1f}")

    # 保存结果
    out_dir = ROOT / "experiments" / "iter2"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = out_dir / "full_wind_results.csv"
    df.to_csv(csv_path, index=False)

    json_path = out_dir / "full_wind_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "params": extra_setting,
            "bt_params": {k: str(v) for k, v in BT_PARAMS.items()},
            "results": results,
            "total_pnl": total_pnl,
            "total_commission": total_commission,
            "win_count": win_count,
            "total_count": total_count,
        }, f, indent=2, default=str)

    print(f"\n  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
