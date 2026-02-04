"""
7 合约基准回测 (iter3).

用法：
    cd quantPlus
    .venv/Scripts/python.exe scripts/run_7bench.py [--setting key=val ...]
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

BENCHMARKS = [
    {"contract": "p2601.DCE", "csv": ROOT / "data/analyse/p2601_1min_202507-202512.csv", "source": "XT", "slot": "bench"},
    {"contract": "p2405.DCE", "csv": ROOT / "data/analyse/wind/p2405_1min_202312-202404.csv", "source": "Wind", "slot": "bench"},
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv", "source": "Wind", "slot": "bench"},
    {"contract": "p2501.DCE", "csv": ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv", "source": "Wind", "slot": "y25"},
    {"contract": "p2505.DCE", "csv": ROOT / "data/analyse/wind/p2505_1min_202412-202504.csv", "source": "Wind", "slot": "y25"},
    {"contract": "p2509.DCE", "csv": ROOT / "data/analyse/wind/p2509_1min_202504-202508.csv", "source": "Wind", "slot": "y25"},
    {"contract": "p2401.DCE", "csv": ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv", "source": "Wind", "slot": "extra"},
]

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

DEFAULT_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "cooldown_losses": 4,
    "cooldown_bars": 30,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "atr_entry_filter": 2.0,
}


def import_csv_to_db(csv_path: Path, vt_symbol: str):
    from vnpy.trader.database import get_database
    from vnpy.trader.object import BarData
    from vnpy.trader.constant import Exchange
    from zoneinfo import ZoneInfo

    CN_TZ = ZoneInfo("Asia/Shanghai")
    db = get_database()
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
    vt_symbol = bench["contract"]
    csv_path = bench["csv"]

    start, end, bar_count = import_csv_to_db(csv_path, vt_symbol)

    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        **BT_PARAMS,
    )

    stats = result.stats or {}
    return {
        "contract": vt_symbol.split(".")[0],
        "source": bench["source"],
        "slot": bench["slot"],
        "trades": stats.get("total_trade_count", 0),
        "total_pnl": stats.get("total_net_pnl", 0),
        "points": stats.get("total_net_pnl", 0) / 10.0,
        "sharpe": round(stats.get("sharpe_ratio", 0), 2),
        "return_pct": round(stats.get("total_return", 0), 2),
        "max_dd_pct": round(stats.get("max_ddpercent", 0), 2),
        "commission": round(stats.get("total_commission", 0), 0),
        "slippage": round(stats.get("total_slippage", 0), 0),
    }


def main():
    setting = dict(DEFAULT_SETTING)
    output_path = None
    for arg in sys.argv[1:]:
        if arg.startswith("--output="):
            output_path = arg.split("=", 1)[1]
        elif "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            try: v = int(v)
            except ValueError:
                try: v = float(v)
                except ValueError: pass
            setting[k] = v

    print(f"Settings: {setting}")
    print("=" * 80)

    results = []
    for bench in BENCHMARKS:
        name = bench["contract"].split(".")[0]
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        r = run_single(bench, setting)
        elapsed = time.time() - t0
        results.append(r)
        print(f"pnl={r['total_pnl']:>8.0f}  pts={r['points']:>8.1f}  trades={r['trades']:>4d}  sharpe={r['sharpe']:>5.2f}  [{elapsed:.1f}s]")

    total_pnl = sum(r["total_pnl"] for r in results)
    total_pts = total_pnl / 10.0
    y25_pnl = sum(r["total_pnl"] for r in results if r["slot"] == "y25")
    y25_all_pos = all(r["total_pnl"] > 0 for r in results if r["slot"] == "y25")

    print("=" * 80)
    print(f"TOTAL: pnl={total_pnl:.0f}  points={total_pts:.1f}  target=5600")
    print(f"Y25 sum={y25_pnl:.0f}  all_positive={y25_all_pos}")
    neg = [r["contract"] for r in results if r["total_pnl"] < 0]
    print(f"Negative contracts: {neg if neg else 'None'}")
    print(f"STATUS: {'PASS' if total_pts >= 5600 else 'FAIL'}")

    if output_path:
        out = {"settings": setting, "results": results, "total_pnl": total_pnl, "total_points": total_pts, "y25_all_positive": y25_all_pos}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        def default_ser(o):
            if hasattr(o, 'item'):
                return o.item()
            raise TypeError(f"Object of type {type(o)} is not JSON serializable")
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=default_ser)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
