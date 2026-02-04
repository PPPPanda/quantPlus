"""全量 Wind 数据回测（12 Wind + 1 XT = 13 合约）."""
from __future__ import annotations
import sys, time, json, logging
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from vnpy.trader.constant import Interval

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

CONTRACTS = [
    {"contract": "p2601.DCE", "csv": ROOT / "data/analyse/p2601_1min_202507-202512.csv", "source": "XT"},
    {"contract": "p2201.DCE", "csv": ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv", "source": "Wind"},
    {"contract": "p2205.DCE", "csv": ROOT / "data/analyse/wind/p2205_1min_202112-202204.csv", "source": "Wind"},
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv", "source": "Wind"},
    {"contract": "p2301.DCE", "csv": ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv", "source": "Wind"},
    {"contract": "p2305.DCE", "csv": ROOT / "data/analyse/wind/p2305_1min_202212-202304.csv", "source": "Wind"},
    {"contract": "p2309.DCE", "csv": ROOT / "data/analyse/wind/p2309_1min_202304-202308.csv", "source": "Wind"},
    {"contract": "p2401.DCE", "csv": ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv", "source": "Wind"},
    {"contract": "p2405.DCE", "csv": ROOT / "data/analyse/wind/p2405_1min_202312-202404.csv", "source": "Wind"},
    {"contract": "p2409.DCE", "csv": ROOT / "data/analyse/wind/p2409_1min_202401-202408.csv", "source": "Wind"},
    {"contract": "p2501.DCE", "csv": ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv", "source": "Wind"},
    {"contract": "p2505.DCE", "csv": ROOT / "data/analyse/wind/p2505_1min_202412-202504.csv", "source": "Wind"},
    {"contract": "p2509.DCE", "csv": ROOT / "data/analyse/wind/p2509_1min_202504-202508.csv", "source": "Wind"},
]

BT_PARAMS = dict(interval=Interval.MINUTE, rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0)

DEFAULT_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "cooldown_losses": 2,
    "cooldown_bars": 20,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "atr_entry_filter": 2.0,
    "stop_buffer_atr_pct": 0.02,
    "div_mode": 1,
    "div_threshold": 0.70,
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


if __name__ == "__main__":
    setting = dict(DEFAULT_SETTING)
    results = []
    total_pnl = 0.0

    print(f"{'Contract':>8} | {'PnL':>8} | {'Pts':>8} | {'Trades':>6} | {'Sharpe':>7} | {'Ret%':>7} | {'MaxDD%':>7} | Time")
    print("-" * 85)

    for item in CONTRACTS:
        vt = item["contract"]
        name = vt.split(".")[0]
        t0 = time.time()
        start, end, n_bars = import_csv_to_db(item["csv"], vt)
        r = run_backtest(
            vt_symbol=vt, start=start - timedelta(days=1), end=end + timedelta(days=1),
            strategy_class=CtaChanPivotStrategy, strategy_setting=setting, **BT_PARAMS
        )
        elapsed = time.time() - t0
        s = r.stats or {}
        pnl = round(s.get("total_net_pnl", 0), 1)
        pts = round(pnl / 10, 1)
        trades = s.get("total_trade_count", 0)
        sharpe = s.get("sharpe_ratio", 0)
        ret = s.get("total_return", 0)
        maxdd = s.get("max_ddpercent", 0)

        total_pnl += pnl

        row = {
            "contract": name, "source": item["source"],
            "pnl": float(pnl), "pts": float(pts), "trades": int(trades),
            "sharpe": round(float(sharpe), 2), "ret_pct": round(float(ret), 2),
            "max_dd_pct": round(float(maxdd), 2),
        }
        results.append(row)
        print(f"{name:>8} | {pnl:>+8.0f} | {pts:>+8.1f} | {trades:>6} | {sharpe:>+7.2f} | {ret:>+7.2f} | {maxdd:>7.2f} | {elapsed:.1f}s")

    total_pts = round(total_pnl / 10, 1)
    print("=" * 85)
    print(f"{'TOTAL':>8} | {total_pnl:>+8.0f} | {total_pts:>+8.1f}")

    p2209_pts = next((r["pts"] for r in results if r["contract"] == "p2209"), 0)
    p2601_pts = next((r["pts"] for r in results if r["contract"] == "p2601"), 0)
    non_p2209_pts = total_pts - p2209_pts
    neg = [r["contract"] for r in results if r["pnl"] < 0]
    pos = [r["contract"] for r in results if r["pnl"] > 0]

    print(f"\np2601 pts: {p2601_pts:>+.1f} (target >=1000)")
    print(f"p2209 pts: {p2209_pts:>+.1f} (target >=1000)")
    print(f"Non-p2209 pts: {non_p2209_pts:>+.1f} (target >4800)")
    print(f"Positive: {len(pos)}/{len(results)}  Negative: {neg}")

    out = ROOT / "experiments/iter4/wind_all_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "setting": setting,
        "results": results,
        "total_pnl": round(total_pnl, 1),
        "total_pts": total_pts,
        "p2601_pts": p2601_pts,
        "p2209_pts": p2209_pts,
        "non_p2209_pts": non_p2209_pts,
        "neg_contracts": neg,
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
