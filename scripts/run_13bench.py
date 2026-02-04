"""
13 合约全量回测 (Wind + XT).

用法：
    cd quantPlus
    .venv/Scripts/python.exe scripts/run_13bench.py [key=val ...] [--output=path.json]
"""
from __future__ import annotations

import json, logging, sys, time
from datetime import timedelta
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

# Reuse import logic from run_7bench
from run_7bench import import_csv_to_db, BT_PARAMS

ALL_CONTRACTS = [
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
    {"contract": "p2601.DCE", "csv": ROOT / "data/analyse/p2601_1min_202507-202512.csv", "source": "XT"},
]

DEFAULT_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
}


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
        "contract": vt_symbol,
        "source": bench["source"],
        "bars": bar_count,
        "start_date": str(start.date()) if hasattr(start, 'date') else str(start),
        "end_date": str(end.date()) if hasattr(end, 'date') else str(end),
        "total_days": stats.get("total_days", 0),
        "trades": stats.get("total_trade_count", 0),
        "total_return%": stats.get("total_return", 0),
        "annual_return%": stats.get("annual_return", 0),
        "max_dd%": stats.get("max_ddpercent", 0),
        "sharpe": stats.get("sharpe_ratio", 0),
        "total_pnl": stats.get("total_net_pnl", 0),
        "commission": stats.get("total_commission", 0),
        "win_rate%": stats.get("winning_rate", 0),
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
                except ValueError:
                    if v.lower() in ('true', '1'): v = True
                    elif v.lower() in ('false', '0'): v = False
            setting[k] = v

    print(f"Settings: {setting}")
    print("=" * 80)

    results = []
    for bench in ALL_CONTRACTS:
        name = bench["contract"].split(".")[0]
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        r = run_single(bench, setting)
        elapsed = time.time() - t0
        results.append(r)
        pts = r["total_pnl"] / 10
        print(f"pnl={r['total_pnl']:>8.0f}  pts={pts:>8.1f}  trades={r['trades']:>4d}  "
              f"sharpe={r['sharpe']:>5.2f}  ret={r['total_return%']:>6.2f}%  [{elapsed:.1f}s]")

    total_pnl = sum(r["total_pnl"] for r in results)
    total_commission = sum(r["commission"] for r in results)
    win_count = sum(1 for r in results if r["total_pnl"] > 0)
    neg = [r["contract"].split(".")[0] for r in results if r["total_pnl"] < 0]

    print("=" * 80)
    print(f"盈利合约: {win_count}/{len(results)}")
    print(f"总PnL: {total_pnl:.0f}  总点数: {total_pnl/10:.1f}")
    print(f"总手续费: {total_commission:.0f}")
    print(f"亏损合约: {neg if neg else 'None'}")

    if output_path:
        out = {
            "params": setting,
            "bt_params": {k: str(v) for k, v in BT_PARAMS.items()},
            "results": results,
            "total_pnl": total_pnl,
            "total_commission": total_commission,
            "win_count": win_count,
            "total_count": len(results),
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        def ser(o):
            if hasattr(o, 'item'): return o.item()
            raise TypeError(f"Not serializable: {type(o)}")
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=ser)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
