#!/usr/bin/env python
"""S26 极端跳空安全网 全量对比脚本"""
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

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

# 基线设置（禁用 S26）
BASELINE_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "cooldown_losses": 2,
    "cooldown_bars": 20,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "atr_entry_filter": 2.0,
    "gap_extreme_atr": 0.0,  # 禁用 S26
    "gap_cooldown_bars": 0,
}

# S26 设置（启用极端跳空安全网）
S26_SETTING = {
    **BASELINE_SETTING,
    "gap_extreme_atr": 3.0,  # 启用 S26
    "gap_cooldown_bars": 3,
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
        "bars": bar_count,
        "trades": stats.get("total_trade_count", 0),
        "total_pnl": stats.get("total_net_pnl", 0),
        "points": stats.get("total_net_pnl", 0) / 10.0,
        "sharpe": round(stats.get("sharpe_ratio", 0), 2),
        "return_pct": round(stats.get("total_return", 0), 2),
        "max_dd_pct": round(stats.get("max_ddpercent", 0), 2),
    }


def main():
    print("=" * 80)
    print("S26 极端跳空安全网 全量对比")
    print("=" * 80)
    print()
    
    baseline_results = []
    s26_results = []
    
    print("Running baseline (S26 disabled)...")
    for bench in BENCHMARKS:
        name = bench["contract"].split(".")[0]
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        r = run_single(bench, BASELINE_SETTING)
        elapsed = time.time() - t0
        baseline_results.append(r)
        print(f"pts={r['points']:>8.1f} [{elapsed:.1f}s]")
    
    print()
    print("Running S26 (gap_extreme_atr=3.0, gap_cooldown_bars=3)...")
    for bench in BENCHMARKS:
        name = bench["contract"].split(".")[0]
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        r = run_single(bench, S26_SETTING)
        elapsed = time.time() - t0
        s26_results.append(r)
        print(f"pts={r['points']:>8.1f} [{elapsed:.1f}s]")
    
    # 输出对比表
    print()
    print("=" * 80)
    print("对比结果")
    print("=" * 80)
    print(f"{'合约':<10} {'基线(pts)':>12} {'S26(pts)':>12} {'Delta':>10}")
    print("-" * 80)
    
    total_baseline = 0
    total_s26 = 0
    
    for b, s in zip(baseline_results, s26_results):
        contract = b['contract']
        bp = b['points']
        sp = s['points']
        delta = sp - bp
        total_baseline += bp
        total_s26 += sp
        sign = '+' if delta > 0 else ''
        emoji = '✅' if delta > 0 else ('⚠️' if delta < -10 else '')
        print(f"{contract:<10} {bp:>12.1f} {sp:>12.1f} {sign}{delta:>9.1f} {emoji}")
    
    print("-" * 80)
    total_delta = total_s26 - total_baseline
    sign = '+' if total_delta > 0 else ''
    print(f"{'TOTAL':<10} {total_baseline:>12.1f} {total_s26:>12.1f} {sign}{total_delta:>9.1f}")
    print()
    print(f"净收益变化: {total_delta:+.1f} pts")
    
    # 保存结果
    output = {
        "baseline": baseline_results,
        "s26": s26_results,
        "summary": {
            "baseline_total": total_baseline,
            "s26_total": total_s26,
            "delta": total_delta,
        }
    }
    output_path = ROOT / "experiments/iter19/s26_full_compare.json"
    
    def default_ser(o):
        if hasattr(o, 'item'):
            return o.item()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=default_ser)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
