#!/usr/bin/env python
"""
测试 Bridge Bar 方案对 iter14 基线的影响。

对比:
1. iter14 基线（bridge_bar_enabled=False）
2. Bridge Bar 方案（bridge_bar_enabled=True）
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

import pandas as pd
from vnpy.trader.constant import Interval, Exchange
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData
from zoneinfo import ZoneInfo

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

CN_TZ = ZoneInfo("Asia/Shanghai")

# iter14 基线参数
BASELINE_SETTINGS = {
    "debug_enabled": False,
    "debug_log_console": False,
    "cooldown_losses": 2,
    "cooldown_bars": 20,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "atr_entry_filter": 2.0,
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
}

# 测试合约
CONTRACTS = [
    ("p2201.DCE", PROJECT_ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv"),
    ("p2205.DCE", PROJECT_ROOT / "data/analyse/wind/p2205_1min_202112-202204.csv"),
    ("p2209.DCE", PROJECT_ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"),
    ("p2301.DCE", PROJECT_ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv"),
    ("p2305.DCE", PROJECT_ROOT / "data/analyse/wind/p2305_1min_202212-202304.csv"),
    ("p2309.DCE", PROJECT_ROOT / "data/analyse/wind/p2309_1min_202304-202308.csv"),
    ("p2401.DCE", PROJECT_ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv"),
    ("p2405.DCE", PROJECT_ROOT / "data/analyse/wind/p2405_1min_202312-202404.csv"),
    ("p2409.DCE", PROJECT_ROOT / "data/analyse/wind/p2409_1min_202401-202408.csv"),
    ("p2501.DCE", PROJECT_ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv"),
    ("p2505.DCE", PROJECT_ROOT / "data/analyse/wind/p2505_1min_202412-202504.csv"),
    ("p2509.DCE", PROJECT_ROOT / "data/analyse/wind/p2509_1min_202504-202508.csv"),
    ("p2601.DCE", PROJECT_ROOT / "data/analyse/p2601_1min_202507-202512.csv"),
]


def import_csv_to_db(csv_path: Path, vt_symbol: str):
    """导入 CSV 到数据库，返回 (start, end, bar_count)."""
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


def run_single_backtest(vt_symbol: str, csv_path: Path, settings: dict) -> dict:
    """运行单个合约回测."""
    from datetime import timedelta
    
    if not csv_path.exists():
        print(f"  [SKIP] {vt_symbol}: 数据文件不存在")
        return None
    
    # 导入数据
    start, end, bar_count = import_csv_to_db(csv_path, vt_symbol)
    
    # 运行回测（扩展时间范围避免边界问题）
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=settings,
        interval=Interval.MINUTE,
        rate=0.0001,
        slippage=1.0,
        size=10.0,
        pricetick=2.0,
    )
    
    stats = result.stats or {}
    return {
        "contract": vt_symbol.split(".")[0],
        "bars": bar_count,
        "trades": stats.get("total_trade_count", 0),
        "total_pnl": stats.get("total_net_pnl", 0),
        "points": stats.get("total_net_pnl", 0) / 10,  # 乘数10
        "sharpe": stats.get("sharpe_ratio", 0),
        "return_pct": stats.get("total_return", 0),
        "max_dd_pct": stats.get("max_ddpercent", 0),
    }


def main():
    """对比 iter14 基线与 Bridge Bar 方案."""
    output_dir = PROJECT_ROOT / "experiments" / "iter21_holiday_gap"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. iter14 基线（禁用 Bridge Bar）
    print("=" * 60)
    print("测试 A: iter14 基线（bridge_bar_enabled=False）")
    print("=" * 60)
    
    baseline_settings = {
        **BASELINE_SETTINGS,
        "bridge_bar_enabled": False,
    }
    
    baseline_results = []
    for vt_symbol, csv_path in CONTRACTS:
        contract = vt_symbol.split(".")[0]
        print(f"\n回测 {contract}...")
        result = run_single_backtest(vt_symbol, csv_path, baseline_settings)
        if result:
            baseline_results.append(result)
            print(f"  PnL: {result['points']:.1f} pts, Trades: {result['trades']}, Sharpe: {result['sharpe']:.2f}")
    
    baseline_total = sum(r['points'] for r in baseline_results)
    print(f"\n基线 TOTAL: {baseline_total:.1f} pts")
    
    # 2. Bridge Bar 方案
    print("\n" + "=" * 60)
    print("测试 B: Bridge Bar 方案（bridge_bar_enabled=True）")
    print("=" * 60)
    
    bridge_settings = {
        **BASELINE_SETTINGS,
        "bridge_bar_enabled": True,
        "bridge_gap_threshold": 1.5,  # 1.5x ATR 以上才插入 bridge
    }
    
    bridge_results = []
    for vt_symbol, csv_path in CONTRACTS:
        contract = vt_symbol.split(".")[0]
        print(f"\n回测 {contract}...")
        result = run_single_backtest(vt_symbol, csv_path, bridge_settings)
        if result:
            bridge_results.append(result)
            print(f"  PnL: {result['points']:.1f} pts, Trades: {result['trades']}, Sharpe: {result['sharpe']:.2f}")
    
    bridge_total = sum(r['points'] for r in bridge_results)
    print(f"\nBridge Bar TOTAL: {bridge_total:.1f} pts")
    
    # 3. 对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    
    print(f"\n{'合约':<10} {'基线':>12} {'BridgeBar':>12} {'差异':>12}")
    print("-" * 48)
    
    for bl, br in zip(baseline_results, bridge_results):
        diff = br['points'] - bl['points']
        diff_str = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
        print(f"{bl['contract']:<10} {bl['points']:>12.1f} {br['points']:>12.1f} {diff_str:>12}")
    
    total_diff = bridge_total - baseline_total
    diff_str = f"+{total_diff:.1f}" if total_diff >= 0 else f"{total_diff:.1f}"
    print("-" * 48)
    print(f"{'TOTAL':<10} {baseline_total:>12.1f} {bridge_total:>12.1f} {diff_str:>12}")
    
    # 保存结果
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "settings": baseline_settings,
            "results": baseline_results,
            "total_points": baseline_total,
        },
        "bridge_bar": {
            "settings": bridge_settings,
            "results": bridge_results,
            "total_points": bridge_total,
        },
        "diff": total_diff,
        "conclusion": "effective" if total_diff > 0 else "ineffective",
    }
    
    with open(output_dir / "bridge_bar_test.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n结果已保存到 {output_dir / 'bridge_bar_test.json'}")
    
    # 结论
    print("\n" + "=" * 60)
    if total_diff > 0:
        print(f"结论: Bridge Bar 方案有效，提升 {total_diff:.1f} pts ({total_diff/baseline_total*100:.1f}%)")
    elif total_diff < 0:
        print(f"结论: Bridge Bar 方案无效，降低 {-total_diff:.1f} pts ({-total_diff/baseline_total*100:.1f}%)")
    else:
        print("结论: Bridge Bar 方案无明显影响")
    print("=" * 60)


if __name__ == "__main__":
    main()
