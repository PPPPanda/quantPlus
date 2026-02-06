"""
iter19: 测试 S26 极端跳空安全网效果.

对比启用/禁用 S26 的回测结果。
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

import pandas as pd
from vnpy.trader.constant import Interval

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

# 13 合约列表
CONTRACTS = [
    {"contract": "p2201.DCE", "csv": ROOT / "data/analyse/wind/p2201_1min_202108-202112.csv", "source": "Wind"},
    {"contract": "p2205.DCE", "csv": ROOT / "data/analyse/wind/p2205_1min_202112-202204.csv", "source": "Wind"},
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv", "source": "Wind"},
    {"contract": "p2301.DCE", "csv": ROOT / "data/analyse/wind/p2301_1min_202208-202212.csv", "source": "Wind"},
    {"contract": "p2305.DCE", "csv": ROOT / "data/analyse/wind/p2305_1min_202212-202304.csv", "source": "Wind"},
    {"contract": "p2309.DCE", "csv": ROOT / "data/analyse/wind/p2309_1min_202304-202308.csv", "source": "Wind"},
    {"contract": "p2401.DCE", "csv": ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv", "source": "Wind"},
    {"contract": "p2405.DCE", "csv": ROOT / "data/analyse/wind/p2405_1min_202312-202404.csv", "source": "Wind"},
    {"contract": "p2409.DCE", "csv": ROOT / "data/analyse/wind/p2409_1min_202404-202408.csv", "source": "Wind"},
    {"contract": "p2501.DCE", "csv": ROOT / "data/analyse/wind/p2501_1min_202404-202412.csv", "source": "Wind"},
    {"contract": "p2505.DCE", "csv": ROOT / "data/analyse/wind/p2505_1min_202412-202504.csv", "source": "Wind"},
    {"contract": "p2509.DCE", "csv": ROOT / "data/analyse/wind/p2509_1min_202504-202508.csv", "source": "Wind"},
    {"contract": "p2601.DCE", "csv": ROOT / "data/analyse/p2601_1min_202507-202512.csv", "source": "XT"},
]

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

# 基线参数
BASE_SETTING = {
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

# S26 启用参数
S26_SETTING = {
    **BASE_SETTING,
    "gap_extreme_atr": 3.0,   # 启用 S26: >3×ATR 触发
    "gap_cooldown_bars": 3,   # 暂停 3 根 5m bar
}


def run_single_backtest(contract_info: dict, setting: dict) -> dict:
    """运行单合约回测."""
    csv_path = contract_info["csv"]
    vt_symbol = contract_info["contract"]
    
    if not csv_path.exists():
        return {"error": f"CSV not found: {csv_path}"}
    
    df = pd.read_csv(csv_path)
    df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
    
    # 转换为 Python datetime（避免 Timestamp 类型问题）
    start = pd.to_datetime(df["datetime"].min()).to_pydatetime()
    end = pd.to_datetime(df["datetime"].max()).to_pydatetime()
    
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start, end=end,
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        **BT_PARAMS,
    )
    
    stats = result.stats if hasattr(result, 'stats') else {}
    return {
        "pnl": stats.get("total_net_pnl", 0),
        "pts": stats.get("total_net_pnl", 0) / 10,  # 点数
        "trades": stats.get("total_trade_count", 0),
        "sharpe": stats.get("sharpe_ratio", 0),
        "bars": len(df),
    }


def main():
    print("=" * 60)
    print("iter19: S26 极端跳空安全网测试")
    print("=" * 60)
    
    # 先测试 p2209（节假日跳空影响最大的合约）
    test_contracts = [c for c in CONTRACTS if c["contract"] in ["p2209.DCE", "p2501.DCE", "p2601.DCE"]]
    
    results = {"baseline": {}, "s26": {}}
    
    print("\n[1/2] 运行基线回测（S26 禁用）...")
    for c in test_contracts:
        name = c["contract"].split(".")[0]
        print(f"  - {name}...", end=" ", flush=True)
        r = run_single_backtest(c, BASE_SETTING)
        results["baseline"][name] = r
        print(f"pts={r.get('pts', 0):.1f}")
    
    print("\n[2/2] 运行 S26 启用回测...")
    for c in test_contracts:
        name = c["contract"].split(".")[0]
        print(f"  - {name}...", end=" ", flush=True)
        r = run_single_backtest(c, S26_SETTING)
        results["s26"][name] = r
        print(f"pts={r.get('pts', 0):.1f}")
    
    # 汇总对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"{'合约':<8} {'基线(pts)':<12} {'S26(pts)':<12} {'Delta':<10}")
    print("-" * 42)
    
    total_base = 0
    total_s26 = 0
    for name in results["baseline"]:
        base_pts = results["baseline"][name].get("pts", 0)
        s26_pts = results["s26"][name].get("pts", 0)
        delta = s26_pts - base_pts
        total_base += base_pts
        total_s26 += s26_pts
        print(f"{name:<8} {base_pts:>10.1f}   {s26_pts:>10.1f}   {delta:>+8.1f}")
    
    print("-" * 42)
    print(f"{'合计':<8} {total_base:>10.1f}   {total_s26:>10.1f}   {total_s26 - total_base:>+8.1f}")
    
    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "iteration": "iter19",
        "experiment": "S26_extreme_gap_safenet",
        "baseline": results["baseline"],
        "s26": results["s26"],
        "summary": {
            "baseline_total": total_base,
            "s26_total": total_s26,
            "delta": total_s26 - total_base,
        }
    }
    
    out_path = ROOT / "experiments/iter19/s26_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
