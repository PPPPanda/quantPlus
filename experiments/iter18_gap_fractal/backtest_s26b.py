#!/usr/bin/env python3
"""
iter18 Phase 4 小循环1: S26b 分级冷却回测

测试配置：
- baseline: gap_extreme_atr=3.0, gap_cooldown_bars=3 (原S26)
- S26b: gap_extreme_atr=1.5, gap_cooldown_bars=6, gap_tier1_atr=10, gap_tier2_atr=30

目标：
- 降低节假日跳空导致的结构性风险
- TOTAL PnL 不退化
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 设置日志级别
import logging
logging.getLogger("vnpy").setLevel(logging.WARNING)

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy


# 13合约列表
CONTRACTS = [
    ("p2201.DCE", "data/analyse/wind/p2201_1min_202107-202112.csv"),
    ("p2205.DCE", "data/analyse/wind/p2205_1min_202112-202204.csv"),
    ("p2209.DCE", "data/analyse/wind/p2209_1min_202204-202208.csv"),
    ("p2301.DCE", "data/analyse/wind/p2301_1min_202208-202212.csv"),
    ("p2305.DCE", "data/analyse/wind/p2305_1min_202212-202304.csv"),
    ("p2309.DCE", "data/analyse/wind/p2309_1min_202304-202308.csv"),
    ("p2401.DCE", "data/analyse/wind/p2401_1min_202308-202312.csv"),
    ("p2405.DCE", "data/analyse/wind/p2405_1min_202312-202404.csv"),
    ("p2409.DCE", "data/analyse/wind/p2409_1min_202304-202408.csv"),
    ("p2501.DCE", "data/analyse/wind/p2501_1min_202404-202412.csv"),
    ("p2505.DCE", "data/analyse/wind/p2505_1min_202408-202504.csv"),
    ("p2509.DCE", "data/analyse/wind/p2509_1min_202412-202508.csv"),
    ("p2601.DCE", "data/analyse/p2601_1min_202507-202512.csv"),
]

# iter14 基线参数
BASELINE_PARAMS = {
    "debug_enabled": False,
    "debug_log_console": False,
    # 核心参数保持不变
    "min_bi_gap": 4,
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
    # S26 原参数
    "gap_extreme_atr": 3.0,
    "gap_cooldown_bars": 3,
    "gap_tier1_atr": 10.0,  # 不影响，因为原逻辑不用
    "gap_tier2_atr": 30.0,
}

# S26b 新参数
S26B_PARAMS = {
    **BASELINE_PARAMS,
    "gap_extreme_atr": 1.5,   # 降低阈值，捕获更多中等跳空
    "gap_cooldown_bars": 6,   # 延长默认冷却
    "gap_tier1_atr": 10.0,    # <10x ATR: 长冷却(6)
    "gap_tier2_atr": 30.0,    # <30x ATR: 短冷却(3), >30x: 无冷却
}


def run_single_backtest(vt_symbol: str, csv_path: str, params: dict) -> dict:
    """运行单合约回测"""
    import pandas as pd
    from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS
    
    full_path = PROJECT_ROOT / csv_path
    if not full_path.exists():
        return {"error": f"File not found: {csv_path}"}
    
    df = pd.read_csv(full_path)
    df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
    
    if df.empty:
        return {"error": "Empty dataframe after normalization"}
    
    start = df["datetime"].min()
    end = df["datetime"].max()
    
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start,
        end=end,
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=params,
        interval=Interval.MINUTE,
        rate=0.0001,
        slippage=1.0,
        size=10.0,
        pricetick=2.0,
    )
    
    stats = result.stats
    return {
        "trades": stats.get("total_trade_count", 0),
        "total_pnl": stats.get("total_net_pnl", 0),
        "return_pct": stats.get("total_return", 0),
        "sharpe": stats.get("sharpe_ratio", 0),
        "max_dd": stats.get("max_drawdown", 0),
    }


def run_all_backtests(params: dict, label: str) -> dict:
    """运行全部合约回测"""
    results = {}
    total_pnl = 0
    
    print(f"\n{'='*60}")
    print(f"Running {label}...")
    print(f"{'='*60}")
    
    for vt_symbol, csv_path in CONTRACTS:
        contract = vt_symbol.split(".")[0]
        print(f"  {contract}...", end=" ", flush=True)
        
        try:
            r = run_single_backtest(vt_symbol, csv_path, params)
            if "error" in r:
                print(f"ERROR: {r['error']}")
                results[contract] = r
            else:
                pnl = r["total_pnl"]
                total_pnl += pnl
                marker = "✓" if pnl > 0 else "✗"
                print(f"{marker} PnL={pnl:+.1f} trades={r['trades']}")
                results[contract] = r
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results[contract] = {"error": str(e)}
    
    results["_TOTAL"] = total_pnl
    print(f"\n  TOTAL PnL: {total_pnl:+.1f}")
    
    return results


def main():
    """主函数"""
    print("iter18 Phase 4 小循环1: S26b 分级冷却回测")
    print("=" * 60)
    
    # 1. 运行 baseline
    baseline_results = run_all_backtests(BASELINE_PARAMS, "BASELINE (S26 原参数)")
    
    # 2. 运行 S26b
    s26b_results = run_all_backtests(S26B_PARAMS, "S26b (分级冷却)")
    
    # 3. 对比
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Contract':<10} {'Baseline':>10} {'S26b':>10} {'Delta':>10}")
    print("-" * 42)
    
    total_baseline = baseline_results.get("_TOTAL", 0)
    total_s26b = s26b_results.get("_TOTAL", 0)
    
    for vt_symbol, _ in CONTRACTS:
        contract = vt_symbol.split(".")[0]
        b = baseline_results.get(contract, {})
        s = s26b_results.get(contract, {})
        
        b_pnl = b.get("total_pnl", 0) if "error" not in b else 0
        s_pnl = s.get("total_pnl", 0) if "error" not in s else 0
        delta = s_pnl - b_pnl
        
        marker = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"{contract:<10} {b_pnl:>10.1f} {s_pnl:>10.1f} {delta:>+10.1f} {marker}")
    
    print("-" * 42)
    total_delta = total_s26b - total_baseline
    marker = "✓ PASS" if total_delta >= 0 else "✗ FAIL"
    print(f"{'TOTAL':<10} {total_baseline:>10.1f} {total_s26b:>10.1f} {total_delta:>+10.1f} {marker}")
    
    # 4. 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "baseline": baseline_results,
        "s26b": s26b_results,
        "delta": total_delta,
        "pass": total_delta >= 0,
    }
    
    output_path = Path(__file__).parent / "s26b_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    return output


if __name__ == "__main__":
    main()
