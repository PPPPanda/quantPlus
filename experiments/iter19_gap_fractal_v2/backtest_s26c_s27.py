#!/usr/bin/env python3
"""iter19 回测：S26c（阈值优化）+ S27（重置包含方向）."""
import sys
import json
from pathlib import Path
from datetime import datetime

# 确保项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS
import pandas as pd

# iter14 基线参数
ITER14_BASELINE = {
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
    "gap_tier1_atr": 10.0,  # iter14 原值
    "gap_tier2_atr": 30.0,  # iter14 原值
    "gap_reset_inclusion": False,
    "debug_enabled": False,
    "debug_log_console": False,
}

# S26c 优化参数（只调阈值，不重置方向）
S26C_PARAMS = {
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
    "gap_tier1_atr": 6.0,   # 从10改为6
    "gap_tier2_atr": 15.0,  # 从30改为15
    "gap_reset_inclusion": False,
    "debug_enabled": False,
    "debug_log_console": False,
}

# 当前测试参数（切换这里测试不同配置）
BASELINE_PARAMS = S26C_PARAMS  # 测试 S26c 阈值优化

# 关键合约列表（聚焦高风险合约）
CONTRACTS = [
    ("p2401.DCE", "data/analyse/wind/p2401_1min_202308-202312.csv"),
    ("p2601.DCE", "data/analyse/p2601_1min_202507-202512.csv"),
    ("p2209.DCE", "data/analyse/wind/p2209_1min_202204-202208.csv"),
]


def backtest_contract(vt_symbol: str, csv_path: str, params: dict) -> dict:
    """运行单合约回测."""
    full_path = PROJECT_ROOT / csv_path
    if not full_path.exists():
        return {"error": f"File not found: {csv_path}"}
    
    df = pd.read_csv(full_path)
    df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
    
    # 解析时间范围（转换为 Python datetime 以兼容 SQLite）
    df["datetime"] = pd.to_datetime(df["datetime"])
    start = df["datetime"].min().to_pydatetime()
    end = df["datetime"].max().to_pydatetime()
    
    print(f"\n回测 {vt_symbol}: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")
    
    try:
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
        
        stats = result.stats if hasattr(result, 'stats') else {}
        return {
            "total_return": stats.get("total_return", 0),
            "sharpe_ratio": stats.get("sharpe_ratio", 0),
            "max_drawdown": stats.get("max_drawdown", 0),
            "total_trade_count": stats.get("total_trade_count", 0),
            "total_pnl": stats.get("total_pnl", 0),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 60)
    print("iter19 回测：S26c + S27")
    print("=" * 60)
    print(f"参数: tier1={BASELINE_PARAMS['gap_tier1_atr']}, tier2={BASELINE_PARAMS['gap_tier2_atr']}")
    print(f"S27: gap_reset_inclusion={BASELINE_PARAMS['gap_reset_inclusion']}")
    
    results = {}
    for vt_symbol, csv_path in CONTRACTS:
        results[vt_symbol] = backtest_contract(vt_symbol, csv_path, BASELINE_PARAMS)
    
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)
    print(f"{'合约':<15} {'总收益%':<12} {'Sharpe':<10} {'最大回撤%':<12} {'交易次数':<10} {'总PnL':<12}")
    print("-" * 70)
    
    total_pnl = 0
    for contract, r in results.items():
        if "error" in r:
            print(f"{contract:<15} ERROR: {r['error']}")
        else:
            print(f"{contract:<15} {r['total_return']:>10.2f}% {r['sharpe_ratio']:>10.2f} "
                  f"{r['max_drawdown']:>10.2f}% {r['total_trade_count']:>10} {r['total_pnl']:>10.0f}")
            total_pnl += r.get("total_pnl", 0)
    
    print("-" * 70)
    print(f"{'3合约合计':<15} {'':<12} {'':<10} {'':<12} {'':<10} {total_pnl:>10.0f}")
    
    # 保存结果
    output_path = Path(__file__).parent / "backtest_s26c_s27_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    # 与 iter14 基线对比
    print("\n" + "=" * 60)
    print("与 iter14 基线对比（待补充 iter14 原始结果）")
    print("=" * 60)


if __name__ == "__main__":
    main()
