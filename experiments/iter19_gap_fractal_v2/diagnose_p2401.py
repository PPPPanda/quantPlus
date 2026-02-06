#!/usr/bin/env python3
"""
p2401 跳空-亏损关联诊断

分析 p2401 的每笔交易与跳空事件的时间关联
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 设置日志级别
import logging
logging.getLogger("vnpy").setLevel(logging.WARNING)

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

# iter14 基线参数
BASELINE_PARAMS = {
    "debug_enabled": False,
    "debug_log_console": False,
    "min_bi_gap": 4,
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
    "gap_extreme_atr": 3.0,
    "gap_cooldown_bars": 3,
}

# p2401 跳空事件（从 analysis_results.json 提取）
GAP_EVENTS = [
    {"datetime": "2023-08-21 09:00:00", "gap_atr": 6.76, "gap": 50.0, "in_fractal": True},
    {"datetime": "2023-08-28 09:00:00", "gap_atr": 4.46, "gap": 38.0, "in_fractal": True},
    {"datetime": "2023-09-18 09:00:00", "gap_atr": 4.52, "gap": 34.0, "in_fractal": True},
    {"datetime": "2023-10-09 09:00:00", "gap_atr": 16.71, "gap": -158.0, "in_fractal": False},  # 国庆节后
    {"datetime": "2023-10-16 09:00:00", "gap_atr": 6.61, "gap": 48.0, "in_fractal": True},
    {"datetime": "2023-10-23 09:00:00", "gap_atr": 6.55, "gap": -44.0, "in_fractal": False},
    {"datetime": "2023-10-30 09:00:00", "gap_atr": 4.31, "gap": 26.0, "in_fractal": True},
    {"datetime": "2023-11-06 09:00:00", "gap_atr": 5.38, "gap": -32.0, "in_fractal": True},
    {"datetime": "2023-11-27 09:00:00", "gap_atr": 4.81, "gap": -32.0, "in_fractal": True},
    {"datetime": "2023-12-04 09:00:00", "gap_atr": 7.42, "gap": -46.0, "in_fractal": True},
    {"datetime": "2023-12-11 09:00:00", "gap_atr": 7.35, "gap": -50.0, "in_fractal": True},
]


def run_p2401_backtest():
    """运行 p2401 回测并返回详细交易记录"""
    csv_path = PROJECT_ROOT / "data/analyse/wind/p2401_1min_202308-202401.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
    
    start = df["datetime"].min()
    end = df["datetime"].max()
    
    print(f"Running p2401 backtest: {start} to {end}")
    print(f"Data bars: {len(df)}")
    
    result = run_backtest(
        vt_symbol="p2401.DCE",
        start=start,
        end=end,
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=BASELINE_PARAMS,
        interval=Interval.MINUTE,
        rate=0.0001,
        slippage=1.0,
        size=10.0,
        pricetick=2.0,
    )
    
    return result


def analyze_trades_vs_gaps(trades, gap_events):
    """分析交易与跳空事件的关联"""
    # 将跳空事件转为 datetime
    for gap in gap_events:
        gap["dt"] = pd.to_datetime(gap["datetime"])
    
    results = []
    
    for trade in trades:
        # 获取交易信息
        trade_dt = trade.datetime
        trade_pnl = getattr(trade, "pnl", None)  # 可能没有 pnl 属性
        trade_dir = trade.direction.value
        trade_offset = trade.offset.value
        trade_price = trade.price
        trade_vol = trade.volume
        
        # 只分析开仓交易
        if trade_offset != "开":
            continue
        
        # 找最近的跳空事件
        nearest_gap = None
        min_gap_hours = float("inf")
        
        for gap in gap_events:
            gap_dt = gap["dt"]
            hours_diff = (trade_dt - gap_dt).total_seconds() / 3600
            
            # 只看跳空后的交易（0-24小时内）
            if 0 <= hours_diff <= 24:
                if hours_diff < min_gap_hours:
                    min_gap_hours = hours_diff
                    nearest_gap = gap
        
        results.append({
            "trade_time": str(trade_dt),
            "direction": trade_dir,
            "price": trade_price,
            "volume": trade_vol,
            "hours_after_gap": round(min_gap_hours, 2) if nearest_gap else None,
            "gap_datetime": str(nearest_gap["dt"]) if nearest_gap else None,
            "gap_atr": nearest_gap["gap_atr"] if nearest_gap else None,
            "gap_size": nearest_gap["gap"] if nearest_gap else None,
            "gap_in_fractal": nearest_gap["in_fractal"] if nearest_gap else None,
        })
    
    return results


def match_trades_to_pnl(trades):
    """将开仓和平仓配对，计算每笔交易盈亏"""
    # 按时间排序
    sorted_trades = sorted(trades, key=lambda t: t.datetime)
    
    positions = []  # 当前持仓
    completed_trades = []  # 完成的交易
    
    for trade in sorted_trades:
        is_open = trade.offset.value == "开"
        is_long = trade.direction.value == "多"
        
        if is_open:
            positions.append({
                "entry_time": trade.datetime,
                "entry_price": trade.price,
                "volume": trade.volume,
                "is_long": is_long,
            })
        else:
            # 平仓 - 匹配最早的同方向持仓
            for pos in positions:
                if pos["is_long"] != is_long:  # 方向相反才能平
                    continue
                
                # 计算盈亏
                if pos["is_long"]:
                    pnl = (trade.price - pos["entry_price"]) * pos["volume"] * 10  # 合约乘数10
                else:
                    pnl = (pos["entry_price"] - trade.price) * pos["volume"] * 10
                
                completed_trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": trade.datetime,
                    "entry_price": pos["entry_price"],
                    "exit_price": trade.price,
                    "direction": "多" if pos["is_long"] else "空",
                    "volume": pos["volume"],
                    "pnl": pnl,
                    "pnl_points": trade.price - pos["entry_price"] if pos["is_long"] else pos["entry_price"] - trade.price,
                })
                positions.remove(pos)
                break
    
    return completed_trades


def main():
    print("=" * 70)
    print("p2401 跳空-亏损关联诊断")
    print("=" * 70)
    
    # 1. 运行回测
    result = run_p2401_backtest()
    
    print(f"\nBacktest stats:")
    print(f"  Total trades: {result.stats.get('total_trade_count', 0)}")
    print(f"  Total PnL: {result.stats.get('total_net_pnl', 0):.1f}")
    print(f"  Sharpe: {result.stats.get('sharpe_ratio', 0):.2f}")
    print(f"  Max DD: {result.stats.get('max_drawdown', 0):.2f}%")
    
    # 2. 配对交易计算盈亏
    trades = result.trades
    completed = match_trades_to_pnl(trades)
    
    print(f"\n完成的交易数: {len(completed)}")
    
    # 3. 按盈亏排序，找出最大亏损
    completed.sort(key=lambda x: x["pnl"])
    
    print("\n" + "=" * 70)
    print("TOP 10 亏损交易")
    print("=" * 70)
    
    top_losses = completed[:10]
    for i, t in enumerate(top_losses, 1):
        print(f"\n{i}. PnL: {t['pnl']:+.0f} ({t['pnl_points']:+.0f}点)")
        print(f"   入场: {t['entry_time']} @ {t['entry_price']:.0f}")
        print(f"   出场: {t['exit_time']} @ {t['exit_price']:.0f}")
        print(f"   方向: {t['direction']}")
    
    # 4. 分析亏损交易与跳空的关联
    print("\n" + "=" * 70)
    print("亏损交易与跳空关联分析")
    print("=" * 70)
    
    loss_trades = [t for t in completed if t["pnl"] < 0]
    print(f"\n总亏损交易数: {len(loss_trades)}")
    print(f"总亏损金额: {sum(t['pnl'] for t in loss_trades):.0f}")
    
    # 将跳空事件转为 datetime
    for gap in GAP_EVENTS:
        gap["dt"] = pd.to_datetime(gap["datetime"])
    
    # 检查每笔亏损交易是否在跳空后
    gap_related_losses = []
    non_gap_losses = []
    
    for trade in loss_trades:
        entry_dt = trade["entry_time"]
        
        # 找最近的跳空（入场前24小时内）
        nearest_gap = None
        min_hours = float("inf")
        
        for gap in GAP_EVENTS:
            hours_diff = (entry_dt - gap["dt"]).total_seconds() / 3600
            # 跳空后0-24小时内入场
            if 0 <= hours_diff <= 24:
                if hours_diff < min_hours:
                    min_hours = hours_diff
                    nearest_gap = gap
        
        if nearest_gap:
            gap_related_losses.append({
                **trade,
                "hours_after_gap": round(min_hours, 2),
                "gap_datetime": str(nearest_gap["dt"]),
                "gap_atr": nearest_gap["gap_atr"],
                "gap_size": nearest_gap["gap"],
                "gap_in_fractal": nearest_gap["in_fractal"],
            })
        else:
            non_gap_losses.append(trade)
    
    print(f"\n跳空相关亏损: {len(gap_related_losses)} 笔, 金额: {sum(t['pnl'] for t in gap_related_losses):.0f}")
    print(f"非跳空亏损: {len(non_gap_losses)} 笔, 金额: {sum(t['pnl'] for t in non_gap_losses):.0f}")
    
    # 5. 详细列出跳空相关亏损
    print("\n" + "=" * 70)
    print("跳空相关亏损详情")
    print("=" * 70)
    
    gap_related_losses.sort(key=lambda x: x["pnl"])
    for t in gap_related_losses:
        print(f"\n入场: {t['entry_time']}")
        print(f"  PnL: {t['pnl']:+.0f} ({t['direction']})")
        print(f"  跳空后 {t['hours_after_gap']:.1f} 小时入场")
        print(f"  跳空: {t['gap_datetime']} ({t['gap_size']:+.0f}点, {t['gap_atr']:.1f}x ATR)")
        print(f"  跳空在分型内: {t['gap_in_fractal']}")
    
    # 6. 统计不同时间段的亏损
    print("\n" + "=" * 70)
    print("按跳空后时间段分析")
    print("=" * 70)
    
    time_buckets = {
        "0-2h": [],
        "2-6h": [],
        "6-12h": [],
        "12-24h": [],
    }
    
    for t in gap_related_losses:
        h = t["hours_after_gap"]
        if h <= 2:
            time_buckets["0-2h"].append(t)
        elif h <= 6:
            time_buckets["2-6h"].append(t)
        elif h <= 12:
            time_buckets["6-12h"].append(t)
        else:
            time_buckets["12-24h"].append(t)
    
    for bucket, trades in time_buckets.items():
        if trades:
            total = sum(t["pnl"] for t in trades)
            print(f"{bucket}: {len(trades)} 笔, 总亏损 {total:.0f}")
    
    # 7. 保存结果
    output = {
        "backtest_stats": {
            "total_trades": result.stats.get("total_trade_count", 0),
            "total_pnl": result.stats.get("total_net_pnl", 0),
            "sharpe": result.stats.get("sharpe_ratio", 0),
            "max_dd": result.stats.get("max_drawdown", 0),
        },
        "loss_analysis": {
            "total_loss_trades": len(loss_trades),
            "total_loss_amount": sum(t["pnl"] for t in loss_trades),
            "gap_related_losses": len(gap_related_losses),
            "gap_related_amount": sum(t["pnl"] for t in gap_related_losses),
            "non_gap_losses": len(non_gap_losses),
            "non_gap_amount": sum(t["pnl"] for t in non_gap_losses),
        },
        "gap_related_trades": gap_related_losses,
        "top_10_losses": top_losses,
        "time_bucket_analysis": {
            bucket: {
                "count": len(trades),
                "total_pnl": sum(t["pnl"] for t in trades),
            }
            for bucket, trades in time_buckets.items()
        },
    }
    
    output_path = Path(__file__).parent / "p2401_diagnosis_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存到: {output_path}")
    
    return output


if __name__ == "__main__":
    main()
