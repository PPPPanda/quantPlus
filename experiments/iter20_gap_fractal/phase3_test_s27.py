"""
Phase 3: 验证 S27 启用效果
比较 gap_reset_inclusion=False vs True
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import logging
logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS
import pandas as pd

# 基准合约配置（按实际数据文件名）
CONTRACTS = [
    {"contract": "p2209.DCE", "csv": ROOT / "data/analyse/wind/p2209_1min_202204-202208.csv"},
    {"contract": "p2401.DCE", "csv": ROOT / "data/analyse/wind/p2401_1min_202308-202312.csv"},
    {"contract": "p2405.DCE", "csv": ROOT / "data/analyse/wind/p2405_1min_202312-202404.csv"},
    {"contract": "p2601.DCE", "csv": ROOT / "data/analyse/p2601_1min_202507-202512.csv"},
]

BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000.0,
)

# iter14 基线
BASELINE_SETTING = {
    "debug_enabled": False,
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
    "gap_reset_inclusion": False,  # 基线不启用
}

# S27 启用
S27_SETTING = {
    **BASELINE_SETTING,
    "gap_reset_inclusion": True,  # 启用 S27
}

# S27 + 冷却延长
S27_EXTENDED_SETTING = {
    **BASELINE_SETTING,
    "gap_reset_inclusion": True,
    "gap_cooldown_bars": 10,  # 6->10
}


def import_csv_to_db(csv_path: Path, vt_symbol: str):
    """导入 CSV 到 vnpy 数据库，返回 (start, end, bar_count)"""
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
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=Interval.MINUTE,
            volume=float(row["volume"]),
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            gateway_name="DB",
        )
        bars.append(bar)
    
    if bars:
        db.save_bar_data(bars)
    
    start = df["datetime"].min()
    end = df["datetime"].max()
    if hasattr(start, 'to_pydatetime'):
        start = start.to_pydatetime()
    if hasattr(end, 'to_pydatetime'):
        end = end.to_pydatetime()
    return start, end, len(bars)


def run_single(contract_info, setting):
    """运行单合约回测"""
    vt_symbol = contract_info["contract"]
    csv_path = contract_info["csv"]
    
    if not csv_path.exists():
        return None
    
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
        "pnl": stats.get("total_net_pnl", 0),
        "trades": stats.get("total_trade_count", 0),
        "win_rate": stats.get("winning_rate", 0),
    }


def run_test(contracts, setting, name):
    """运行回测测试"""
    print(f"\n=== {name} ===")
    results = {}
    total = 0
    
    for c in contracts:
        contract = c["contract"]
        
        if not c["csv"].exists():
            print(f"  {contract}: file not found")
            continue
        
        r = run_single(c, setting)
        if r is None:
            print(f"  {contract}: backtest failed")
            continue
        
        pnl = r["pnl"]
        trades = r["trades"]
        win_rate = r["win_rate"]
        
        results[contract] = r
        total += pnl
        
        status = "[+]" if pnl > 0 else "[-]"
        print(f"  {contract}: {pnl:+.1f} pts | {trades} trades | {win_rate:.1%} {status}")
    
    print(f"  TOTAL: {total:+.1f} pts")
    return results, total


def main():
    print("=== Phase 3: S27 Validation ===")
    print(f"Time: {datetime.now()}")
    
    # 运行三组实验
    baseline_results, baseline_total = run_test(CONTRACTS, BASELINE_SETTING, "BASELINE (S27 OFF)")
    s27_results, s27_total = run_test(CONTRACTS, S27_SETTING, "S27 ENABLED")
    s27ext_results, s27ext_total = run_test(CONTRACTS, S27_EXTENDED_SETTING, "S27 + EXTENDED COOLDOWN")
    
    # 比较
    print("\n=== Summary ===")
    print(f"BASELINE:        {baseline_total:+.1f} pts")
    print(f"S27 ENABLED:     {s27_total:+.1f} pts ({s27_total - baseline_total:+.1f})")
    print(f"S27 + EXTENDED:  {s27ext_total:+.1f} pts ({s27ext_total - baseline_total:+.1f})")
    
    # 逐合约对比
    print("\n=== Per-Contract Delta ===")
    for contract in baseline_results:
        base = baseline_results[contract]["pnl"]
        s27 = s27_results.get(contract, {}).get("pnl", 0)
        s27ext = s27ext_results.get(contract, {}).get("pnl", 0)
        
        delta_s27 = s27 - base
        delta_s27ext = s27ext - base
        
        print(f"{contract}:")
        print(f"  BASELINE: {base:+.1f}")
        print(f"  S27:      {s27:+.1f} ({delta_s27:+.1f})")
        print(f"  S27+EXT:  {s27ext:+.1f} ({delta_s27ext:+.1f})")
    
    # 结论
    print("\n=== Conclusion ===")
    if s27_total > baseline_total:
        print(f"[OK] S27 enabled is effective: +{s27_total - baseline_total:.1f} pts")
    else:
        print(f"[WARN] S27 enabled is ineffective or harmful: {s27_total - baseline_total:+.1f} pts")
    
    if s27ext_total > s27_total:
        print(f"[OK] Extended cooldown is effective: +{s27ext_total - s27_total:.1f} pts")
    else:
        print(f"[WARN] Extended cooldown is ineffective or harmful: {s27ext_total - s27_total:+.1f} pts")


if __name__ == "__main__":
    main()
