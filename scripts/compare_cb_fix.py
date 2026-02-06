"""测试 CB2.1 时间驱动恢复修复"""
import sys
from datetime import timedelta
sys.path.insert(0, "E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus")
sys.path.insert(0, "E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/scripts")

from pathlib import Path
from run_13bench import import_csv_to_db, BENCHMARKS, BT_PARAMS
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy

test_contract = "p2501.DCE"

# 基线参数（简单CB）
BASE = {"circuit_breaker_losses": 7, "circuit_breaker_bars": 70, "div_threshold": 0.39, "max_pullback_atr": 3.2}

# CB2.1 + 时间驱动恢复（70 bars，与简单CB一致）
CB21_FIXED = {**BASE, "cb2_enabled": True, "cb2_window_trades": 10,
    "cb2_l1_threshold": -4.0, "cb2_l2_threshold": -6.0, "cb2_l3_threshold": -8.0,
    "cb2_magnitude_weight": True, "cb2_l1_skip_pct": 0.0,
    "cb2_max_pause_bars": 70}  # 与简单CB一致

# CB2.1 无时间恢复（原问题版本）
CB21_BROKEN = {**BASE, "cb2_enabled": True, "cb2_window_trades": 10,
    "cb2_l1_threshold": -4.0, "cb2_l2_threshold": -6.0, "cb2_l3_threshold": -8.0,
    "cb2_magnitude_weight": True, "cb2_l1_skip_pct": 0.0,
    "cb2_max_pause_bars": 0}  # 禁用时间恢复（原问题）

print("Test CB2.1 time-based recovery fix on p2501...")

for bench in BENCHMARKS:
    if bench.get("contract") == test_contract:
        csv_path = Path(bench["csv"])
        start, end, _ = import_csv_to_db(csv_path, test_contract)
        
        print("\n--- BASELINE (simple CB) ---")
        r1 = run_backtest(test_contract, start - timedelta(days=1), end + timedelta(days=1),
            CtaChanPivotStrategy, BASE, **BT_PARAMS)
        s1 = r1.stats or {}
        pnl1, trades1 = s1.get('total_net_pnl', 0), s1.get('total_trade_count', 0)
        print(f"PnL: {pnl1:.1f} pts, Trades: {trades1}")
        
        print("\n--- CB2.1 FIXED (time recovery=70) ---")
        r2 = run_backtest(test_contract, start - timedelta(days=1), end + timedelta(days=1),
            CtaChanPivotStrategy, CB21_FIXED, **BT_PARAMS)
        s2 = r2.stats or {}
        pnl2, trades2 = s2.get('total_net_pnl', 0), s2.get('total_trade_count', 0)
        print(f"PnL: {pnl2:.1f} pts, Trades: {trades2}")
        
        print("\n--- CB2.1 BROKEN (no time recovery) ---")
        r3 = run_backtest(test_contract, start - timedelta(days=1), end + timedelta(days=1),
            CtaChanPivotStrategy, CB21_BROKEN, **BT_PARAMS)
        s3 = r3.stats or {}
        pnl3, trades3 = s3.get('total_net_pnl', 0), s3.get('total_trade_count', 0)
        print(f"PnL: {pnl3:.1f} pts, Trades: {trades3}")
        
        print("\n=== COMPARISON ===")
        print(f"BASELINE:     PnL={pnl1:8.1f}, Trades={trades1}")
        print(f"CB2.1 FIXED:  PnL={pnl2:8.1f}, Trades={trades2} (vs BASE: {pnl2-pnl1:+.1f})")
        print(f"CB2.1 BROKEN: PnL={pnl3:8.1f}, Trades={trades3} (vs BASE: {pnl3-pnl1:+.1f})")
        
        if abs(pnl2 - pnl1) < abs(pnl3 - pnl1):
            print("\n[OK] FIX WORKS! CB2.1 FIXED is closer to BASELINE")
        else:
            print("\n[WARN] Fix may not be sufficient")
        break
