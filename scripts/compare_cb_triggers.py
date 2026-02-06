"""对比简单CB vs CB2.1触发频率"""
import sys
from datetime import timedelta
sys.path.insert(0, "E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus")
sys.path.insert(0, "E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/scripts")

from pathlib import Path
from run_13bench import import_csv_to_db, BENCHMARKS, BT_PARAMS
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy

test_contract = "p2501.DCE"

BASE = {"circuit_breaker_losses": 7, "circuit_breaker_bars": 70, "div_threshold": 0.39, "max_pullback_atr": 3.2}

CB21_FULL = {**BASE, "cb2_enabled": True, "cb2_window_trades": 10,
    "cb2_l1_threshold": -4.0, "cb2_l2_threshold": -6.0, "cb2_l3_threshold": -8.0,
    "cb2_magnitude_weight": True, "cb2_l1_skip_pct": 0.0}

CB21_L3 = {**BASE, "cb2_enabled": True, "cb2_window_trades": 10,
    "cb2_l1_threshold": -999.0, "cb2_l2_threshold": -999.0, "cb2_l3_threshold": -8.0,
    "cb2_magnitude_weight": True, "cb2_l1_skip_pct": 0.0}

print("Compare p2501 signal filtering...")

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
        
        print("\n--- CB2.1 (L2=only 3B) ---")
        r2 = run_backtest(test_contract, start - timedelta(days=1), end + timedelta(days=1),
            CtaChanPivotStrategy, CB21_FULL, **BT_PARAMS)
        s2 = r2.stats or {}
        pnl2, trades2 = s2.get('total_net_pnl', 0), s2.get('total_trade_count', 0)
        print(f"PnL: {pnl2:.1f} pts, Trades: {trades2}")
        
        print("\n--- CB2.1 (L3 only) ---")
        r3 = run_backtest(test_contract, start - timedelta(days=1), end + timedelta(days=1),
            CtaChanPivotStrategy, CB21_L3, **BT_PARAMS)
        s3 = r3.stats or {}
        pnl3, trades3 = s3.get('total_net_pnl', 0), s3.get('total_trade_count', 0)
        print(f"PnL: {pnl3:.1f} pts, Trades: {trades3}")
        
        print("\n=== SUMMARY ===")
        print(f"BASELINE:      PnL={pnl1:8.1f}, Trades={trades1}")
        print(f"CB2.1(L1L2L3): PnL={pnl2:8.1f}, Trades={trades2} (diff: {trades2-trades1})")
        print(f"CB2.1(L3only): PnL={pnl3:8.1f}, Trades={trades3} (diff: {trades3-trades1})")
        
        if trades1 != trades2:
            print(f"\nL1/L2 filtered {trades1-trades2} trades ({(trades1-trades2)/trades1*100:.1f}%)")
        if trades3 == trades1:
            print("L3-only matches BASELINE => L1/L2 is the problem!")
        break
