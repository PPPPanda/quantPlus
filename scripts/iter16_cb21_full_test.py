"""iter16: CB2.1 时间驱动恢复修复 - 全13合约测试"""
import sys
from datetime import timedelta
import json
sys.path.insert(0, "E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus")
sys.path.insert(0, "E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/scripts")

from pathlib import Path
from run_13bench import import_csv_to_db, BENCHMARKS, BT_PARAMS
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy

# 基线参数（禁用调试加速回测）
BASE = {"circuit_breaker_losses": 7, "circuit_breaker_bars": 70, "div_threshold": 0.39, "max_pullback_atr": 3.2, "debug": False, "debug_enabled": False}

# CB2.1 + 时间驱动恢复（与简单CB一致）
CB21_FIXED = {**BASE, "cb2_enabled": True, "cb2_window_trades": 10,
    "cb2_l1_threshold": -4.0, "cb2_l2_threshold": -6.0, "cb2_l3_threshold": -8.0,
    "cb2_magnitude_weight": True, "cb2_l1_skip_pct": 0.0,
    "cb2_max_pause_bars": 70, "debug": False, "debug_enabled": False}

results = {"baseline": {}, "cb21_fixed": {}}

print("=" * 60)
print("iter16: CB2.1 Time-Based Recovery Fix - Full 13 Contract Test")
print("=" * 60)

for bench in BENCHMARKS:
    contract = bench.get("contract", "")
    csv_path = Path(bench["csv"])
    
    try:
        start, end, _ = import_csv_to_db(csv_path, contract)
    except Exception as e:
        print(f"Skip {contract}: {e}")
        continue
    
    # Baseline
    r1 = run_backtest(contract, start - timedelta(days=1), end + timedelta(days=1),
        CtaChanPivotStrategy, BASE, **BT_PARAMS)
    s1 = r1.stats or {}
    pnl1, trades1 = s1.get('total_net_pnl', 0), s1.get('total_trade_count', 0)
    
    # CB2.1 Fixed
    r2 = run_backtest(contract, start - timedelta(days=1), end + timedelta(days=1),
        CtaChanPivotStrategy, CB21_FIXED, **BT_PARAMS)
    s2 = r2.stats or {}
    pnl2, trades2 = s2.get('total_net_pnl', 0), s2.get('total_trade_count', 0)
    
    sym = contract.split(".")[0]
    results["baseline"][sym] = {"pnl": pnl1, "trades": trades1}
    results["cb21_fixed"][sym] = {"pnl": pnl2, "trades": trades2}
    
    delta = pnl2 - pnl1
    status = "OK" if abs(delta) < 100 else ("BETTER" if delta > 0 else "WORSE")
    print(f"  {sym}: BASE={pnl1:8.1f} FIXED={pnl2:8.1f} delta={delta:+8.1f} [{status}]")

# Summary
base_total = sum(v["pnl"] for v in results["baseline"].values())
fixed_total = sum(v["pnl"] for v in results["cb21_fixed"].values())

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"BASELINE TOTAL: {base_total:.1f} pts")
print(f"CB2.1 FIXED TOTAL: {fixed_total:.1f} pts")
print(f"DELTA: {fixed_total - base_total:+.1f} pts")

# Check if fix is working (should be very close to baseline)
if abs(fixed_total - base_total) < 500:
    print("\n[SUCCESS] CB2.1 FIXED matches BASELINE!")
else:
    print(f"\n[WARNING] CB2.1 FIXED differs from BASELINE by {abs(fixed_total - base_total):.1f} pts")

# Save results
out_path = Path("experiments/iter16/cb21_fixed_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"baseline_total": base_total, "cb21_fixed_total": fixed_total, 
               "delta": fixed_total - base_total, "by_contract": results}, f, indent=2, default=float)
print(f"\nResults saved to {out_path}")
