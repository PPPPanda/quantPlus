"""ATR 参数网格搜索."""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 直接复用 run_3bench 的逻辑
import importlib.util
spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "run_3bench.py")
bench_mod = importlib.util.module_from_spec(spec)

# 我们不执行 main，而是直接调用 run_single
import logging
logging.getLogger("vnpy").setLevel(logging.CRITICAL)
logging.getLogger("qp").setLevel(logging.CRITICAL)

sys.path.insert(0, str(ROOT / "scripts"))
from run_3bench import BENCHMARKS, BT_PARAMS, import_csv_to_db, STRATEGY_SETTING
from datetime import timedelta
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy

def run_one(bench, setting):
    vt_symbol = bench["contract"]
    start, end, bar_count = import_csv_to_db(bench["csv"], vt_symbol)
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        **BT_PARAMS,
    )
    stats = result.stats or {}
    return stats.get("total_net_pnl", 0)

# Pre-import all data once
print("Pre-importing data...")
for b in BENCHMARKS:
    import_csv_to_db(b["csv"], b["contract"])

results = []
for act in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for trail in [2.0, 3.0, 4.0, 5.0, 6.0]:
        setting = dict(STRATEGY_SETTING)
        setting["atr_activate_mult"] = act
        setting["atr_trailing_mult"] = trail
        
        pnls = []
        for b in BENCHMARKS:
            pnl = run_one(b, setting)
            pnls.append(pnl)
        
        total = sum(pnls)
        row = {"act": act, "trail": trail, "p2601": pnls[0], "p2405": pnls[1], "p2209": pnls[2], "total": total}
        results.append(row)
        print(f"act={act} trail={trail}: p2601={pnls[0]:+.0f} p2405={pnls[1]:+.0f} p2209={pnls[2]:+.0f} total={total:+.0f}")

# 排序并输出 top 5
results.sort(key=lambda x: x["total"], reverse=True)
print("\n=== TOP 5 ===")
for r in results[:5]:
    print(f"  act={r['act']} trail={r['trail']}: total={r['total']:+.0f} "
          f"(p2601={r['p2601']:+.0f} p2405={r['p2405']:+.0f} p2209={r['p2209']:+.0f})")

# 保存
out = ROOT / "experiments" / "iter1" / "p4_round2" / "atr_grid.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
