"""B03 止损 buffer 参数网格搜索.

基于 Round 3 最优参数 (act=2.5, trail=3.0, cooldown_losses=4, cooldown_bars=30)，
搜索 stop_buffer_atr_pct 的最优值。
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import logging
logging.getLogger("vnpy").setLevel(logging.CRITICAL)
logging.getLogger("qp").setLevel(logging.CRITICAL)

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


# Round 3 最优基准参数
BASE_SETTING = {
    **STRATEGY_SETTING,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
    "cooldown_losses": 4,
    "cooldown_bars": 30,
}

# Pre-import data
print("Pre-importing data...")
for b in BENCHMARKS:
    import_csv_to_db(b["csv"], b["contract"])

# 先跑 baseline（stop_buffer_atr_pct=0.05 默认值，但代码之前是硬编码±1，现在改成动态了）
# 也跑一个 pct=0 看硬编码 ±2（pricetick）的效果
results = []

for pct in [0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
    setting = dict(BASE_SETTING)
    setting["stop_buffer_atr_pct"] = pct
    
    pnls = {}
    for b in BENCHMARKS:
        pnl = run_one(b, setting)
        key = b["contract"].split(".")[0]
        pnls[key] = pnl
    
    total = sum(pnls.values())
    row = {"pct": pct, **pnls, "total": total}
    results.append(row)
    pnl_str = " ".join(f"{k}={v:+.0f}" for k, v in pnls.items())
    print(f"pct={pct:.2f}: {pnl_str} total={total:+.0f}")

results.sort(key=lambda x: x["total"], reverse=True)

print(f"\n=== RESULTS (sorted) ===")
for r in results:
    print(f"  pct={r['pct']:.2f}: total={r['total']:+.0f} "
          f"(p2601={r['p2601']:+.0f} p2405={r['p2405']:+.0f} p2209={r['p2209']:+.0f})")

# 保存
out = ROOT / "experiments" / "iter1" / "p4_round4" / "buffer_grid.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
