"""B02 连亏冷却参数网格搜索.

基于 Round 2 最优 ATR 参数 (act=2.5, trail=3.0)，
搜索 cooldown_losses × cooldown_bars 的最优组合。
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
    return {
        "pnl": stats.get("total_net_pnl", 0),
        "trades": stats.get("total_trade_count", 0),
        "max_dd": stats.get("max_drawdown", 0),
        "sharpe": stats.get("sharpe_ratio", 0),
    }


# Round 2 最优基准参数
BASE_SETTING = {
    **STRATEGY_SETTING,
    "atr_activate_mult": 2.5,
    "atr_trailing_mult": 3.0,
}

# Pre-import data
print("Pre-importing data...")
for b in BENCHMARKS:
    import_csv_to_db(b["csv"], b["contract"])

# 先跑一次 baseline（cooldown_losses=0，即无冷却）
print("\n=== BASELINE (no cooldown) ===")
baseline_pnls = []
for b in BENCHMARKS:
    r = run_one(b, BASE_SETTING)
    baseline_pnls.append(r["pnl"])
    print(f"  {b['contract']}: pnl={r['pnl']:+.0f} trades={r['trades']} sharpe={r['sharpe']:.2f}")
baseline_total = sum(baseline_pnls)
print(f"  TOTAL: {baseline_total:+.0f}")

# 网格搜索
results = []

# cooldown_losses: 连亏多少笔触发
# cooldown_bars: 冷却多少根 5m bar
for losses in [2, 3, 4, 5]:
    for bars in [5, 10, 15, 20, 30]:
        setting = dict(BASE_SETTING)
        setting["cooldown_losses"] = losses
        setting["cooldown_bars"] = bars
        
        pnls = {}
        details = {}
        for b in BENCHMARKS:
            r = run_one(b, setting)
            key = b["contract"].split(".")[0]
            pnls[key] = r["pnl"]
            details[key] = r
        
        total = sum(pnls.values())
        row = {
            "losses": losses,
            "bars": bars,
            **pnls,
            "total": total,
            "delta": total - baseline_total,
            "details": details,
        }
        results.append(row)
        pnl_str = " ".join(f"{k}={v:+.0f}" for k, v in pnls.items())
        print(f"L={losses} B={bars}: {pnl_str} total={total:+.0f} (Δ{total-baseline_total:+.0f})")

# 排序
results.sort(key=lambda x: x["total"], reverse=True)

print(f"\n=== TOP 5 (baseline={baseline_total:+.0f}) ===")
for r in results[:5]:
    print(f"  L={r['losses']} B={r['bars']}: total={r['total']:+.0f} (Δ{r['delta']:+.0f})")

print(f"\n=== WORST 3 ===")
for r in results[-3:]:
    print(f"  L={r['losses']} B={r['bars']}: total={r['total']:+.0f} (Δ{r['delta']:+.0f})")

# 保存
out = ROOT / "experiments" / "iter1" / "p4_round3" / "cooldown_grid.json"
out.parent.mkdir(parents=True, exist_ok=True)
# 去掉 details（太大），单独保存
slim = [{k: v for k, v in r.items() if k != "details"} for r in results]
with open(out, "w") as f:
    json.dump(slim, f, indent=2)
print(f"\nSaved: {out}")
