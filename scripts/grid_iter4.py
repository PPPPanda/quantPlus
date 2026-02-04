"""iter4: 多参数网格搜索 — 目标: 除p2209的6合约总pts > 5600."""
from __future__ import annotations
import sys, time, json, logging, itertools
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from run_7bench import BENCHMARKS, BT_PARAMS, import_csv_to_db
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy

# 搜索空间
GRID = {
    "cooldown_losses": [2, 3],
    "cooldown_bars": [15, 20, 30],
    "atr_trailing_mult": [2.0, 2.5, 3.0],
    "atr_activate_mult": [1.5, 2.0, 2.5],
    "atr_entry_filter": [1.5, 2.0, 3.0],
}

BASE_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "div_mode": 1,
    "stop_buffer_atr_pct": 0.02,
}

keys = list(GRID.keys())
combos = list(itertools.product(*[GRID[k] for k in keys]))
print(f"Total combos: {len(combos)}")

results = []
best_non209 = -99999
best_combo = None

for i, vals in enumerate(combos):
    setting = dict(BASE_SETTING)
    for k, v in zip(keys, vals):
        setting[k] = v

    total_pnl = 0
    p2209_pnl = 0
    p2601_pnl = 0
    neg = []
    contract_pts = {}

    for bench in BENCHMARKS:
        vt = bench["contract"]
        name = vt.split(".")[0]
        start, end, _ = import_csv_to_db(bench["csv"], vt)
        r = run_backtest(vt_symbol=vt, start=start - timedelta(days=1), end=end + timedelta(days=1),
                         strategy_class=CtaChanPivotStrategy, strategy_setting=setting, **BT_PARAMS)
        s = r.stats or {}
        pnl = round(s.get("total_net_pnl", 0), 1)
        pts = pnl / 10
        total_pnl += pnl
        contract_pts[name] = pts
        if name == "p2209": p2209_pnl = pnl
        if name == "p2601": p2601_pnl = pnl
        if pnl < 0: neg.append(name)

    non209_pts = (total_pnl - p2209_pnl) / 10
    p2601_pts = p2601_pnl / 10
    total_pts = total_pnl / 10

    row = {
        "combo": dict(zip(keys, vals)),
        "total_pts": round(total_pts, 1),
        "non209_pts": round(non209_pts, 1),
        "p2601_pts": round(p2601_pts, 1),
        "p2209_pts": round(p2209_pnl / 10, 1),
        "n_neg": len(neg),
        "neg": neg,
        "contracts": contract_pts,
    }
    results.append(row)

    status = "✓" if non209_pts > 5600 and p2601_pts >= 1000 and p2209_pnl / 10 >= 1000 else " "
    marker = " ***BEST***" if non209_pts > best_non209 else ""
    if non209_pts > best_non209:
        best_non209 = non209_pts
        best_combo = row
    print(f"[{i+1}/{len(combos)}]{status} non209={non209_pts:>7.1f} total={total_pts:>8.1f} neg={len(neg)} p2601={p2601_pts:>+7.1f}{marker}")

print(f"\n=== BEST (non209_pts) ===")
print(json.dumps(best_combo, indent=2, default=str))

# Save top 20
results.sort(key=lambda x: -x["non209_pts"])
out = ROOT / "experiments/iter4/grid_multi.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(results[:50], f, indent=2, default=str)
print(f"\nSaved top 50: {out}")
