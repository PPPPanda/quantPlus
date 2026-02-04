"""R1+R4 参数网格：降频相关参数搜索."""
from __future__ import annotations
import sys, json, logging, itertools
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

GRID = {
    "cooldown_losses": [2, 3, 4],
    "cooldown_bars": [20, 30, 40],
    "daily_loss_limit": [0, 100, 150, 200],
    "trade_interval": [0, 2, 4],
}

BASE_SETTING = {
    "debug_enabled": False,
    "debug_log_console": False,
    "div_mode": 1,
    "stop_buffer_atr_pct": 0.02,
    "min_hold_bars": 0,
}

keys = list(GRID.keys())
combos = list(itertools.product(*[GRID[k] for k in keys]))
print(f"Total combos: {len(combos)}")

results = []
best_non209 = -99999

for i, vals in enumerate(combos):
    setting = dict(BASE_SETTING)
    for k, v in zip(keys, vals):
        setting[k] = v

    total_pnl = 0
    p2209_pnl = 0
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
        contract_pts[name] = round(pts, 1)
        if name == "p2209": p2209_pnl = pnl
        if pnl < 0: neg.append(name)

    non209_pts = round((total_pnl - p2209_pnl) / 10, 1)
    total_pts = round(total_pnl / 10, 1)

    row = {
        "combo": dict(zip(keys, vals)),
        "total_pts": total_pts,
        "non209_pts": non209_pts,
        "n_neg": len(neg),
        "neg": neg,
        "contracts": contract_pts,
    }
    results.append(row)

    marker = " ***" if non209_pts > best_non209 else ""
    if non209_pts > best_non209:
        best_non209 = non209_pts
    print(f"[{i+1}/{len(combos)}] non209={non209_pts:>7.1f} total={total_pts:>8.1f} neg={len(neg)}{marker}")

results.sort(key=lambda x: -x["non209_pts"])
print(f"\n=== TOP 5 (non209_pts) ===")
for r in results[:5]:
    print(f"  non209={r['non209_pts']:>7.1f} total={r['total_pts']:>8.1f} neg={r['n_neg']} {r['combo']}")

out = ROOT / "experiments/iter4/grid_r1r4.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(results[:30], f, indent=2, default=str)
print(f"\nSaved top 30: {out}")
