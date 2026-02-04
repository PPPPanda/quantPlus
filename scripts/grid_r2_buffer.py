"""R2: stop_buffer_atr_pct grid search."""
import sys, logging, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
logging.getLogger('vnpy').setLevel(logging.WARNING)
logging.getLogger('qp').setLevel(logging.WARNING)

from run_7bench import BENCHMARKS, BT_PARAMS, DEFAULT_SETTING, import_csv_to_db
from datetime import timedelta
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy

results = []
for pct in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
    setting = dict(DEFAULT_SETTING)
    setting['stop_buffer_atr_pct'] = pct
    total_pnl = 0
    neg = []
    details = []
    for bench in BENCHMARKS:
        vt = bench['contract']
        start, end, _ = import_csv_to_db(bench['csv'], vt)
        r = run_backtest(vt_symbol=vt, start=start - timedelta(days=1), end=end + timedelta(days=1),
                         strategy_class=CtaChanPivotStrategy, strategy_setting=setting, **BT_PARAMS)
        s = r.stats or {}
        pnl = round(s.get('total_net_pnl', 0), 1)
        total_pnl += pnl
        c = vt.split('.')[0]
        if pnl < 0: neg.append(c)
        details.append(f'{c}={pnl:>+7.0f}')
    pts = total_pnl / 10
    row = {'pct': pct, 'pts': pts, 'n_neg': len(neg), 'neg': neg, 'total_pnl': total_pnl}
    results.append(row)
    print(f'pct={pct:.2f} | pts={pts:>8.1f} neg={len(neg)} | {" ".join(details)}')

print('\n=== SORTED ===')
results.sort(key=lambda x: (x['n_neg'], -x['pts']))
for r in results:
    print(f"pct={r['pct']:.2f} | pts={r['pts']:>8.1f} neg={r['n_neg']} | neg: {r['neg']}")

out = ROOT / 'experiments/iter3/p4_rounds/r2_buffer_grid.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {out}')
