"""Cooldown parameter grid search."""
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
for losses in [0, 2, 3, 4, 5, 6]:
    for bars in [10, 20, 30, 50, 80]:
        if losses == 0 and bars != 10:
            continue  # skip redundant combos when disabled
        setting = dict(DEFAULT_SETTING)
        setting['cooldown_losses'] = losses
        setting['cooldown_bars'] = bars
        total_pnl = 0
        neg = []
        details = []
        p2401_pnl = 0
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
            if c == 'p2401': p2401_pnl = pnl
            details.append(f'{c}={pnl:>+7.0f}')
        pts = total_pnl / 10
        row = {'losses': losses, 'bars': bars, 'pts': pts, 'n_neg': len(neg),
               'p2401': p2401_pnl, 'neg': neg, 'total_pnl': total_pnl}
        results.append(row)
        print(f'L={losses} B={bars:>2} | pts={pts:>8.1f} neg={len(neg)} p2401={p2401_pnl:>+7.0f} | {" ".join(details)}')

print('\n=== SORTED (fewest neg, highest pts) ===')
results.sort(key=lambda x: (x['n_neg'], -x['pts']))
for r in results[:10]:
    print(f"L={r['losses']} B={r['bars']:>2} | pts={r['pts']:>8.1f} neg={r['n_neg']} p2401={r['p2401']:>+7.0f} | neg: {r['neg']}")

out = ROOT / 'experiments/iter3/p4_rounds/cooldown_grid.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {out}')
