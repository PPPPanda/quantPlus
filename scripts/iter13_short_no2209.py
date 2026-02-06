"""
Iter13 续：排除 p2209 后，条件性做空对其他 12 合约的影响
严格按 chan-work.md Phase 4 小循环
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from run_13bench import import_csv_to_db, BENCHMARKS

PROJECT = Path(__file__).resolve().parents[1]

# iter10-full 最优基线参数
BEST_BASE = {
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.0,
}

def run_single(csv_path, vt_symbol, extra_setting):
    setting = {**BEST_BASE, **extra_setting, "debug_enabled": False, "debug_log_console": False}
    start, end, _ = import_csv_to_db(csv_path, vt_symbol)
    result = run_backtest(
        vt_symbol=vt_symbol, start=start, end=end,
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        interval=Interval.MINUTE,
        rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0,
        capital=1_000_000,
    )
    return result.stats.get('total_net_pnl', 0) / 10

# 排除 p2209 的 12 合约
BENCHMARKS_NO2209 = [b for b in BENCHMARKS if "p2209" not in b["contract"]]

CONFIGS = {
    "baseline": {},
    # 条件性做空：2中枢递降
    "short_p2": {"conditional_short": True, "short_min_pivots": 2, "short_max_diff_15m": 0},
    # 条件性做空：2中枢递降 + diff<-5
    "short_p2_diff5": {"conditional_short": True, "short_min_pivots": 2, "short_max_diff_15m": -5},
    # 条件性做空：2中枢递降 + diff<-10（更严格）
    "short_p2_diff10": {"conditional_short": True, "short_min_pivots": 2, "short_max_diff_15m": -10},
    # 条件性做空：3中枢递降
    "short_p3": {"conditional_short": True, "short_min_pivots": 3, "short_max_diff_15m": 0},
    # 条件性做空：3中枢递降 + diff<-5
    "short_p3_diff5": {"conditional_short": True, "short_min_pivots": 3, "short_max_diff_15m": -5},
}

def run_full(extra_setting, name):
    """Run all 12 contracts (no p2209)"""
    total = 0
    details = {}
    neg = []
    below_180 = []
    
    for bench in BENCHMARKS_NO2209:
        vt = bench["contract"]
        csv = bench["csv"]
        pts = run_single(csv, vt, extra_setting)
        cn = vt.split('.')[0]
        total += pts
        details[cn] = round(pts, 1)
        if pts < 0:
            neg.append(f"{cn}({pts:.1f})")
        if pts < -180:
            below_180.append(cn)
    
    return total, details, neg, below_180

def main():
    print("=== Iter13: Conditional Short WITHOUT p2209 (12 contracts) ===\n")
    
    all_results = {}
    
    for config_name, setting in CONFIGS.items():
        total, details, neg, below_180 = run_full(setting, config_name)
        all_results[config_name] = {
            "total": round(total, 1),
            "details": details,
            "neg": neg,
            "below_180": below_180,
            "setting": setting
        }
        
        print(f"=== {config_name} ===")
        print(f"  TOTAL(12): {total:.1f}pts")
        print(f"  Neg: {neg}")
        print(f"  Below -180: {below_180}")
        # Show key contracts
        for k in ['p2401', 'p2601', 'p2405', 'p2201', 'p2305', 'p2309']:
            if k in details:
                print(f"  {k}: {details[k]:+.1f}")
        print()
    
    # Comparison table
    print("\n=== COMPARISON TABLE (12 contracts, no p2209) ===")
    base_total = all_results['baseline']['total']
    base_details = all_results['baseline']['details']
    
    header = f"{'Config':<18} {'TOTAL':>8} {'diff':>7} {'p2401':>8} {'p2601':>8} {'p2405':>8} {'p2201':>8} {'p2305':>8} {'neg_cnt':>7}"
    print(header)
    print("-" * len(header))
    
    for config_name, data in all_results.items():
        d = data['details']
        diff = data['total'] - base_total
        neg_cnt = len(data['neg'])
        print(f"{config_name:<18} {data['total']:>8.1f} {diff:>+7.1f} {d.get('p2401',0):>+8.1f} {d.get('p2601',0):>+8.1f} {d.get('p2405',0):>+8.1f} {d.get('p2201',0):>+8.1f} {d.get('p2305',0):>+8.1f} {neg_cnt:>7}")
    
    # Detailed per-contract comparison: baseline vs best short config
    print("\n=== PER-CONTRACT DELTA (baseline vs short_p2) ===")
    if 'short_p2' in all_results:
        for cn in sorted(base_details.keys()):
            base_v = base_details[cn]
            short_v = all_results['short_p2']['details'].get(cn, 0)
            delta = short_v - base_v
            direction = "better" if delta > 0 else ("same" if delta == 0 else "WORSE")
            print(f"  {cn}: {base_v:>+8.1f} -> {short_v:>+8.1f} ({delta:>+7.1f}) {direction}")
    
    # Save
    # Convert bools for JSON serialization
    def convert(obj):
        if isinstance(obj, bool):
            return int(obj)
        return str(obj)
    with open(PROJECT / "experiments/iter13/short_no2209_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nSaved to experiments/iter13/short_no2209_results.json")

if __name__ == "__main__":
    main()
