"""
Iter15: CB2.0 增强版断路器测试
目标：改善 p2201/p2305/p2309 且 p2401/TOTAL 不退化
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from vnpy.trader.constant import Interval
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from run_13bench import import_csv_to_db, BENCHMARKS

PROJECT = Path(__file__).resolve().parents[1]

# 当前最优基线
BASE = {
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
}

def run13(extra):
    total = 0; key = {}; neg = []; below = []
    for b in BENCHMARKS:
        start, end, _ = import_csv_to_db(b['csv'], b['contract'])
        r = run_backtest(
            vt_symbol=b['contract'], start=start, end=end,
            strategy_class=CtaChanPivotStrategy,
            strategy_setting={**BASE, **extra, "debug_enabled": False, "debug_log_console": False},
            interval=Interval.MINUTE, rate=0.0001, slippage=1.0, size=10.0, pricetick=2.0, capital=1_000_000
        )
        pts = r.stats.get('total_net_pnl', 0) / 10
        cn = b['contract'].split('.')[0]
        total += pts
        key[cn] = round(pts, 1)
        if pts < 0: neg.append(f"{cn}({pts:.1f})")
        if pts < -180: below.append(cn)
    return total, key, neg, below

# CB2.0 参数网格
CONFIGS = {
    "baseline": {},
    # CB2.0 启用，不同阈值
    "cb2_10_4_6_8": {"cb2_enabled": True, "cb2_window_trades": 10, "cb2_l1_threshold": -4.0, "cb2_l2_threshold": -6.0, "cb2_l3_threshold": -8.0},
    "cb2_10_3_5_7": {"cb2_enabled": True, "cb2_window_trades": 10, "cb2_l1_threshold": -3.0, "cb2_l2_threshold": -5.0, "cb2_l3_threshold": -7.0},
    "cb2_10_5_7_9": {"cb2_enabled": True, "cb2_window_trades": 10, "cb2_l1_threshold": -5.0, "cb2_l2_threshold": -7.0, "cb2_l3_threshold": -9.0},
    # 更大窗口
    "cb2_15_4_6_8": {"cb2_enabled": True, "cb2_window_trades": 15, "cb2_l1_threshold": -4.0, "cb2_l2_threshold": -6.0, "cb2_l3_threshold": -8.0},
    # 更小窗口
    "cb2_8_4_6_8": {"cb2_enabled": True, "cb2_window_trades": 8, "cb2_l1_threshold": -4.0, "cb2_l2_threshold": -6.0, "cb2_l3_threshold": -8.0},
    # 恢复参数
    "cb2_10_4_6_8_r2": {"cb2_enabled": True, "cb2_window_trades": 10, "cb2_l1_threshold": -4.0, "cb2_l2_threshold": -6.0, "cb2_l3_threshold": -8.0, "cb2_recovery_win": 2},
}

def main():
    print("=== Iter15: CB2.0 Test ===")
    
    base_total, base_key, _, _ = run13({})
    base_p2401 = base_key['p2401']
    
    print(f"{'Config':<20} {'TOTAL':>8} {'p2201':>8} {'p2305':>8} {'p2309':>8} {'p2401':>8} {'neg_cnt':>7} {'pass':>5}")
    print("-" * 85)
    
    results = []
    for name, extra in CONFIGS.items():
        total, key, neg, below = run13(extra)
        neg_cnt = len(neg)
        
        # 检查约束：TOTAL不退化，p2401不退化
        total_ok = total >= base_total * 0.99  # 允许1%误差
        p2401_ok = key['p2401'] >= base_p2401 * 0.99
        passed = "PASS" if total_ok and p2401_ok else "FAIL"
        
        print(f"{name:<20} {total:>8.1f} {key['p2201']:>+8.1f} {key['p2305']:>+8.1f} {key['p2309']:>+8.1f} {key['p2401']:>+8.1f} {neg_cnt:>7} {passed:>5}")
        
        results.append({
            "name": name, "total": total, "details": key, "neg": neg, "below_180": below,
            "setting": extra, "total_ok": total_ok, "p2401_ok": p2401_ok
        })
    
    # 找最优配置
    valid = [r for r in results if r['total_ok'] and r['p2401_ok'] and r['name'] != 'baseline']
    if valid:
        # 按 p2201+p2305+p2309 改善排序
        valid.sort(key=lambda r: r['details']['p2201'] + r['details']['p2305'] + r['details']['p2309'], reverse=True)
        best = valid[0]
        print(f"\nBest CB2.0 config: {best['name']}")
        print(f"  Neg contracts: {best['neg']}")
        print(f"  Below -180: {best['below_180']}")
    else:
        print("\nNo valid CB2.0 config found (all degraded TOTAL or p2401)")
    
    with open(PROJECT / "experiments/iter15/cb2_test_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
