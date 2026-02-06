"""
Iter14: p2401 专项攻关（不做空路线）
在 mpa=3.2 新基线上，测试各种降频/过滤方案对 p2401 的影响
同时验证对其他合约的副作用
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

# NEW baseline: mpa=3.2
NEW_BASE = {
    "circuit_breaker_losses": 7,
    "circuit_breaker_bars": 70,
    "div_threshold": 0.39,
    "max_pullback_atr": 3.2,
}

def run_single(csv_path, vt_symbol, extra_setting):
    setting = {**NEW_BASE, **extra_setting, "debug_enabled": False, "debug_log_console": False}
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

# Phase 1: p2401 only search
P2401 = next(b for b in BENCHMARKS if "p2401" in b["contract"])

# Key insight from diagnosis: p2401 problems are
# 1. High-price entries (7200-7500: 31 trades, -966pts)
# 2. Sept disaster (15 trades, -894pts)  
# 3. Cross-holiday position (09-27→10-10, -224pts)

CONFIGS = {
    "baseline": {},
    # ATR percentage threshold (don't trade when ATR/price is too low)
    "atr_pct_005": {"min_atr_pct": 0.005},
    "atr_pct_006": {"min_atr_pct": 0.006},
    "atr_pct_007": {"min_atr_pct": 0.007},
    # Wider stop buffer (reduce noise stops)
    "sb005": {"stop_buffer_atr_pct": 0.05},
    "sb008": {"stop_buffer_atr_pct": 0.08},
    "sb010": {"stop_buffer_atr_pct": 0.10},
    # More aggressive trailing (lock profits faster)
    "trail25_act20": {"atr_trailing_mult": 2.5, "atr_activate_mult": 2.0},
    # Cooldown adjustment
    "cd3_30": {"cooldown_losses": 3, "cooldown_bars": 30},
    "cd3_40": {"cooldown_losses": 3, "cooldown_bars": 40},
    # Min hold bars (don't exit too early)
    "mhb3": {"min_hold_bars": 3},
    "mhb4": {"min_hold_bars": 4},
    # Stricter entry: higher ATR multiplier for entry filter
    "ef25": {"atr_entry_filter": 2.5},
    "ef30": {"atr_entry_filter": 3.0},
    # Reduce max pivot entries
    "mpe1": {"max_pivot_entries": 1},
    # Combined approaches
    "sb005_cd3_30": {"stop_buffer_atr_pct": 0.05, "cooldown_losses": 3, "cooldown_bars": 30},
    "sb008_mhb3": {"stop_buffer_atr_pct": 0.08, "min_hold_bars": 3},
    "cd3_30_mpe1": {"cooldown_losses": 3, "cooldown_bars": 30, "max_pivot_entries": 1},
    "sb005_mpe1_cd3_30": {"stop_buffer_atr_pct": 0.05, "max_pivot_entries": 1, "cooldown_losses": 3, "cooldown_bars": 30},
    "atr006_sb005": {"min_atr_pct": 0.006, "stop_buffer_atr_pct": 0.05},
}

def main():
    # Phase 1: p2401 single contract
    print("=== p2401 Attack (on mpa=3.2 baseline) ===")
    print(f"{'Config':<25} {'pts':>8} {'improve':>8}")
    print("-" * 45)
    
    results = []
    base_pts = None
    
    for name, setting in CONFIGS.items():
        pts = run_single(P2401["csv"], P2401["contract"], setting)
        if name == "baseline":
            base_pts = pts
        improve = pts - base_pts
        print(f"{name:<25} {pts:>+8.1f} {improve:>+8.1f}")
        results.append({"name": name, "pts": pts, "improve": improve, "setting": setting})
    
    # Top 5
    results.sort(key=lambda x: x['pts'], reverse=True)
    top5 = [r for r in results if r['name'] != 'baseline'][:5]
    print(f"\nTop 5: {[(t['name'], t['pts']) for t in top5]}")
    
    # Full 13 validation for top 3
    top3 = top5[:3]
    print(f"\n=== Full 13 validation ===")
    
    for cfg in top3:
        total = 0
        neg = []
        below = []
        key = {}
        for b in BENCHMARKS:
            pts = run_single(b['csv'], b['contract'], cfg['setting'])
            cn = b['contract'].split('.')[0]
            total += pts
            key[cn] = round(pts, 1)
            if pts < 0: neg.append(f"{cn}({pts:.1f})")
            if pts < -180: below.append(cn)
        
        status = "PASS" if total >= 12164 and not below else "FAIL"
        print(f"{cfg['name']}: TOTAL={total:.1f} p2401={key['p2401']:+.1f} p2305={key['p2305']:+.1f} p2209={key['p2209']:+.1f} p2601={key['p2601']:+.1f} below180={below} {status}")
        
        with open(PROJECT / f"experiments/iter14/test_{cfg['name']}_full.json", 'w') as f:
            json.dump({"setting": {**NEW_BASE, **cfg['setting']}, "total": round(total,1), "details": key, "neg": neg, "below_180": below, "status": status}, f, indent=2, default=str)
    
    with open(PROJECT / "experiments/iter14/p2401_attack_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
