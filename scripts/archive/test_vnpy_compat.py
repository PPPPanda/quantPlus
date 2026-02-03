"""Quick test: run vnpy-compat on one contract to verify it works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Importing...")
try:
    from iteration_vnpy_compat import ChanPivotTesterVnpyCompat, run_one, build_configs
    from iteration_v6 import StrategyParams, load_csv, calc_stats
    print("Import OK")
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

data_dir = Path(r"E:/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus/data/analyse")
contract = "p2401"

print(f"Loading {contract}...")
matches = list(data_dir.glob(f"{contract}_1min_*.csv"))
if not matches:
    print(f"No CSV for {contract}")
    sys.exit(1)

df = load_csv(matches[0])
print(f"Loaded {len(df)} bars")

# S6 params
p = StrategyParams("S6_test",
    activate_atr=1.5, trail_atr=2.0, entry_filter_atr=1.5,
    min_bi_gap=5, trend_filter=True, disable_3s_short=True,
    bi_amp_filter=True, bi_amp_min_atr=1.5, macd_consistency=3,
    macd15_mag_cap_atr=3.0)

print("Running vnpy-compat tester...")
tester = ChanPivotTesterVnpyCompat(df, p)
trades = tester.run()
s = calc_stats(trades)
print(f"Result: PnL={s['pnl']:.1f}, Trades={s['trades']}, Win={s['win%']:.1f}%")
print("Done!")
