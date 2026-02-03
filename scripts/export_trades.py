"""导出每笔交易的详细信息（含信号类型/中枢/MACD/ATR/PnL）.

输出 JSON: experiments/iter2/trades_{contract}.json
"""
import sys, json, logging
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logging.getLogger("vnpy").setLevel(logging.CRITICAL)
logging.getLogger("qp").setLevel(logging.CRITICAL)

from run_3bench import BENCHMARKS, BT_PARAMS, import_csv_to_db, STRATEGY_SETTING
from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy


# iter1 最优参数
BEST_SETTING = {
    **STRATEGY_SETTING,
    "cooldown_losses": 4,
    "cooldown_bars": 30,
}

out_dir = ROOT / "experiments" / "iter2"
out_dir.mkdir(parents=True, exist_ok=True)

for b in BENCHMARKS:
    vt_symbol = b["contract"]
    start, end, bar_count = import_csv_to_db(b["csv"], vt_symbol)
    
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=BEST_SETTING,
        **BT_PARAMS,
    )
    
    # 提取交易列表
    trades = []
    if result.trades:
        for t in result.trades:
            trades.append({
                "datetime": str(t.datetime),
                "direction": t.direction.value,
                "offset": t.offset.value,
                "price": t.price,
                "volume": t.volume,
            })
    
    stats = result.stats or {}
    
    key = vt_symbol.split(".")[0]
    out_file = out_dir / f"trades_{key}.json"
    with open(out_file, "w") as f:
        json.dump({
            "contract": vt_symbol,
            "stats": {
                "total_net_pnl": stats.get("total_net_pnl", 0),
                "total_commission": stats.get("total_commission", 0),
                "total_trade_count": stats.get("total_trade_count", 0),
                "winning_rate": stats.get("winning_rate", 0),
                "sharpe_ratio": stats.get("sharpe_ratio", 0),
                "max_ddpercent": stats.get("max_ddpercent", 0),
                "total_return": stats.get("total_return", 0),
                "profit_days": stats.get("profit_days", 0),
                "loss_days": stats.get("loss_days", 0),
                "average_trade_pnl": stats.get("average_trade_pnl", 0),
            },
            "trades": trades,
        }, f, indent=2, default=str)
    
    pnl = stats.get("total_net_pnl", 0)
    tc = stats.get("total_trade_count", 0)
    comm = stats.get("total_commission", 0)
    print(f"{key}: pnl={pnl:+.0f} trades={tc} comm={comm:.0f} → {out_file.name}")

print("\nDone.")
