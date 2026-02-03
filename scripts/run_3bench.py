"""
3 合约基准回测 + 汇总输出.

用法：
    cd quantPlus
    .venv/Scripts/python.exe scripts/run_3bench.py [--setting key=val ...]

输出：experiments/iter{N}/ 下的 baseline_metrics.csv + JSON
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 确保项目路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from vnpy.trader.constant import Interval

# 设置日志级别减少刷屏
logging.getLogger("vnpy").setLevel(logging.WARNING)
logging.getLogger("qp").setLevel(logging.WARNING)

from qp.backtest.engine import run_backtest
from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy
from qp.datafeed.normalizer import normalize_1m_bars, PALM_OIL_SESSIONS

# ---------- 基准合约定义 ----------
BENCHMARKS = [
    {
        "contract": "p2601.DCE",
        "csv": ROOT / "data" / "analyse" / "p2601_1min_202507-202512.csv",
        "source": "XT",
        "slot": "标杆",
    },
    {
        "contract": "p2405.DCE",
        "csv": ROOT / "data" / "analyse" / "wind" / "p2405_1min_202312-202404.csv",
        "source": "Wind",
        "slot": "修复",
    },
    {
        "contract": "p2209.DCE",
        "csv": ROOT / "data" / "analyse" / "wind" / "p2209_1min_202204-202208.csv",
        "source": "Wind",
        "slot": "压力",
    },
]

# ---------- 回测口径（冻结） ----------
BT_PARAMS = dict(
    interval=Interval.MINUTE,
    rate=0.0001,
    slippage=1.0,
    size=10.0,
    pricetick=2.0,
    capital=1_000_000.0,
)

STRATEGY_SETTING = {
    "debug_enabled": True,
    "debug_log_console": False,
}


def import_csv_to_db(csv_path: Path, vt_symbol: str) -> tuple[datetime, datetime, int]:
    """读取 CSV → 归一化 → 导入 vnpy 数据库，返回 (start, end, bar_count)."""
    from vnpy.trader.database import get_database
    from vnpy.trader.object import BarData
    from vnpy.trader.constant import Exchange

    db = get_database()
    symbol, exchange_str = vt_symbol.split(".")
    exchange = Exchange(exchange_str)

    # 清除旧数据
    db.delete_bar_data(symbol, exchange, Interval.MINUTE)

    # 读取并归一化
    from zoneinfo import ZoneInfo
    CN_TZ = ZoneInfo("Asia/Shanghai")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = normalize_1m_bars(df, PALM_OIL_SESSIONS)
    df.sort_values("datetime", inplace=True)
    df.drop_duplicates(subset=["datetime"], keep="first", inplace=True)

    # vnpy sqlite 要求 tz-aware datetime
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(CN_TZ)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(CN_TZ)

    # 转换为 BarData 列表
    bars = []
    for _, row in df.iterrows():
        dt = row["datetime"]
        # pandas Timestamp → Python datetime (sqlite 需要)
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=Interval.MINUTE,
            volume=float(row.get("volume", 0)),
            turnover=float(row.get("turnover", 0)),
            open_interest=float(row.get("open_interest", 0)),
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            gateway_name="DB",
        )
        bars.append(bar)

    db.save_bar_data(bars)
    start = df["datetime"].min().to_pydatetime()
    end = df["datetime"].max().to_pydatetime()
    return start, end, len(bars)


def run_single(bench: dict, setting: dict) -> dict:
    """跑单合约回测，返回结果 dict."""
    vt_symbol = bench["contract"]
    csv_path = bench["csv"]

    print(f"\n{'='*60}")
    print(f"[{bench['slot']}] {vt_symbol} ({bench['source']})")
    print(f"{'='*60}")

    # 导入数据
    t0 = time.time()
    start, end, bar_count = import_csv_to_db(csv_path, vt_symbol)
    print(f"  导入 {bar_count} bars ({start.date()} ~ {end.date()}) [{time.time()-t0:.1f}s]")

    # 回测
    t0 = time.time()
    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        strategy_class=CtaChanPivotStrategy,
        strategy_setting=setting,
        **BT_PARAMS,
    )
    elapsed = time.time() - t0

    stats = result.stats or {}
    row = {
        "contract": vt_symbol,
        "source": bench["source"],
        "slot": bench["slot"],
        "bars": result.history_data_count,
        "start_date": str(stats.get("start_date", "")),
        "end_date": str(stats.get("end_date", "")),
        "total_days": stats.get("total_days", 0),
        "trades": stats.get("total_trade_count", 0),
        "total_return%": stats.get("total_return", 0),
        "annual_return%": stats.get("annual_return", 0),
        "max_dd%": stats.get("max_ddpercent", 0),
        "sharpe": stats.get("sharpe_ratio", 0),
        "total_pnl": stats.get("total_net_pnl", 0),
        "commission": stats.get("total_commission", 0),
        "win_rate%": stats.get("winning_rate", 0),
        "avg_trade_pnl": stats.get("average_trade_pnl", 0) if stats.get("average_trade_pnl") else 0,
        "elapsed_s": round(elapsed, 1),
    }

    print(f"  ret={row['total_return%']:.2f}%  sharpe={row['sharpe']:.2f}  "
          f"pnl={row['total_pnl']:.0f}  trades={row['trades']}  "
          f"maxdd={row['max_dd%']:.2f}%  [{elapsed:.1f}s]")

    return row


def main():
    # 解析额外参数
    extra_setting = dict(STRATEGY_SETTING)
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.lstrip("-")
            # 尝试数值转换
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            extra_setting[k] = v

    print(f"策略参数: {extra_setting}")
    print(f"回测口径: slippage={BT_PARAMS['slippage']}, rate={BT_PARAMS['rate']}, size={BT_PARAMS['size']}")

    results = []
    for bench in BENCHMARKS:
        row = run_single(bench, extra_setting)
        results.append(row)

    # 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    total_pnl = sum(r["total_pnl"] for r in results)
    for r in results:
        print(f"  [{r['slot']}] {r['contract']}: ret={r['total_return%']:.2f}%  "
              f"sharpe={r['sharpe']:.2f}  pnl={r['total_pnl']:.0f}")
    print(f"\n  合计 PnL = {total_pnl:.0f}  (目标 ≥ 2400)")
    print(f"  {'PASS' if total_pnl >= 2400 else 'FAIL'}")

    # 保存结果
    out_dir = Path(sys.argv[0]).resolve().parent.parent / "experiments" / "iter1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    df = pd.DataFrame(results)
    csv_path = out_dir / "baseline_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  结果保存: {csv_path}")

    # JSON
    json_path = out_dir / "baseline_results.json"
    with open(json_path, "w") as f:
        json.dump({"params": extra_setting, "bt_params": {k: str(v) for k, v in BT_PARAMS.items()}, "results": results, "total_pnl": total_pnl}, f, indent=2, default=str)
    print(f"  结果保存: {json_path}")


if __name__ == "__main__":
    main()
