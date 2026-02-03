"""
vnpy 回测引擎验证 - 原始策略 × 7合约

使用 vnpy BacktestingEngine 对 CtaChanPivotStrategy 进行标准回测，
验证原始策略在 7 个合约上的表现。

运行方式（必须在 Windows .venv 下）：
    cd E:\clawdbot_bridge\clawdbot_workspace\work\quant\quantPlus
    .\.venv\Scripts\python.exe scripts/vnpy_backtest_7contracts.py

输出：
    experiments/<timestamp>_vnpy_bt/results.json
    experiments/<timestamp>_vnpy_bt/trades/  (每合约成交明细)
"""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 确保项目 src 在 sys.path 中
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(PROJ_ROOT / "strategies"))

import pandas as pd
from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine

from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy


def _print(*args, **kwargs):
    """强制刷新的 print."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ─── 合约列表 & 数据映射 ───────────────────────────────────────────

CONTRACTS = [
    {"name": "P2201", "file": "p2201_1min_202108-202112.csv", "symbol": "p2201", "exchange": Exchange.DCE},
    {"name": "P2205", "file": "p2205_1min_202112-202204.csv", "symbol": "p2205", "exchange": Exchange.DCE},
    {"name": "P2401", "file": "p2401_1min_202308-202312.csv", "symbol": "p2401", "exchange": Exchange.DCE},
    {"name": "P2405", "file": "p2405_1min_202312-202404.csv", "symbol": "p2405", "exchange": Exchange.DCE},
    {"name": "P2505", "file": "p2505_1min_202501-202504.csv", "symbol": "p2505", "exchange": Exchange.DCE},
    {"name": "P2509", "file": "p2509_1min_202503-202508.csv", "symbol": "p2509", "exchange": Exchange.DCE},
    {"name": "P2601", "file": "p2601_1min_202507-202512.csv", "symbol": "p2601", "exchange": Exchange.DCE},
]

DATA_DIR = PROJ_ROOT / "data" / "analyse"


def load_csv_as_bardata(csv_path: Path, symbol: str, exchange: Exchange) -> list[BarData]:
    """从 CSV 加载 1min 数据并转换为 vnpy BarData 列表."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    bars = []
    for _, row in df.iterrows():
        # 确保 datetime 是 Python datetime（不是 pandas Timestamp）
        dt = row["datetime"]
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=Interval.MINUTE,
            volume=float(row.get("volume", 0)),
            turnover=0.0,
            open_interest=float(row.get("open_interest", 0)) if "open_interest" in row else 0.0,
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            gateway_name="BACKTEST",
        )
        bars.append(bar)
    return bars


def run_vnpy_backtest(
    contract: dict,
    strategy_setting: dict | None = None,
) -> dict:
    """对单个合约运行 vnpy 回测引擎."""

    csv_path = DATA_DIR / contract["file"]
    if not csv_path.exists():
        return {"error": f"CSV not found: {csv_path}"}

    _print(f"  Loading {contract['name']} from {csv_path.name} ...", end=" ")
    bars = load_csv_as_bardata(csv_path, contract["symbol"], contract["exchange"])
    _print(f"({len(bars)} bars)")

    if not bars:
        return {"error": "No bars loaded"}

    # 设置回测引擎
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=f"{contract['symbol']}.{contract['exchange'].value}",
        interval=Interval.MINUTE,
        start=bars[0].datetime - timedelta(days=1),
        end=bars[-1].datetime + timedelta(days=1),
        rate=0.0001,       # 手续费率 0.01%
        slippage=0,         # 无滑点（与脚本回测一致）
        size=10,            # 棕榈油合约乘数 10吨/手
        pricetick=2.0,      # 最小变动价位
        capital=1_000_000,  # 初始资金
    )

    # 加载策略（使用默认参数 = S6 参数）
    setting = strategy_setting or {}
    setting.setdefault("debug_enabled", False)  # 回测时关闭 debug 文件
    engine.add_strategy(CtaChanPivotStrategy, setting)

    # 直接注入数据（不走数据库）— 绕过 load_data() 的 DB 查询
    engine.history_data = bars

    # 绕过 on_init() 中的 load_bar()（它会查 DB），
    # 直接设置策略状态为已初始化+已启动
    strategy = engine.strategy

    # 手动创建 BarGenerator（on_init 中会做的事）
    from vnpy.trader.utility import BarGenerator as BG
    strategy.bg = BG(strategy._on_1m_bar)

    # 关闭 debug 文件输出
    strategy.debug_enabled = False
    strategy._debugger = None

    strategy.inited = True
    strategy.trading = True

    # 运行回测
    engine.run_backtesting()

    # 计算结果
    engine.calculate_result()
    stats = engine.calculate_statistics(output=False) or {}

    # 获取成交列表
    trades = engine.get_all_trades()

    # 聚合 round-trip 信息
    trade_records = []
    for t in trades:
        trade_records.append({
            "datetime": str(t.datetime),
            "direction": t.direction.value,
            "offset": t.offset.value,
            "price": t.price,
            "volume": t.volume,
        })

    # 计算简单 round-trip PnL
    round_trip_pnl = 0.0
    round_trips = 0
    wins = 0
    open_price = 0.0
    open_dir = 0  # 1=long, -1=short

    from vnpy.trader.constant import Direction, Offset
    for t in trades:
        if t.offset == Offset.OPEN:
            open_price = t.price
            open_dir = 1 if t.direction == Direction.LONG else -1
        elif t.offset in (Offset.CLOSE, Offset.CLOSETODAY, Offset.CLOSEYESTERDAY):
            if open_dir != 0:
                pnl = (t.price - open_price) * open_dir
                round_trip_pnl += pnl
                round_trips += 1
                if pnl > 0:
                    wins += 1
                open_dir = 0

    win_rate = (wins / round_trips * 100) if round_trips > 0 else 0.0

    result = {
        "contract": contract["name"],
        "vnpy_stats": {
            "total_net_pnl": stats.get("total_net_pnl", 0),
            "total_trade_count": stats.get("total_trade_count", 0),
            "max_drawdown": stats.get("max_drawdown", 0),
            "sharpe_ratio": stats.get("sharpe_ratio", 0),
            "total_return": stats.get("total_return", 0),
        },
        "round_trip": {
            "count": round_trips,
            "total_pnl": round_trip_pnl,
            "win_rate": win_rate,
        },
        "bars_count": len(bars),
        "trades_count": len(trades),
        "trade_records": trade_records,
    }

    _print(f"    -> vnpy PnL={stats.get('total_net_pnl', 0):.1f}, "
           f"trades={stats.get('total_trade_count', 0)}, "
           f"RT={round_trips}, RT_PnL={round_trip_pnl:.1f}, "
           f"WR={win_rate:.1f}%")

    return result


def main():
    _print("=" * 70)
    _print("  vnpy BacktestingEngine - 原始策略 x 7合约")
    _print(f"  策略: CtaChanPivotStrategy (S6 默认参数)")
    _print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _print("=" * 70)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJ_ROOT / "experiments" / f"{timestamp}_vnpy_bt"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = out_dir / "trades"
    trades_dir.mkdir(exist_ok=True)

    all_results = {}
    total_pnl = 0.0
    total_rt_pnl = 0.0

    for c in CONTRACTS:
        _print(f"\n--- {c['name']} ---")
        try:
            result = run_vnpy_backtest(c)
        except Exception as e:
            _print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            result = {"error": str(e)}

        all_results[c["name"]] = result

        # 保存成交明细
        if "trade_records" in result:
            trades_file = trades_dir / f"{c['name']}_trades.json"
            with open(trades_file, "w", encoding="utf-8") as f:
                json.dump(result["trade_records"], f, indent=2, ensure_ascii=False)
            # 从结果中移除大量明细（汇总文件更简洁）
            result["trades_file"] = str(trades_file.relative_to(PROJ_ROOT))
            del result["trade_records"]

        if "vnpy_stats" in result:
            total_pnl += result["vnpy_stats"]["total_net_pnl"]
        if "round_trip" in result:
            total_rt_pnl += result["round_trip"]["total_pnl"]

    # 汇总
    _print("\n" + "=" * 70)
    _print("  汇总")
    _print("=" * 70)
    _print(f"  vnpy total_net_pnl (含手续费): {total_pnl:.1f}")
    _print(f"  Round-trip PnL (不含手续费):   {total_rt_pnl:.1f}")
    _print()

    for name, r in all_results.items():
        if "vnpy_stats" in r:
            s = r["vnpy_stats"]
            rt = r.get("round_trip", {})
            ap = "+" if s["total_net_pnl"] > 0 else ""
            _print(f"  {name}: vnpy={ap}{s['total_net_pnl']:.1f}  "
                   f"RT_PnL={rt.get('total_pnl', 0):+.1f}  "
                   f"RT={rt.get('count', 0)}  "
                   f"WR={rt.get('win_rate', 0):.0f}%")
        else:
            _print(f"  {name}: ERROR - {r.get('error', 'unknown')}")

    # 保存结果
    summary = {
        "timestamp": timestamp,
        "strategy": "CtaChanPivotStrategy",
        "params": "S6 defaults",
        "total_vnpy_pnl": total_pnl,
        "total_rt_pnl": total_rt_pnl,
        "contracts": all_results,
    }

    results_file = out_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    _print(f"\n  结果已保存: {results_file}")


if __name__ == "__main__":
    main()
