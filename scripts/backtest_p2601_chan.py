"""
P2601 缠论策略回测脚本.

使用 XT 数据源的 1 分钟 K 线数据，运行缠论中枢策略回测。

用法:
    cd /mnt/e/clawdbot_bridge/clawdbot_workspace/work/quant/quantPlus
    uv run python scripts/backtest_p2601_chan.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

# 添加项目路径
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy.trader.database import get_database
from vnpy_ctastrategy.backtesting import BacktestingEngine

from qp.strategies.cta_chan_pivot import CtaChanPivotStrategy


def load_csv_data(csv_path: Path) -> list[BarData]:
    """加载 CSV 数据并转换为 vnpy BarData 格式."""
    print(f"[1/4] 加载数据: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 设置中国时区
    china_tz = ZoneInfo("Asia/Shanghai")
    
    bars = []
    for _, row in df.iterrows():
        # 将 naive datetime 转换为带时区的 datetime
        dt = row['datetime'].to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=china_tz)
        
        bar = BarData(
            symbol="p2601",
            exchange=Exchange.DCE,
            datetime=dt,
            interval=Interval.MINUTE,
            volume=float(row['volume']),
            turnover=float(row.get('turnover', 0)),
            open_interest=float(row.get('open_interest', 0)),
            open_price=float(row['open']),
            high_price=float(row['high']),
            low_price=float(row['low']),
            close_price=float(row['close']),
            gateway_name="XT",
        )
        bars.append(bar)
    
    print(f"    加载 {len(bars)} 根 K 线")
    print(f"    时间范围: {bars[0].datetime} ~ {bars[-1].datetime}")
    
    return bars


def save_to_database(bars: list[BarData]) -> int:
    """保存 K 线数据到 vnpy 数据库."""
    if not bars:
        return 0
    
    print(f"[2/4] 保存到数据库...")
    
    database = get_database()
    database.save_bar_data(bars)
    
    print(f"    保存 {len(bars)} 根 K 线")
    
    return len(bars)


def run_backtest(
    vt_symbol: str,
    start: datetime,
    end: datetime,
) -> dict:
    """运行回测."""
    print(f"[3/4] 运行回测...")
    print(f"    合约: {vt_symbol}")
    print(f"    时间: {start} ~ {end}")
    
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=Interval.MINUTE,
        start=start,
        end=end,
        rate=0.0001,         # 手续费率（和 GUI 一致）
        slippage=1.0,        # 滑点（和 GUI 一致）
        size=10.0,           # 合约乘数（棕榈油 10 吨/手）
        pricetick=2.0,       # 最小价格变动
        capital=100_000.0,   # 初始资金（和 GUI 一致）
    )
    
    # 添加缠论策略（启用调试日志）
    strategy_setting = {
        "debug": True,
        "debug_enabled": True,
        "debug_log_console": True,
    }
    engine.add_strategy(CtaChanPivotStrategy, strategy_setting)
    
    # 加载数据
    engine.load_data()
    print(f"    加载历史数据: {len(engine.history_data)} 条")
    
    if not engine.history_data:
        print("    ❌ 无历史数据，无法回测")
        return {}
    
    # 运行回测
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()
    
    return {
        "stats": stats,
        "trades": engine.get_all_trades(),
        "daily_results": engine.get_all_daily_results(),
        "history_data_count": len(engine.history_data),
    }


def print_result(result: dict) -> None:
    """打印回测结果."""
    print("\n[4/4] Backtest Result")
    
    if not result:
        print("    [X] Backtest failed, no result")
        return
    
    stats = result.get("stats", {})
    trades = result.get("trades", [])
    
    print("=" * 70)
    print("  P2601 ChanLun Pivot Strategy Backtest Result")
    print("=" * 70)
    
    print(f"""
[Basic Info]
    Symbol: p2601.DCE
    Interval: 1 minute
    Strategy: CtaChanPivotStrategy
    Data count: {result.get('history_data_count', 0)}

[Statistics]
    Total days: {stats.get('total_days', 0)}
    Total trades: {stats.get('total_trade_count', 0)}
    Profit days: {stats.get('profit_days', 0)}
    Loss days: {stats.get('loss_days', 0)}
    
[Returns]
    Total PnL: {stats.get('total_net_pnl', 0):,.2f} CNY
    Total return: {stats.get('total_return', 0):.2%}
    Annual return: {stats.get('annual_return', 0):.2%}
    
[Risk]
    Max drawdown: {stats.get('max_ddpercent', 0):.2%}
    Max drawdown value: {stats.get('max_drawdown', 0):,.2f} CNY
    Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}
    Calmar Ratio: {stats.get('calmar_ratio', 0):.2f}
    
[Win Rate]
    Winning trades: {stats.get('winning_trades', 0)}
    Losing trades: {stats.get('losing_trades', 0)}
    Win rate: {stats.get('win_rate', 0):.2%}
    Profit factor: {stats.get('profit_factor', 0):.2f}
""")
    
    if trades:
        print("[Trades]")
        print(f"    First trade: {trades[0].datetime}")
        print(f"    Last trade: {trades[-1].datetime}")
        print(f"\n    Recent 10 trades:")
        for t in trades[-10:]:
            print(f"      {t.datetime}: {t.direction.value} {t.offset.value} {t.volume} lots @ {t.price}")
    else:
        print("[Trades]")
        print("    No trades executed")
        print("    Possible reasons:")
        print("      1. Strategy needs warmup period")
        print("      2. Signal conditions not met")
    
    print("=" * 70)


def main():
    """主函数."""
    print("=" * 70)
    print("  P2601 缠论中枢策略回测")
    print("=" * 70)
    
    # 数据文件路径
    data_path = REPO_ROOT / "data" / "analyse" / "p2601_1min_202507-202512.csv"
    
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        sys.exit(1)
    
    # 1. 加载数据
    bars = load_csv_data(data_path)
    
    if not bars:
        print("❌ 加载数据失败")
        sys.exit(1)
    
    # 2. 保存到数据库
    save_to_database(bars)
    
    # 3. 运行回测
    start = bars[0].datetime
    end = bars[-1].datetime
    
    result = run_backtest(
        vt_symbol="p2601.DCE",
        start=start,
        end=end,
    )
    
    # 4. 打印结果
    print_result(result)


if __name__ == "__main__":
    main()
