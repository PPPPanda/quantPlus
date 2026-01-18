"""
使用迅投研 Tick 数据进行 CTA 策略回测.

完整数据链路:
    Tick 数据 -> 分钟 K 线 -> 时段 K 线 (6根/天) -> 回测

用法:
    python -m qp.backtest.run_tick_backtest --days 365
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest, BarData, TickData
from vnpy.trader.database import get_database
from vnpy_xt import Datafeed as XtDatafeed
from vnpy_ctastrategy.backtesting import BacktestingEngine

from qp.common.logging import setup_logging, get_logger
from qp.datafeed.bar_generator import BarSynthesizer
from qp.datafeed.session_synthesizer import SessionBarSynthesizer, DCE_PALM_OIL_SESSIONS
from qp.strategies.cta_turtle_enhanced import CtaTurtleEnhancedStrategy

logger = get_logger(__name__)


def fetch_tick_data(
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[TickData]:
    """从迅投研获取 Tick 数据."""
    logger.info("初始化迅投研数据源...")

    datafeed = XtDatafeed()
    if not datafeed.init(output=lambda msg: logger.info(msg)):
        logger.error("迅投研初始化失败")
        return []

    req = HistoryRequest(
        symbol=symbol,
        exchange=Exchange.DCE,
        start=start,
        end=end,
        interval=Interval.TICK,
    )

    logger.info("获取 Tick 数据: %s, %s - %s", symbol, start.date(), end.date())
    ticks = datafeed.query_tick_history(req, output=lambda msg: logger.info(msg))

    if ticks:
        logger.info("获取到 %d 条 Tick 数据", len(ticks))
    else:
        logger.warning("未获取到 Tick 数据")

    return ticks or []


def synthesize_minute_bars(ticks: list[TickData]) -> list[BarData]:
    """从 Tick 合成分钟 K 线."""
    if not ticks:
        return []

    minute_bars = BarSynthesizer.ticks_to_bars(ticks, gateway_name="XT_TICK")
    logger.info("从 Tick 合成分钟 K 线: %d 根", len(minute_bars))

    return minute_bars


def synthesize_session_bars(minute_bars: list[BarData]) -> list[BarData]:
    """从分钟 K 线合成时段 K 线."""
    if not minute_bars:
        return []

    synthesizer = SessionBarSynthesizer(sessions=DCE_PALM_OIL_SESSIONS)
    session_bars = synthesizer.synthesize_from_minutes(minute_bars, gateway_name="XT_SESSION")

    logger.info("从分钟合成时段 K 线: %d 根 (6根/天)", len(session_bars))

    return session_bars


def save_to_database(bars: list[BarData], symbol: str) -> int:
    """保存 K 线数据到 vnpy 数据库."""
    if not bars:
        return 0

    database = get_database()

    # 修改 symbol 以区分数据源
    for bar in bars:
        bar.symbol = symbol

    database.save_bar_data(bars)
    logger.info("保存 %d 根 K 线到数据库: %s.DCE", len(bars), symbol)

    return len(bars)


def run_backtest(
    vt_symbol: str,
    start: datetime,
    end: datetime,
    strategy_class: type,
    interval: Interval = Interval.HOUR,
) -> dict:
    """运行回测."""
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=interval,
        start=start,
        end=end,
        rate=0.0001,
        slippage=2.0,
        size=10.0,
        pricetick=2.0,
        capital=1_000_000.0,
    )

    engine.add_strategy(strategy_class, {})
    engine.load_data()

    logger.info("加载数据: %d 条", len(engine.history_data))

    if not engine.history_data:
        logger.error("无历史数据，无法回测")
        return {}

    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()

    return {
        "stats": stats,
        "trades": engine.get_all_trades(),
        "history_data_count": len(engine.history_data),
    }


def print_result(result: dict, vt_symbol: str, tick_count: int, minute_count: int, session_count: int) -> None:
    """打印回测结果."""
    if not result:
        print("\n回测失败，无结果")
        return

    stats = result.get("stats", {})
    trades = result.get("trades", [])

    print("\n" + "=" * 70)
    print("  迅投研 Tick 数据回测结果 (完整合成链路)")
    print("=" * 70)
    print(f"""
数据链路:
  Tick 数据:    {tick_count:>10,} 条
  分钟 K 线:    {minute_count:>10,} 根  (Tick -> 1min)
  时段 K 线:    {session_count:>10,} 根  (1min -> Session, 6根/天)

合约: {vt_symbol}
周期: HOUR (时段 K 线, 6 根/天)
策略: CtaTurtleEnhancedStrategy

=== 统计指标 ===
总交易日: {stats.get('total_days', 0)}
总成交笔数: {stats.get('total_trade_count', 0)}
盈利交易日: {stats.get('profit_days', 0)}
亏损交易日: {stats.get('loss_days', 0)}
总盈亏: {stats.get('total_net_pnl', 0):,.2f}
总收益率: {stats.get('total_return', 0):.2f}%
年化收益: {stats.get('annual_return', 0):.2f}%
最大回撤: {stats.get('max_ddpercent', 0):.2f}%
Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}
""")

    if trades:
        print("=== 成交记录 ===")
        print(f"首笔成交: {trades[0].datetime.date()}")
        print(f"末笔成交: {trades[-1].datetime.date()}")
        print(f"\n最近 10 笔成交:")
        for t in trades[-10:]:
            print(f"  {t.datetime}: {t.direction.value} {t.offset.value} {t.volume}@{t.price}")
    else:
        print("=== 无成交记录 ===")

    print("=" * 70)


def main() -> None:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        description="使用迅投研 Tick 数据进行 CTA 策略回测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol", "-s",
        default="p00",
        help="合约代码 (p00 = 棕榈油主力连续)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="回测天数"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出"
    )
    parser.add_argument(
        "--db-symbol",
        default="p_tick",
        help="数据库中的合约代码 (用于区分数据源)"
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    # 计算时间范围
    end = datetime.now()
    start = end - timedelta(days=args.days + 30)  # 额外 30 天用于预热

    logger.info("=" * 60)
    logger.info("迅投研 Tick 数据回测流程开始")
    logger.info("数据链路: Tick -> 分钟 -> 时段 K 线")
    logger.info("=" * 60)

    # 1. 获取 Tick 数据
    logger.info("[1/5] 获取迅投研 Tick 数据...")
    ticks = fetch_tick_data(args.symbol, start, end)

    if not ticks:
        logger.error("获取 Tick 数据失败，终止执行")
        sys.exit(1)

    tick_count = len(ticks)

    # 2. 合成分钟 K 线
    logger.info("[2/5] 从 Tick 合成分钟 K 线...")
    minute_bars = synthesize_minute_bars(ticks)

    if not minute_bars:
        logger.error("合成分钟 K 线失败，终止执行")
        sys.exit(1)

    minute_count = len(minute_bars)

    # 3. 合成时段 K 线
    logger.info("[3/5] 从分钟合成时段 K 线...")
    session_bars = synthesize_session_bars(minute_bars)

    if not session_bars:
        logger.error("合成时段 K 线失败，终止执行")
        sys.exit(1)

    session_count = len(session_bars)

    # 4. 保存到数据库
    logger.info("[4/5] 保存到 vnpy 数据库...")
    save_to_database(session_bars, args.db_symbol)

    # 5. 运行回测
    logger.info("[5/5] 运行回测...")
    vt_symbol = f"{args.db_symbol}.DCE"

    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start,
        end=end,
        strategy_class=CtaTurtleEnhancedStrategy,
        interval=Interval.HOUR,
    )

    # 打印结果
    print_result(result, vt_symbol, tick_count, minute_count, session_count)

    logger.info("回测流程完成")


if __name__ == "__main__":
    main()
