"""
使用迅投研数据进行 CTA 策略回测.

从迅投研获取分钟数据，使用 SessionBarSynthesizer 合成时段 K 线，
然后运行 CtaTurtleEnhancedStrategy 回测。

用法:
    python -m qp.backtest.run_xtquant_backtest --days 365
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest, BarData
from vnpy.trader.database import get_database
from vnpy_xt import Datafeed as XtDatafeed
from vnpy_ctastrategy.backtesting import BacktestingEngine

from qp.common.logging import setup_logging, get_logger
from qp.datafeed.session_synthesizer import SessionBarSynthesizer, DCE_PALM_OIL_SESSIONS
from qp.strategies.cta_turtle_enhanced import CtaTurtleEnhancedStrategy

logger = get_logger(__name__)


def fetch_minute_data(
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[BarData]:
    """从迅投研获取分钟数据."""
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
        interval=Interval.MINUTE,
    )

    logger.info("获取分钟数据: %s, %s - %s", symbol, start.date(), end.date())
    bars = datafeed.query_bar_history(req, output=lambda msg: logger.info(msg))

    if bars:
        logger.info("获取到 %d 根分钟 K 线", len(bars))
    else:
        logger.warning("未获取到分钟数据")

    return bars or []


def synthesize_session_bars(minute_bars: list[BarData]) -> list[BarData]:
    """使用 SessionBarSynthesizer 合成时段 K 线."""
    if not minute_bars:
        return []

    synthesizer = SessionBarSynthesizer(sessions=DCE_PALM_OIL_SESSIONS)
    session_bars = synthesizer.synthesize_from_minutes(minute_bars, gateway_name="XT_SESSION")

    logger.info("合成时段 K 线: %d 根 (从 %d 根分钟线)", len(session_bars), len(minute_bars))

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


def print_result(result: dict, vt_symbol: str) -> None:
    """打印回测结果."""
    if not result:
        print("\n回测失败，无结果")
        return

    stats = result.get("stats", {})
    trades = result.get("trades", [])

    print("\n" + "=" * 60)
    print("  迅投研数据 CTA 回测结果 (SessionBarSynthesizer)")
    print("=" * 60)
    print(f"""
合约: {vt_symbol}
周期: HOUR (时段 K 线, 6 根/天)
策略: CtaTurtleEnhancedStrategy
数据量: {result.get('history_data_count', 0)} 条

=== 统计指标 ===
总交易日: {stats.get('total_days', 0)}
总成交笔数: {stats.get('total_trade_count', 0)}
盈利交易日: {stats.get('profit_days', 0)}
亏损交易日: {stats.get('loss_days', 0)}
总盈亏: {stats.get('total_net_pnl', 0):.2f}
总收益率: {stats.get('total_return', 0):.2%}
年化收益: {stats.get('annual_return', 0):.2%}
最大回撤: {stats.get('max_ddpercent', 0):.2%}
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

    print("=" * 60)


def main() -> None:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        description="使用迅投研数据进行 CTA 策略回测",
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
        default="p_session",
        help="数据库中的合约代码 (用于区分数据源)"
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    # 计算时间范围
    end = datetime.now()
    start = end - timedelta(days=args.days + 30)  # 额外 30 天用于预热

    logger.info("=" * 50)
    logger.info("迅投研数据回测流程开始")
    logger.info("=" * 50)

    # 1. 获取分钟数据
    logger.info("[1/4] 获取迅投研分钟数据...")
    minute_bars = fetch_minute_data(args.symbol, start, end)

    if not minute_bars:
        logger.error("获取分钟数据失败，终止执行")
        sys.exit(1)

    # 2. 合成时段 K 线
    logger.info("[2/4] 合成时段 K 线...")
    session_bars = synthesize_session_bars(minute_bars)

    if not session_bars:
        logger.error("合成时段 K 线失败，终止执行")
        sys.exit(1)

    # 3. 保存到数据库
    logger.info("[3/4] 保存到 vnpy 数据库...")
    save_to_database(session_bars, args.db_symbol)

    # 4. 运行回测
    logger.info("[4/4] 运行回测...")
    vt_symbol = f"{args.db_symbol}.DCE"

    result = run_backtest(
        vt_symbol=vt_symbol,
        start=start,
        end=end,
        strategy_class=CtaTurtleEnhancedStrategy,
        interval=Interval.HOUR,
    )

    # 打印结果
    print_result(result, vt_symbol)

    logger.info("回测流程完成")


if __name__ == "__main__":
    main()
