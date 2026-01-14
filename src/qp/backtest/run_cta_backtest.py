"""
CTA 策略脚本化回测工具.

用法:
    python -m qp.backtest.run_cta_backtest --vt_symbol p0.DCE --days 90

此脚本用于命令行验证策略回测结果，不依赖 GUI。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 确保工作目录为仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "strategies"))

from vnpy_ctastrategy.backtesting import BacktestingEngine
from vnpy.trader.constant import Interval

logger = logging.getLogger(__name__)


def run_backtest(
    vt_symbol: str,
    start: datetime,
    end: datetime,
    strategy_class: type,
    strategy_setting: dict | None = None,
    rate: float = 0.0001,
    slippage: float = 2.0,
    size: float = 10.0,
    pricetick: float = 2.0,
    capital: float = 1_000_000.0,
) -> dict:
    """
    运行 CTA 策略回测.

    Args:
        vt_symbol: 合约代码，如 "p0.DCE"
        start: 回测开始日期
        end: 回测结束日期
        strategy_class: 策略类
        strategy_setting: 策略参数
        rate: 手续费率
        slippage: 滑点
        size: 合约乘数
        pricetick: 最小价格变动
        capital: 初始资金

    Returns:
        回测统计结果字典
    """
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=Interval.DAILY,
        start=start,
        end=end,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
    )

    engine.add_strategy(strategy_class, strategy_setting or {})
    engine.load_data()

    logger.info("加载数据: %d 条", len(engine.history_data))

    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()

    return {
        "stats": stats,
        "trades": engine.get_all_trades(),
        "daily_results": engine.get_all_daily_results(),
        "history_data_count": len(engine.history_data),
    }


def main() -> None:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        prog="qp.backtest.run_cta_backtest",
        description="CTA 策略脚本化回测",
    )
    parser.add_argument(
        "--vt_symbol",
        type=str,
        default="p0.DCE",
        help="合约代码 (默认: p0.DCE)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="回测天数 (默认: 365)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="CtaPalmOilStrategy",
        help="策略类名 (默认: CtaPalmOilStrategy)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 动态导入策略
    try:
        strategy_module = __import__("cta_palm_oil")
        strategy_class = getattr(strategy_module, args.strategy)
    except (ImportError, AttributeError) as e:
        logger.error("无法加载策略 %s: %s", args.strategy, e)
        sys.exit(1)

    # 计算日期范围
    end = datetime.now()
    start = end - timedelta(days=args.days + 30)  # 额外 30 天用于预热

    logger.info("回测参数: vt_symbol=%s, start=%s, end=%s", args.vt_symbol, start.date(), end.date())

    try:
        result = run_backtest(
            vt_symbol=args.vt_symbol,
            start=start,
            end=end,
            strategy_class=strategy_class,
        )
    except Exception as e:
        logger.exception("回测失败: %s", e)
        sys.exit(1)

    stats = result["stats"]
    trades = result["trades"]

    print("\n" + "=" * 60)
    print("  CTA 回测结果")
    print("=" * 60)
    print(f"""
合约: {args.vt_symbol}
策略: {args.strategy}
数据量: {result['history_data_count']} 条

=== 统计指标 ===
总交易日: {stats.get('total_days', 0)}
总成交笔数: {stats.get('total_trade_count', 0)}
盈利交易日: {stats.get('profit_days', 0)}
亏损交易日: {stats.get('loss_days', 0)}
总盈亏: {stats.get('total_net_pnl', 0):.2f}
总收益率: {stats.get('total_return', 0):.2%}
最大回撤: {stats.get('max_ddpercent', 0):.2%}
Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}
""")

    if trades:
        print("=== 成交记录 ===")
        print(f"首笔成交: {trades[0].datetime.date()}")
        print(f"末笔成交: {trades[-1].datetime.date()}")
        print(f"\n最近 10 笔成交:")
        for t in trades[-10:]:
            print(f"  {t.datetime.date()}: {t.direction.value} {t.offset.value} {t.volume}@{t.price}")
    else:
        print("=== 无成交记录 ===")
        print("可能原因:")
        print("  1. 数据量不足（需要 > ArrayManager.size）")
        print("  2. 策略参数不适合当前行情")
        print("  3. vt_symbol/interval 配置错误")

    print("=" * 60)


if __name__ == "__main__":
    main()
