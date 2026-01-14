"""
CTA 策略回测命令行接口.

用法:
    python -m qp.backtest.cli --vt_symbol p0.DCE --days 90
    python -m qp.backtest.cli --vt_symbol p0.DCE --interval HOUR --days 180
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from qp.common.constants import INTERVAL_MAP
from qp.common.logging import setup_logging, get_logger
from qp.backtest.engine import run_backtest, load_strategy_class

# 确保工作目录为仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "strategies"))

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        prog="qp.backtest.cli",
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
        "--interval",
        type=str,
        default="DAILY",
        choices=list(INTERVAL_MAP.keys()),
        help="数据周期 (默认: DAILY)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )
    return parser.parse_args()


def print_result(
    result,
    vt_symbol: str,
    interval: str,
    strategy_name: str,
) -> None:
    """打印回测结果."""
    stats = result.stats
    trades = result.trades

    print("\n" + "=" * 60)
    print("  CTA 回测结果")
    print("=" * 60)
    print(f"""
合约: {vt_symbol}
周期: {interval}
策略: {strategy_name}
数据量: {result.history_data_count} 条

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


def main() -> None:
    """CLI 入口."""
    args = parse_args()

    setup_logging(verbose=args.verbose)

    # 获取 Interval 枚举
    interval = INTERVAL_MAP.get(args.interval)
    if interval is None:
        logger.error("未知的数据周期: %s", args.interval)
        sys.exit(1)

    # 动态加载策略
    try:
        strategy_class = load_strategy_class(args.strategy)
    except (ImportError, AttributeError) as e:
        logger.error("无法加载策略 %s: %s", args.strategy, e)
        logger.info("可用策略: CtaPalmOilStrategy, CtaTurtleEnhancedStrategy")
        sys.exit(1)

    # 计算日期范围
    end = datetime.now()
    start = end - timedelta(days=args.days + 30)  # 额外 30 天用于预热

    logger.info(
        "回测参数: vt_symbol=%s, interval=%s, start=%s, end=%s",
        args.vt_symbol,
        args.interval,
        start.date(),
        end.date(),
    )

    try:
        result = run_backtest(
            vt_symbol=args.vt_symbol,
            start=start,
            end=end,
            strategy_class=strategy_class,
            interval=interval,
        )
    except Exception as e:
        logger.exception("回测失败: %s", e)
        sys.exit(1)

    print_result(result, args.vt_symbol, args.interval, args.strategy)


if __name__ == "__main__":
    main()
