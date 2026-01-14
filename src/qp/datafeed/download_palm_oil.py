"""
下载棕榈油期货 Tick 数据并合成分钟 K 线.

使用示例:
    python -m qp.datafeed.download_palm_oil --symbol p2505 --days 365
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest

from qp.common.logging import setup_logging
from qp.datafeed.bar_generator import BarSynthesizer
from qp.datafeed.xtquant_feed import XTQuantDatafeed

logger = logging.getLogger(__name__)


def download_and_synthesize(
    symbol: str = "p2505",
    days: int = 365,
    output_dir: str | None = None,
    target_interval: str = "1m",
) -> None:
    """
    下载 Tick 数据并合成 K 线.

    Args:
        symbol: 合约代码，如 "p2505" (棕榈油 2505 合约)
        days: 下载天数
        output_dir: 输出目录
        target_interval: 目标 K 线周期 ("1m", "5m", "15m", "30m", "1h")
    """
    setup_logging(verbose=True)

    # 初始化数据源
    feed = XTQuantDatafeed()
    if not feed.init(output=lambda msg: logger.info(msg)):
        logger.error("数据源初始化失败")
        return

    # 计算时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(
        "开始下载棕榈油期货数据: %s, 时间范围: %s - %s",
        symbol, start_date.date(), end_date.date()
    )

    # 创建请求
    req = HistoryRequest(
        symbol=symbol,
        exchange=Exchange.DCE,
        start=start_date,
        end=end_date,
    )

    # 下载 Tick 数据
    logger.info("正在下载 Tick 数据...")
    ticks = feed.query_tick_history(req, output=lambda msg: logger.info(msg))

    if not ticks:
        logger.warning("未获取到 Tick 数据")
        return

    logger.info("获取到 %d 条 Tick 数据", len(ticks))

    # 解析目标周期
    interval, window = _parse_interval(target_interval)

    # 合成 K 线
    logger.info("正在合成 %s K 线...", target_interval)
    bars = BarSynthesizer.ticks_to_target_bars(
        ticks,
        target_interval=interval,
        window=window,
        gateway_name="XTQUANT",
    )

    logger.info("合成完成: %d 根 K 线", len(bars))

    # 保存数据
    if output_dir:
        _save_to_csv(bars, output_dir, symbol, target_interval)

    # 显示统计信息
    if bars:
        _print_stats(bars)

    feed.close()


def _parse_interval(interval_str: str) -> tuple[Interval, int]:
    """解析 K 线周期字符串."""
    interval_map = {
        "1m": (Interval.MINUTE, 1),
        "5m": (Interval.MINUTE, 5),
        "15m": (Interval.MINUTE, 15),
        "30m": (Interval.MINUTE, 30),
        "60m": (Interval.HOUR, 1),
        "1h": (Interval.HOUR, 1),
    }

    if interval_str not in interval_map:
        raise ValueError(f"不支持的周期: {interval_str}，可选: {list(interval_map.keys())}")

    return interval_map[interval_str]


def _save_to_csv(bars, output_dir: str, symbol: str, interval: str) -> None:
    """保存 K 线数据到 CSV."""
    import csv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "datetime", "open", "high", "low", "close",
            "volume", "turnover", "open_interest"
        ])

        for bar in bars:
            writer.writerow([
                bar.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                bar.open_price,
                bar.high_price,
                bar.low_price,
                bar.close_price,
                bar.volume,
                bar.turnover,
                bar.open_interest,
            ])

    logger.info("数据已保存到: %s", filename)


def _print_stats(bars) -> None:
    """打印统计信息."""
    first_bar = bars[0]
    last_bar = bars[-1]

    logger.info("=" * 50)
    logger.info("数据统计:")
    logger.info("  合约: %s.%s", first_bar.symbol, first_bar.exchange.value)
    logger.info("  K 线数量: %d", len(bars))
    logger.info("  时间范围: %s - %s", first_bar.datetime, last_bar.datetime)
    logger.info("  开盘价: %.2f", first_bar.open_price)
    logger.info("  收盘价: %.2f", last_bar.close_price)
    logger.info("  最高价: %.2f", max(b.high_price for b in bars))
    logger.info("  最低价: %.2f", min(b.low_price for b in bars))
    logger.info("  总成交量: %.0f", sum(b.volume for b in bars))
    logger.info("=" * 50)


def main() -> None:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        description="下载棕榈油期货 Tick 数据并合成 K 线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol", "-s",
        default="p2505",
        help="合约代码，如 p2505 (棕榈油 2505)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="下载天数"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录，留空则不保存"
    )
    parser.add_argument(
        "--interval", "-i",
        default="1m",
        choices=["1m", "5m", "15m", "30m", "60m", "1h"],
        help="目标 K 线周期"
    )

    args = parser.parse_args()

    download_and_synthesize(
        symbol=args.symbol,
        days=args.days,
        output_dir=args.output,
        target_interval=args.interval,
    )


if __name__ == "__main__":
    main()
