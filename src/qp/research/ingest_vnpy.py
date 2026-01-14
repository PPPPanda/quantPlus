"""
vn.py 数据库入库模块.

将 CSV 数据写入 vn.py 的 SQLite 数据库，供 GUI 回测使用。

用法:
    python -m qp.research.ingest_vnpy --csv data/openbb/p0.DCE_1d.csv --vt_symbol p0.DCE --interval DAILY

注意:
    - 日线数据的 datetime 按交易日 00:00:00 记录
    - 必须在仓库根目录运行，以确保使用项目内的 .vntrader 数据库
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 确保工作目录为仓库根目录（在导入 vnpy 之前）
REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(REPO_ROOT)

# 现在导入 vnpy（它会检测 cwd 下的 .vntrader）
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData

from qp.common import EXCHANGE_MAP, INTERVAL_MAP, parse_vt_symbol
from qp.common.logging import setup_logging, get_logger

logger = get_logger(__name__)


def csv_to_bars(
    csv_path: Path,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
) -> list[BarData]:
    """
    将 CSV 文件转换为 BarData 列表.

    Args:
        csv_path: CSV 文件路径
        symbol: 合约代码
        exchange: 交易所
        interval: 数据周期

    Returns:
        BarData 列表
    """
    logger.info("读取 CSV: %s", csv_path)

    df = pd.read_csv(csv_path)
    logger.info("CSV 原始数据 %d 条", len(df))

    # 验证必需列
    required_cols = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必需列: {missing}")

    bars: list[BarData] = []

    for _, row in df.iterrows():
        # 解析日期时间
        dt_str = str(row["datetime"])
        try:
            # 尝试多种日期格式
            if len(dt_str) == 10:  # YYYY-MM-DD
                dt = datetime.strptime(dt_str, "%Y-%m-%d")
            elif "T" in dt_str:  # ISO format
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                dt = dt.replace(tzinfo=None)
            else:
                dt = pd.to_datetime(dt_str).to_pydatetime()
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
        except Exception as e:
            logger.warning("跳过无效日期行: %s, 错误: %s", dt_str, e)
            continue

        # 创建 BarData
        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=interval,
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            volume=float(row["volume"]),
            open_interest=float(row.get("open_interest", 0)),
            turnover=float(row.get("turnover", 0)),
            gateway_name="DB",
        )
        bars.append(bar)

    # 按时间排序
    bars.sort(key=lambda x: x.datetime)

    logger.info("转换完成: %d 条 BarData", len(bars))
    return bars


def ingest_bars(bars: list[BarData]) -> int:
    """
    将 BarData 写入数据库.

    Args:
        bars: BarData 列表

    Returns:
        写入的记录数
    """
    if not bars:
        logger.warning("没有数据需要写入")
        return 0

    db = get_database()
    logger.info("获取数据库实例: %s", type(db).__name__)

    # 写入数据
    success = db.save_bar_data(bars)

    if not success:
        raise RuntimeError("数据库写入失败")

    logger.info("成功写入 %d 条数据", len(bars))
    return len(bars)


def verify_ingestion(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    sample_size: int = 5,
) -> list[BarData]:
    """
    验证数据是否成功入库.

    Args:
        symbol: 合约代码
        exchange: 交易所
        interval: 数据周期
        sample_size: 抽样数量

    Returns:
        最近的 sample_size 条数据
    """
    db = get_database()

    # 加载最近的数据
    end = datetime.now()
    start = datetime(2000, 1, 1)  # 很早的日期，确保能获取所有数据

    bars = db.load_bar_data(symbol, exchange, interval, start, end)

    if not bars:
        raise RuntimeError(
            f"验证失败: 数据库中未找到 {symbol}.{exchange.value} 的数据"
        )

    logger.info("验证成功: 数据库中有 %d 条记录", len(bars))

    # 返回最近的记录
    return bars[-sample_size:] if len(bars) >= sample_size else bars


def main() -> None:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        prog="qp.research.ingest_vnpy",
        description="将 CSV 数据入库到 vn.py SQLite 数据库",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="输入 CSV 文件路径",
    )
    parser.add_argument(
        "--vt_symbol",
        type=str,
        required=True,
        help="合约代码，格式: symbol.exchange (如 p0.DCE)",
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

    args = parser.parse_args()

    # 配置日志
    setup_logging(verbose=args.verbose)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV 文件不存在: %s", csv_path)
        sys.exit(1)

    try:
        # 解析参数
        symbol, exchange = parse_vt_symbol(args.vt_symbol)
        interval = INTERVAL_MAP[args.interval]

        logger.info("入库参数: symbol=%s, exchange=%s, interval=%s",
                    symbol, exchange.value, interval.value)

        # CSV -> BarData
        bars = csv_to_bars(csv_path, symbol, exchange, interval)

        if not bars:
            logger.error("没有有效的数据可入库")
            sys.exit(1)

        # 入库
        count = ingest_bars(bars)

        # 验证
        logger.info("验证入库结果...")
        recent_bars = verify_ingestion(symbol, exchange, interval)

        # 打印摘要
        print(f"\n=== 入库完成 ===")
        print(f"合约: {args.vt_symbol}")
        print(f"周期: {interval.value}")
        print(f"写入: {count} 条")
        print(f"日期范围: {bars[0].datetime.strftime('%Y-%m-%d')} ~ "
              f"{bars[-1].datetime.strftime('%Y-%m-%d')}")
        print(f"\n最近 {len(recent_bars)} 条数据:")
        for bar in recent_bars:
            print(f"  {bar.datetime.strftime('%Y-%m-%d')}: "
                  f"O={bar.open_price:.2f} H={bar.high_price:.2f} "
                  f"L={bar.low_price:.2f} C={bar.close_price:.2f} "
                  f"V={bar.volume:.0f}")

    except Exception as e:
        logger.exception("入库失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
