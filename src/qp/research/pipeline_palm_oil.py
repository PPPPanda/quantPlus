"""
棕榈油数据流水线.

一键完成：数据拉取 → 入库 → 输出 GUI 回测指引。

用法:
    python -m qp.research.pipeline_palm_oil --vt_symbol p0.DCE --days 90

流程:
    1. 使用 openbb_fetch 获取数据（OpenBB 优先，akshare 降级）
    2. 使用 ingest_vnpy 入库到 .vntrader/database.db
    3. 输出 GUI 回测操作指引
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 确保工作目录为仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(REPO_ROOT)

from vnpy.trader.constant import Interval

from qp.common import EXCHANGE_MAP, parse_vt_symbol
from qp.common.logging import setup_logging, get_logger
from qp.research.openbb_fetch import fetch_futures_data
from qp.research.ingest_vnpy import csv_to_bars, ingest_bars, verify_ingestion

logger = get_logger(__name__)


def run_pipeline(vt_symbol: str, days: int) -> None:
    """
    运行完整的数据流水线.

    Args:
        vt_symbol: 合约代码，如 "p0.DCE"
        days: 获取天数
    """
    symbol, exchange = parse_vt_symbol(vt_symbol, return_exchange_enum=True)

    # === Step 1: 拉取数据 ===
    logger.info("=" * 50)
    logger.info("Step 1: 拉取数据")
    logger.info("=" * 50)

    csv_path = REPO_ROOT / "data" / "openbb" / f"{vt_symbol}_1d.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_futures_data(vt_symbol, days)

    if df.empty:
        raise RuntimeError("未获取到任何数据")

    # 保存 CSV
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("数据已保存: %s (%d 条)", csv_path, len(df))

    # === Step 2: 入库 ===
    logger.info("=" * 50)
    logger.info("Step 2: 入库到 vn.py 数据库")
    logger.info("=" * 50)

    bars = csv_to_bars(csv_path, symbol, exchange, Interval.DAILY)
    count = ingest_bars(bars)

    # 验证
    recent_bars = verify_ingestion(symbol, exchange, Interval.DAILY)

    # === Step 3: 输出指引 ===
    logger.info("=" * 50)
    logger.info("Step 3: 完成")
    logger.info("=" * 50)

    print("\n" + "=" * 60)
    print("  棕榈油数据流水线完成")
    print("=" * 60)
    print(f"""
数据摘要:
  - 合约: {vt_symbol}
  - 记录数: {count} 条
  - 日期范围: {bars[0].datetime.strftime('%Y-%m-%d')} ~ {bars[-1].datetime.strftime('%Y-%m-%d')}
  - CSV 文件: {csv_path}
  - 数据库: .vntrader/database.db

最近 5 条数据:""")
    for bar in recent_bars[-5:]:
        print(f"  {bar.datetime.strftime('%Y-%m-%d')}: "
              f"O={bar.open_price:.2f} H={bar.high_price:.2f} "
              f"L={bar.low_price:.2f} C={bar.close_price:.2f} V={bar.volume:.0f}")

    print(f"""
================================================================================
下一步: 启动 GUI 进行回测
================================================================================

1. 启动 GUI (research profile):
   uv run python -m qp.runtime.trader_app --profile research

2. 在 GUI 中操作:
   a) 菜单栏: 功能 → CTA回测
   b) 在回测界面配置:
      - 本地代码: {symbol}
      - 交易所: {exchange.value}
      - K线周期: 日线 (DAILY)
      - 开始日期: {bars[0].datetime.strftime('%Y-%m-%d')}
      - 结束日期: {bars[-1].datetime.strftime('%Y-%m-%d')}
      - 策略: CtaPalmOilStrategy (双均线策略)
      - 参数: fast_window=10, slow_window=20, fixed_size=1
   c) 点击「开始回测」
   d) 查看结果: 统计指标、资金曲线、交易明细等

3. 或者使用 DataManager 查看数据:
   a) 菜单栏: 功能 → 数据管理
   b) 查看已入库的 {vt_symbol} 数据

================================================================================
""")


def main() -> None:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        prog="qp.research.pipeline_palm_oil",
        description="棕榈油数据流水线：拉取 → 入库 → GUI 回测指引",
    )
    parser.add_argument(
        "--vt_symbol",
        type=str,
        default="p0.DCE",
        help="合约代码，格式: symbol.exchange (默认: p0.DCE)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="获取天数 (默认: 365)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(verbose=args.verbose)

    try:
        run_pipeline(args.vt_symbol, args.days)
    except Exception as e:
        logger.exception("流水线执行失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
