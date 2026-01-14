"""
OpenBB 数据获取模块.

从 OpenBB 获取期货历史数据，如果 OpenBB 不支持目标品种，则降级使用 akshare。

用法:
    python -m qp.research.openbb_fetch --vt_symbol p0.DCE --days 90 --out data/openbb/p0.DCE_1d.csv

vt_symbol 规范:
    - 格式: {symbol}.{exchange}
    - 示例: p0.DCE (棕榈油连续), p2501.DCE (棕榈油2501合约)
    - exchange: DCE=大商所, SHFE=上期所, CZCE=郑商所, CFFEX=中金所
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 确保工作目录为仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(REPO_ROOT)

from qp.common import SYMBOL_MAP, parse_vt_symbol
from qp.common.constants import CHINA_FUTURES_EXCHANGES
from qp.common.logging import setup_logging, get_logger

logger = get_logger(__name__)


def try_fetch_openbb(
    symbol: str,
    exchange: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame | None:
    """
    尝试使用 OpenBB 获取期货数据.

    Returns:
        DataFrame 或 None（如果 OpenBB 不支持）
    """
    try:
        from openbb import obb

        # 检查 futures historical 的可用 provider
        logger.info("检查 OpenBB futures historical provider 覆盖...")

        # OpenBB 期货数据主要通过 yfinance，格式如 "PA=F" (钯金)
        # 中国期货不在 yfinance 覆盖范围
        # 尝试查询，预期会失败或返回空数据

        # 对于中国期货，OpenBB 暂不支持，直接返回 None
        if exchange in CHINA_FUTURES_EXCHANGES:
            logger.warning(
                "OpenBB 暂不支持中国期货交易所 %s，将使用 akshare 降级获取",
                exchange
            )
            return None

        # 其他交易所尝试查询
        result = obb.futures.historical(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if result is None or result.results is None:
            logger.warning("OpenBB 返回空数据，将降级使用 akshare")
            return None

        df = result.to_df()
        if df.empty:
            logger.warning("OpenBB 返回空 DataFrame，将降级使用 akshare")
            return None

        logger.info("OpenBB 成功获取 %d 条数据", len(df))
        return df

    except Exception as e:
        logger.warning("OpenBB 获取数据失败: %s，将降级使用 akshare", e)
        return None


def fetch_akshare(
    symbol: str,
    exchange: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    使用 akshare 获取中国期货数据.

    Args:
        symbol: 品种代码，如 "p0"
        exchange: 交易所代码，如 "DCE"
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        标准化的 DataFrame
    """
    try:
        import akshare as ak
    except ImportError as e:
        raise ImportError(
            "akshare 未安装，请运行: uv sync --extra research"
        ) from e

    logger.info("使用 akshare 获取数据: symbol=%s, exchange=%s", symbol, exchange)

    # 映射 symbol 到 akshare 格式
    ak_symbol = SYMBOL_MAP.get(symbol.lower(), symbol.upper())

    # 如果是具体合约（如 p2501），使用合约代码
    if symbol[0].isalpha() and len(symbol) > 1 and symbol[1:].isdigit():
        ak_symbol = symbol.upper()  # p2501 -> P2501

    try:
        # 获取期货日线数据
        # akshare 的期货历史数据接口
        df = ak.futures_zh_daily_sina(symbol=ak_symbol)

        if df is None or df.empty:
            raise ValueError(f"akshare 返回空数据: symbol={ak_symbol}")

        logger.info("akshare 原始数据 %d 条", len(df))

        # 标准化列名
        # akshare 返回的列通常是: date, open, high, low, close, volume, hold (持仓量)
        df = df.rename(columns={
            "date": "datetime",
            "hold": "open_interest",
        })

        # 确保必需的列存在
        required_cols = ["datetime", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"akshare 数据缺少必需列: {col}")

        # 添加 open_interest 如果不存在
        if "open_interest" not in df.columns:
            df["open_interest"] = 0

        # 转换日期
        df["datetime"] = pd.to_datetime(df["datetime"])

        # 过滤日期范围
        df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

        # 按日期排序
        df = df.sort_values("datetime").reset_index(drop=True)

        logger.info("akshare 过滤后 %d 条数据 (%s ~ %s)",
                    len(df),
                    df["datetime"].min().strftime("%Y-%m-%d") if not df.empty else "N/A",
                    df["datetime"].max().strftime("%Y-%m-%d") if not df.empty else "N/A")

        return df

    except Exception as e:
        logger.error("akshare 获取数据失败: %s", e)
        raise


def fetch_futures_data(
    vt_symbol: str,
    days: int = 90,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """
    获取期货历史数据（OpenBB 优先，akshare 降级）.

    Args:
        vt_symbol: 合约代码，如 "p0.DCE"
        days: 获取天数
        end_date: 结束日期，默认今天

    Returns:
        标准化的 DataFrame，包含列:
        datetime, open, high, low, close, volume, open_interest, symbol, exchange, interval
    """
    symbol, exchange = parse_vt_symbol(vt_symbol, return_exchange_enum=False)

    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info(
        "获取数据: vt_symbol=%s, 日期范围=%s ~ %s",
        vt_symbol,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    # 尝试 OpenBB
    df = try_fetch_openbb(symbol, exchange, start_date, end_date)

    # 降级到 akshare
    if df is None:
        df = fetch_akshare(symbol, exchange, start_date, end_date)

    # 标准化输出格式
    df = df[["datetime", "open", "high", "low", "close", "volume", "open_interest"]].copy()
    df["symbol"] = symbol
    df["exchange"] = exchange
    df["interval"] = "DAILY"

    # 格式化 datetime 为日期字符串
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")

    # 去重
    df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    return df


def main() -> None:
    """CLI 入口."""
    parser = argparse.ArgumentParser(
        prog="qp.research.openbb_fetch",
        description="获取期货历史数据（OpenBB 优先，akshare 降级）",
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
        default=90,
        help="获取天数 (默认: 90)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 CSV 文件路径 (默认: data/openbb/{vt_symbol}_1d.csv)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(verbose=args.verbose)

    # 确定输出路径
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = REPO_ROOT / "data" / "openbb" / f"{args.vt_symbol}_1d.csv"

    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 获取数据
        df = fetch_futures_data(args.vt_symbol, args.days)

        if df.empty:
            logger.error("未获取到任何数据")
            sys.exit(1)

        # 保存 CSV
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info("数据已保存: %s (%d 条)", out_path, len(df))

        # 打印摘要
        print(f"\n=== 数据摘要 ===")
        print(f"合约: {args.vt_symbol}")
        print(f"记录数: {len(df)}")
        print(f"日期范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
        print(f"输出文件: {out_path}")

    except Exception as e:
        logger.exception("获取数据失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
