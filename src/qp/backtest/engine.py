"""
CTA 策略回测引擎.

提供纯函数式的回测接口，可被 CLI、GUI 或其他模块调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from vnpy.trader.constant import Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回测结果数据类."""

    stats: dict[str, Any]
    """统计指标字典"""

    trades: list[Any]
    """成交记录列表"""

    daily_results: list[Any]
    """每日结果列表"""

    history_data_count: int
    """历史数据条数"""


def run_backtest(
    vt_symbol: str,
    start: datetime,
    end: datetime,
    strategy_class: type,
    strategy_setting: dict[str, Any] | None = None,
    interval: Interval = Interval.DAILY,
    rate: float = 0.0001,
    slippage: float = 2.0,
    size: float = 10.0,
    pricetick: float = 2.0,
    capital: float = 1_000_000.0,
) -> BacktestResult:
    """
    运行 CTA 策略回测.

    Args:
        vt_symbol: 合约代码，如 "p0.DCE"
        start: 回测开始日期
        end: 回测结束日期
        strategy_class: 策略类
        strategy_setting: 策略参数
        interval: 数据周期 (默认 DAILY)
        rate: 手续费率
        slippage: 滑点
        size: 合约乘数
        pricetick: 最小价格变动
        capital: 初始资金

    Returns:
        BacktestResult 对象，包含统计指标、成交记录等
    """
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=interval,
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

    return BacktestResult(
        stats=stats,
        trades=engine.get_all_trades(),
        daily_results=engine.get_all_daily_results(),
        history_data_count=len(engine.history_data),
    )


def load_strategy_class(strategy_name: str) -> type:
    """
    动态加载策略类.

    Args:
        strategy_name: 策略类名，如 "CtaPalmOilStrategy"

    Returns:
        策略类

    Raises:
        ImportError: 策略模块导入失败
        AttributeError: 策略类不存在
    """
    import importlib

    # 策略名到模块的映射
    strategy_modules: dict[str, str] = {
        "CtaPalmOilStrategy": "qp.strategies.cta_palm_oil",
        "CtaTurtleEnhancedStrategy": "qp.strategies.cta_turtle_enhanced",
    }

    module_name = strategy_modules.get(strategy_name)
    if not module_name:
        # 尝试根据策略类名推断模块名
        # CtaXxxYyyStrategy -> qp.strategies.cta_xxx_yyy
        base_name = strategy_name.replace("Strategy", "")
        parts = []
        current = ""
        for char in base_name:
            if char.isupper() and current:
                parts.append(current.lower())
                current = char
            else:
                current += char
        if current:
            parts.append(current.lower())
        module_name = "qp.strategies." + "_".join(parts)

    strategy_module = importlib.import_module(module_name)
    return getattr(strategy_module, strategy_name)
