"""CTA 回测模块."""

from qp.backtest.engine import run_backtest, load_strategy_class, BacktestResult

__all__ = [
    "run_backtest",
    "load_strategy_class",
    "BacktestResult",
]
