"""
CTA 策略脚本化回测工具.

用法:
    python -m qp.backtest.run_cta_backtest --vt_symbol p0.DCE --days 90
    python -m qp.backtest.run_cta_backtest --vt_symbol p0.DCE --interval HOUR --days 180

此脚本用于命令行验证策略回测结果，不依赖 GUI。

注：此模块为向后兼容保留，实际实现已迁移到 qp.backtest.cli
"""

from qp.backtest.cli import main

if __name__ == "__main__":
    main()
