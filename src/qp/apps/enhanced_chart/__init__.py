"""
增强K线图表模块 (Enhanced Chart Wizard)

扩展 vnpy_chartwizard，增加以下功能：
- 多周期支持（1分钟、5分钟、15分钟、60分钟）
- 技术指标（MA、MACD、BOLL 等）
- 指标参数配置对话框
- 工具栏和菜单
"""

from pathlib import Path
from vnpy.trader.app import BaseApp
from vnpy_chartwizard.engine import ChartWizardEngine, APP_NAME

__all__ = ["EnhancedChartWizardApp"]

__version__ = "1.0.0"


class EnhancedChartWizardApp(BaseApp):
    """增强K线图表 App"""

    app_name: str = APP_NAME  # 复用官方的 APP_NAME 避免重复注册
    app_module: str = __module__
    app_path: Path = Path(__file__).parent
    display_name: str = "增强K线图表"
    engine_class: type[ChartWizardEngine] = ChartWizardEngine  # 复用官方 Engine
    widget_name: str = "EnhancedChartWizardWidget"
    icon_name: str = str(app_path.joinpath("ui", "enhanced_cw.ico"))
