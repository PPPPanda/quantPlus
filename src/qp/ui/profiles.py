"""
Profile 定义模块.

定义 trade / research / all 三种启动配置，返回对应的 App 类列表。
对可选模块使用延迟导入，导入失败则跳过。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vnpy.trader.app import BaseApp

logger = logging.getLogger(__name__)

# 有效的 profile 名称
VALID_PROFILES: frozenset[str] = frozenset({"trade", "research", "all"})


def _try_import_app(module_name: str, class_name: str) -> type[BaseApp] | None:
    """
    尝试导入 App 类及其 UI 模块，失败则返回 None 并记录警告.

    会同时验证 App 的 UI 模块是否可导入，避免 MainWindow 初始化时崩溃。

    Args:
        module_name: 模块名，例如 "vnpy_ctastrategy"
        class_name: 类名，例如 "CtaStrategyApp"

    Returns:
        App 类或 None
    """
    try:
        module = __import__(module_name, fromlist=[class_name])
        app_cls = getattr(module, class_name)

        # 验证 UI 模块是否可导入（MainWindow 在初始化菜单时会导入）
        app_instance = app_cls()
        ui_module_name = app_instance.app_module + ".ui"
        __import__(ui_module_name)

        logger.debug("成功加载 App: %s.%s", module_name, class_name)
        return app_cls
    except ImportError as e:
        logger.warning("模块 %s 或其 UI 依赖导入失败: %s", module_name, e)
        return None
    except AttributeError as e:
        logger.warning("模块 %s 中未找到类 %s: %s", module_name, class_name, e)
        return None
    except Exception as e:
        logger.warning("加载 %s.%s 时发生错误: %s", module_name, class_name, e)
        return None


def _try_import_enhanced_chart() -> type[BaseApp] | None:
    """
    尝试导入增强K线图表 App，失败时回退到官方版本.

    Returns:
        EnhancedChartWizardApp 或 ChartWizardApp 或 None
    """
    # 优先尝试加载增强版
    try:
        from qp.apps.enhanced_chart import EnhancedChartWizardApp
        logger.info("成功加载增强K线图表: EnhancedChartWizardApp")
        return EnhancedChartWizardApp
    except Exception as e:
        logger.warning("加载增强K线图表失败: %s，回退到官方版本", e)
        # 回退到官方版本
        return _try_import_app("vnpy_chartwizard", "ChartWizardApp")


def _get_trade_apps() -> list[type[BaseApp]]:
    """获取 trade profile 的 App 列表."""
    apps: list[type[BaseApp] | None] = []

    # 必需：CTA 策略
    apps.append(_try_import_app("vnpy_ctastrategy", "CtaStrategyApp"))

    # 必需：K线图表（使用增强版）
    apps.append(_try_import_enhanced_chart())

    # 可选：数据记录
    apps.append(_try_import_app("vnpy_datarecorder", "DataRecorderApp"))

    # 可选：风控管理
    apps.append(_try_import_app("vnpy_riskmanager", "RiskManagerApp"))

    # 过滤掉 None
    return [app for app in apps if app is not None]


def _get_research_apps() -> list[type[BaseApp]]:
    """获取 research profile 的 App 列表."""
    apps: list[type[BaseApp] | None] = []

    # 必需：CTA 回测
    apps.append(_try_import_app("vnpy_ctabacktester", "CtaBacktesterApp"))

    # 必需：数据管理
    apps.append(_try_import_app("vnpy_datamanager", "DataManagerApp"))

    # 可选：图表向导（使用增强版）
    apps.append(_try_import_enhanced_chart())

    # 过滤掉 None
    return [app for app in apps if app is not None]


def _get_all_apps() -> list[type[BaseApp]]:
    """获取 all profile 的 App 列表（trade + research + 额外模块）."""
    apps: list[type[BaseApp] | None] = []

    # === Trade 相关 ===
    apps.append(_try_import_app("vnpy_ctastrategy", "CtaStrategyApp"))
    apps.append(_try_import_app("vnpy_datarecorder", "DataRecorderApp"))
    apps.append(_try_import_app("vnpy_riskmanager", "RiskManagerApp"))

    # === Research 相关 ===
    apps.append(_try_import_app("vnpy_ctabacktester", "CtaBacktesterApp"))
    apps.append(_try_import_app("vnpy_datamanager", "DataManagerApp"))
    apps.append(_try_import_enhanced_chart())  # 使用增强版

    # === 额外可选 ===
    apps.append(_try_import_app("vnpy_paperaccount", "PaperAccountApp"))

    # 过滤掉 None
    return [app for app in apps if app is not None]


def get_apps_for_profile(profile: str) -> list[type[BaseApp]]:
    """
    根据 profile 名称返回对应的 App 类列表.

    Args:
        profile: profile 名称，必须是 "trade", "research" 或 "all"

    Returns:
        App 类列表

    Raises:
        ValueError: 如果 profile 名称无效
    """
    if profile not in VALID_PROFILES:
        valid_str = ", ".join(sorted(VALID_PROFILES))
        raise ValueError(
            f"无效的 profile: '{profile}'。有效选项: {valid_str}"
        )

    if profile == "trade":
        return _get_trade_apps()
    elif profile == "research":
        return _get_research_apps()
    else:  # profile == "all"
        return _get_all_apps()
