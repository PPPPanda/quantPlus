"""
GUI 启动器模块.

提供 start_trader_gui 函数，负责创建 Qt 应用、事件引擎、主引擎和主窗口。
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp

from qp.ui.profiles import get_apps_for_profile, VALID_PROFILES

if TYPE_CHECKING:
    from vnpy.trader.gateway import BaseGateway

logger = logging.getLogger(__name__)


def start_trader_gui(
    gateway_cls: type[BaseGateway],
    profile: str = "all",
    title: str = "QuantPlus Trader",
) -> None:
    """
    启动 Trader GUI.

    Args:
        gateway_cls: Gateway 类（如 CtpGateway）
        profile: 启动配置，必须是 "trade", "research" 或 "all"
        title: 窗口标题（目前仅用于日志，MainWindow 有自己的标题逻辑）

    Raises:
        ValueError: 如果 profile 名称无效
    """
    # 验证 profile
    if profile not in VALID_PROFILES:
        valid_str = ", ".join(sorted(VALID_PROFILES))
        raise ValueError(
            f"无效的 profile: '{profile}'。有效选项: {valid_str}"
        )

    logger.info("启动 %s，profile=%s", title, profile)

    # 创建 Qt 应用
    qapp = create_qapp()
    logger.debug("Qt 应用已创建")

    # 创建事件引擎
    event_engine = EventEngine()
    logger.debug("事件引擎已创建")

    # 创建主引擎
    main_engine = MainEngine(event_engine)
    logger.debug("主引擎已创建")

    # 添加 Gateway
    main_engine.add_gateway(gateway_cls)
    logger.info("已加载 Gateway: %s", gateway_cls.__name__)

    # 根据 profile 加载 Apps
    app_classes = get_apps_for_profile(profile)
    for app_cls in app_classes:
        main_engine.add_app(app_cls)
        logger.info("已加载 App: %s", app_cls.__name__)

    if not app_classes:
        logger.warning("profile '%s' 没有加载任何 App，请检查依赖安装", profile)

    # 创建并显示主窗口
    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()
    logger.info("主窗口已显示")

    # 进入事件循环
    sys.exit(qapp.exec())
