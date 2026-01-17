"""
Trader GUI 入口模块.

薄入口：解析命令行参数，调用 launcher 启动 GUI。
用法：python -m qp.runtime.trader_app --profile {trade|research|all}
"""

from __future__ import annotations

import argparse
import sys

from qp.common.logging import setup_logging, get_logger
from qp.ui.launcher import start_trader_gui
from qp.ui.profiles import VALID_PROFILES


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        prog="qp.runtime.trader_app",
        description="QuantPlus Trader GUI 启动器",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=sorted(VALID_PROFILES),
        default="all",
        help="启动配置: trade=实盘, research=投研回测, all=全功能 (默认: all)",
    )
    parser.add_argument(
        "--gateway",
        type=str,
        choices=["ctp", "tts"],
        default="ctp",
        help="交易网关: ctp=CTP/SimNow, tts=OpenCTP 7x24 (默认: ctp)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细日志输出",
    )
    return parser.parse_args()


def main() -> None:
    """主入口函数."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger = get_logger(__name__)
    logger.info("QuantPlus Trader 启动，profile=%s, gateway=%s", args.profile, args.gateway)

    # 延迟导入 Gateway，避免在 --help 时触发重量级模块加载
    if args.gateway == "tts":
        try:
            from vnpy_tts import TtsGateway
            gateway_cls = TtsGateway
            gateway_name = "TTS (OpenCTP 7x24)"
        except ImportError as e:
            logger.error("无法导入 vnpy_tts.TtsGateway: %s", e)
            logger.error("请确保已安装 vnpy_tts 模块: pip install vnpy_tts")
            sys.exit(1)
    else:  # ctp
        try:
            from vnpy_ctp import CtpGateway
            gateway_cls = CtpGateway
            gateway_name = "CTP"
        except ImportError as e:
            logger.error("无法导入 vnpy_ctp.CtpGateway: %s", e)
            logger.error("请确保已安装 vnpy_ctp 模块")
            sys.exit(1)

    logger.info("使用网关: %s", gateway_name)

    try:
        start_trader_gui(
            gateway_cls=gateway_cls,
            profile=args.profile,
            title=f"QuantPlus Trader ({gateway_name})",
        )
    except ValueError as e:
        logger.error("启动失败: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("启动时发生未预期的错误: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
