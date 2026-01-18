"""
OpenCTP TTS 网关连接测试脚本.

测试流程：
1. 创建主引擎和事件引擎
2. 添加 TTS 网关
3. 连接 OpenCTP 服务器
4. 验证连接状态
5. 查询合约信息
6. 订阅行情数据
7. 断开连接

用法：
    python tests/test_openctp_connection.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from vnpy.event import EventEngine, Event
from vnpy.trader.engine import MainEngine
from vnpy.trader.event import (
    EVENT_LOG,
    EVENT_CONTRACT,
    EVENT_TICK,
)
from vnpy.trader.constant import Exchange


def load_tts_config() -> dict:
    """加载 TTS 配置文件."""
    config_path = Path(".vntrader/connect_tts.json")
    if not config_path.exists():
        print(f"[错误] 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"[OK] 加载配置: 用户名={config['用户名']}, 交易服务器={config['交易服务器']}")
    return config


def on_log(event: Event) -> None:
    """日志事件处理."""
    log = event.data
    print(f"[日志] {log.msg}")


def on_contract(event: Event) -> None:
    """合约信息事件处理."""
    contract = event.data
    print(f"[合约] {contract.symbol}.{contract.exchange.value} - {contract.name}")


def on_tick(event: Event) -> None:
    """行情数据事件处理."""
    tick = event.data
    print(
        f"[行情] {tick.symbol}.{tick.exchange.value} "
        f"最新价={tick.last_price}, 买一={tick.bid_price_1}, 卖一={tick.ask_price_1}"
    )


def main() -> None:
    """主测试函数."""
    print("=" * 60)
    print("OpenCTP TTS 网关连接测试")
    print("=" * 60)

    # 1. 加载配置
    config = load_tts_config()

    # 2. 创建引擎
    print("\n[步骤 1] 创建事件引擎和主引擎...")
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)

    # 注册事件监听
    event_engine.register(EVENT_LOG, on_log)
    event_engine.register(EVENT_CONTRACT, on_contract)
    event_engine.register(EVENT_TICK, on_tick)

    print("[OK] 引擎创建成功")

    # 3. 添加 TTS 网关
    print("\n[步骤 2] 添加 TTS 网关...")
    try:
        from vnpy_tts import TtsGateway

        main_engine.add_gateway(TtsGateway)
        print("[OK] TTS 网关添加成功")
    except ImportError as e:
        print(f"[错误] 无法导入 TtsGateway: {e}")
        print("请安装: pip install vnpy_tts")
        sys.exit(1)

    # 4. 连接到 OpenCTP
    print("\n[步骤 3] 连接到 OpenCTP TTS 服务器...")
    print(f"  交易前置: {config['交易服务器']}")
    print(f"  行情前置: {config['行情服务器']}")

    gateway_name = "TTS"
    main_engine.connect(config, gateway_name)

    # 等待连接建立
    print("\n[步骤 4] 等待连接建立（最多 30 秒）...")
    max_wait = 30
    for i in range(max_wait):
        time.sleep(1)
        # 检查连接状态
        if main_engine.get_all_contracts():
            print(f"\n[OK] 连接成功！已接收合约信息（耗时 {i+1} 秒）")
            break
        print(f"  等待中... {i+1}/{max_wait} 秒", end="\r")
    else:
        print("\n[错误] 连接超时，未收到合约信息")
        print("\n请检查：")
        print("  1. 配置文件中的用户名、密码是否正确")
        print("  2. 服务器地址是否正确")
        print("  3. 网络连接是否正常")
        print("  4. 是否同时运行了其他使用 CTP DLL 的程序")
        main_engine.close()
        sys.exit(1)

    # 5. 查询合约信息
    print("\n[步骤 5] 查询合约信息...")
    contracts = main_engine.get_all_contracts()
    print(f"[OK] 共获取 {len(contracts)} 个合约")

    # 显示部分合约
    print("\n前 10 个合约示例：")
    for i, contract in enumerate(list(contracts)[:10]):
        print(
            f"  {i+1}. {contract.symbol}.{contract.exchange.value} - "
            f"{contract.name} (大小={contract.size})"
        )

    # 6. 订阅行情
    print("\n[步骤 6] 订阅行情数据...")

    # 查找一个活跃合约进行订阅
    test_symbols = ["p2505", "rb2505", "i2505", "IF2501", "ag2506"]
    subscribed = False

    for symbol in test_symbols:
        for contract in contracts:
            if contract.symbol == symbol:
                print(f"\n尝试订阅: {contract.symbol}.{contract.exchange.value}")
                from vnpy.trader.object import SubscribeRequest

                req = SubscribeRequest(
                    symbol=contract.symbol, exchange=contract.exchange
                )
                main_engine.subscribe(req, gateway_name)
                subscribed = True
                break
        if subscribed:
            break

    if subscribed:
        print("\n等待行情数据（5 秒）...")
        time.sleep(5)
    else:
        print("\n[警告] 未找到测试合约，跳过行情订阅")

    # 7. 断开连接
    print("\n[步骤 7] 断开连接...")
    main_engine.close()
    print("[OK] 测试完成")

    print("\n" + "=" * 60)
    print("测试结果：")
    print(f"  - 连接状态: 成功")
    print(f"  - 合约数量: {len(contracts)}")
    print(f"  - 行情订阅: {'成功' if subscribed else '跳过'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
