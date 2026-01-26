"""
测试ChanDebugger功能.

运行方式:
    python scripts/test_chan_debugger.py
"""
import sys
from pathlib import Path

# 直接导入chan_debugger模块
chan_debugger_path = Path(__file__).parent.parent / "src" / "qp" / "utils"
sys.path.insert(0, str(chan_debugger_path))

from datetime import datetime
from chan_debugger import ChanDebugger


def test_debugger():
    """测试ChanDebugger基本功能."""
    print("=" * 60)
    print("测试 ChanDebugger")
    print("=" * 60)

    # 创建debugger
    debugger = ChanDebugger(
        strategy_name="TestStrategy",
        base_dir="data/debug",
        enabled=True,
        log_level="DEBUG",
        log_console=True
    )

    # 保存配置
    debugger.save_config({
        "strategy_name": "TestStrategy",
        "vt_symbol": "SA2601.CZCE",
        "atr_trailing_mult": 3.0,
        "min_bi_gap": 4,
    })

    # 模拟1分钟K线
    for i in range(10):
        bar = {
            'datetime': datetime(2025, 1, 24, 9, 30 + i),
            'open': 1200 + i,
            'high': 1205 + i,
            'low': 1198 + i,
            'close': 1202 + i,
            'volume': 1000 + i * 100
        }
        debugger.log_kline_1m(bar)

    # 模拟5分钟K线
    for i in range(3):
        bar = {
            'datetime': datetime(2025, 1, 24, 9, 35 + i * 5),
            'open': 1200 + i * 5,
            'high': 1210 + i * 5,
            'low': 1195 + i * 5,
            'close': 1208 + i * 5,
            'volume': 5000 + i * 500
        }
        debugger.log_kline_5m(
            bar,
            diff_5m=0.5 + i * 0.1,
            dea_5m=0.3 + i * 0.05,
            atr=8.5 + i * 0.5,
            diff_15m=1.2 + i * 0.1,
            dea_15m=1.0 + i * 0.05
        )

    # 模拟笔
    bi1 = {'type': 'bottom', 'price': 1195, 'idx': 5, 'data': {'high': 1200, 'low': 1195}}
    debugger.log_bi(bi1, bi_idx=1, k_lines_count=10)

    bi2 = {'type': 'top', 'price': 1220, 'idx': 12, 'data': {'high': 1220, 'low': 1215}}
    debugger.log_bi(bi2, bi_idx=2, k_lines_count=15)

    bi3 = {'type': 'bottom', 'price': 1205, 'idx': 18, 'data': {'high': 1210, 'low': 1205}}
    debugger.log_bi(bi3, bi_idx=3, k_lines_count=20)

    # 模拟中枢
    pivot = {
        'zg': 1215,
        'zd': 1200,
        'start_bi_idx': 1,
        'end_bi_idx': 3
    }
    debugger.log_pivot(pivot, pivot_idx=1, status="confirmed")

    # 模拟缠论状态
    debugger.log_chan_state(
        k_lines_count=20,
        bi_count=3,
        pivot_count=1,
        bi_points=[bi1, bi2, bi3],
        pivots=[pivot]
    )

    # 模拟信号
    debugger.log_signal(
        signal_type="3B",
        direction="Buy",
        trigger_price=1210,
        stop_price=1200,
        atr=8.5,
        reason="3B买点: 回踩不破中枢ZG=1215"
    )

    # 模拟开仓
    debugger.log_trade(
        action="OPEN_LONG",
        price=1210,
        volume=1,
        position=1,
        pnl=0,
        signal_type="3B"
    )

    # 模拟持仓状态
    debugger.log_position_status(
        position=1,
        entry_price=1210,
        stop_price=1200,
        current_price=1218,
        trailing_active=False
    )

    # 模拟平仓
    debugger.log_trade(
        action="CLOSE_LONG",
        price=1225,
        volume=1,
        position=0,
        pnl=15,
        signal_type="3B"
    )

    # 保存摘要
    debugger.close()

    print("\n" + "=" * 60)
    print(f"测试完成! Debug文件保存在: {debugger.debug_dir}")
    print("=" * 60)

    # 显示生成的文件
    print("\n生成的文件:")
    for f in debugger.debug_dir.glob("*"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    test_debugger()
