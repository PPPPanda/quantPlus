"""测试 MACD 在大索引时不会递归超限"""

import sys
from datetime import datetime, timedelta
from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval
from vnpy.chart.manager import BarManager

sys.path.insert(0, "E:/work/quant/quantPlus/src")

from qp.apps.enhanced_chart.items import MACDItem


def create_large_test_bars(count: int = 1500) -> list[BarData]:
    """创建大量测试K线数据"""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 0)
    base_price = 100.0

    for i in range(count):
        price_change = 0.1 * i + 0.5 * (i % 10 - 5)
        bar = BarData(
            gateway_name="TEST",
            symbol="TEST",
            exchange=Exchange.LOCAL,
            datetime=base_time + timedelta(minutes=i),
            interval=Interval.MINUTE,
            volume=1000 + i * 10,
            turnover=100000,
            open_interest=5000,
            open_price=base_price + price_change,
            high_price=base_price + price_change + 0.5,
            low_price=base_price + price_change - 0.3,
            close_price=base_price + price_change + 0.2,
        )
        bars.append(bar)

    return bars


def test_large_index_no_recursion():
    """测试大索引时不会递归超限"""
    print("=" * 60)
    print("测试大索引（模拟真实场景）")
    print("=" * 60)

    # 创建 1500 根K线（模拟真实场景）
    bars = create_large_test_bars(1500)
    print(f"创建了 {len(bars)} 根K线")

    # 创建 BarManager
    manager = BarManager()
    manager.update_history(bars)

    # 创建 MACDItem
    macd_item = MACDItem(manager)
    macd_item.update_history(bars)

    print("\n测试场景1: 从大索引开始计算（模拟 vnpy 可见区域绘制）")
    print("直接计算索引 1279 的 MACD（缓存为空）")

    try:
        # 直接计算大索引（这会触发从 1279 向前迭代到 period-1）
        dif, dea, macd = macd_item._calculate_macd(1279)

        dif_str = f"{dif:.4f}" if dif is not None else "None"
        dea_str = f"{dea:.4f}" if dea is not None else "None"
        macd_str = f"{macd:.4f}" if macd is not None else "None"

        print(f"索引 1279:")
        print(f"  DIF: {dif_str}")
        print(f"  DEA: {dea_str}")
        print(f"  MACD: {macd_str}")

        # 检查缓存已建立
        print(f"\n缓存状态:")
        print(f"  _ema_fast 长度: {len(macd_item._ema_fast)}")
        print(f"  _ema_slow 长度: {len(macd_item._ema_slow)}")
        print(f"  应该包含索引 11-1279: {11 in macd_item._ema_fast and 1279 in macd_item._ema_fast}")

        print("\n[PASS] 大索引计算成功，无递归超限")

    except RecursionError as e:
        print(f"\n[FAIL] 递归超限错误: {e}")
        raise


def test_skip_calculation():
    """测试跳跃计算（模拟只绘制可见区域）"""
    print("\n" + "=" * 60)
    print("测试场景2: 跳跃计算（只计算可见区域）")
    print("=" * 60)

    bars = create_large_test_bars(1500)
    manager = BarManager()
    manager.update_history(bars)

    macd_item = MACDItem(manager)
    macd_item.update_history(bars)

    # 模拟只计算索引 1279-1380（vnpy 可见区域）
    print("计算索引 1279-1380（102 根K线）")

    try:
        for ix in range(1279, 1381):
            macd_item._draw_bar_picture(ix, bars[ix])

        print(f"\n成功计算 102 根K线")
        print(f"缓存大小: {len(macd_item._dif_values)} 条")

        # 验证几个点
        test_indices = [1279, 1300, 1350, 1380]
        print("\n验证数据:")
        for ix in test_indices:
            dif = macd_item._dif_values.get(ix)
            dea = macd_item._dea_values.get(ix)
            macd = macd_item._macd_values.get(ix)
            dif_str = f"{dif:.3f}" if dif is not None else "None"
            dea_str = f"{dea:.3f}" if dea is not None else "None"
            macd_str = f"{macd:.3f}" if macd is not None else "None"
            print(f"  索引 {ix}: DIF={dif_str}, DEA={dea_str}, MACD={macd_str}")

        print("\n[PASS] 跳跃计算成功")

    except RecursionError as e:
        print(f"\n[FAIL] 递归超限错误: {e}")
        raise


def test_performance():
    """测试性能（迭代 vs 原递归）"""
    print("\n" + "=" * 60)
    print("测试场景3: 性能测试")
    print("=" * 60)

    import time

    bars = create_large_test_bars(1500)
    manager = BarManager()
    manager.update_history(bars)

    macd_item = MACDItem(manager)
    macd_item.update_history(bars)

    # 测试计算所有K线的时间
    start_time = time.time()

    for ix, bar in enumerate(bars):
        macd_item._draw_bar_picture(ix, bar)

    elapsed = time.time() - start_time

    print(f"计算 1500 根K线的 MACD 用时: {elapsed:.3f} 秒")
    print(f"平均每根K线: {elapsed/1500*1000:.2f} 毫秒")

    if elapsed < 1.0:
        print("[PASS] 性能良好（< 1秒）")
    else:
        print(f"[WARNING] 性能较慢（{elapsed:.3f} 秒）")


if __name__ == "__main__":
    try:
        test_large_index_no_recursion()
        test_skip_calculation()
        test_performance()

        print("\n" + "=" * 60)
        print("[PASS] 所有大索引测试通过！")
        print("修复后的迭代算法可以处理任意大的索引")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[FAIL] 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
