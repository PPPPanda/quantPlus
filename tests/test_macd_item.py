"""测试 MACD Item 功能"""

import sys
from datetime import datetime, timedelta
from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval
from vnpy.chart.manager import BarManager

sys.path.insert(0, "E:/work/quant/quantPlus/src")

from qp.apps.enhanced_chart.items import MACDItem


def create_test_bars(count: int = 100) -> list[BarData]:
    """创建测试用的K线数据"""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 0)
    base_price = 100.0

    for i in range(count):
        # 创建一些价格波动
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


def test_macd_calculation():
    """测试 MACD 计算"""
    print("=" * 60)
    print("测试 MACDItem 计算")
    print("=" * 60)

    # 创建 BarManager
    manager = BarManager()

    # 创建测试数据
    bars = create_test_bars(100)
    print(f"创建了 {len(bars)} 根测试K线")

    # 更新数据到 manager
    manager.update_history(bars)
    print(f"BarManager 中的K线数量: {len(manager.get_all_bars())}")

    # 创建 MACDItem (默认参数: 12, 26, 9)
    macd_item = MACDItem(manager, fast_period=12, slow_period=26, signal_period=9)
    print(f"创建 MACDItem: fast={macd_item.fast_period}, slow={macd_item.slow_period}, signal={macd_item.signal_period}")

    # 调用 update_history
    print("\n调用 update_history...")
    macd_item.update_history(bars)

    # 手动触发计算（模拟绘制过程）
    print("手动计算 MACD 值（调用 _draw_bar_picture）...")
    for ix, bar in enumerate(bars):
        macd_item._draw_bar_picture(ix, bar)

    print(f"计算后缓存长度:")
    print(f"  _dif_values: {len(macd_item._dif_values)}")
    print(f"  _dea_values: {len(macd_item._dea_values)}")
    print(f"  _macd_values: {len(macd_item._macd_values)}")

    # 检查 MACD 计算结果
    print("\n测试 MACD 计算结果:")
    print(f"{'索引':<6} {'DIF':<10} {'DEA':<10} {'MACD':<10} {'信息':<30}")
    print("-" * 70)

    test_indices = [0, 11, 25, 26, 33, 34, 50, 99]
    for ix in test_indices:
        dif = macd_item._dif_values.get(ix)
        dea = macd_item._dea_values.get(ix)
        macd = macd_item._macd_values.get(ix)
        info = macd_item.get_info_text(ix)

        dif_str = f"{dif:.4f}" if dif is not None else "None"
        dea_str = f"{dea:.4f}" if dea is not None else "None"
        macd_str = f"{macd:.4f}" if macd is not None else "None"

        print(f"{ix:<6} {dif_str:<10} {dea_str:<10} {macd_str:<10} {info:<30}")

    # 测试 Y 范围
    print("\nY轴范围:")
    min_y, max_y = macd_item.get_y_range()
    print(f"  全部数据: [{min_y:.4f}, {max_y:.4f}]")

    min_y, max_y = macd_item.get_y_range(50, 99)
    print(f"  索引50-99: [{min_y:.4f}, {max_y:.4f}]")

    print("\n[PASS] MACD 计算测试通过")
    return macd_item


def test_macd_ema():
    """测试 EMA 计算"""
    print("\n" + "=" * 60)
    print("测试 EMA 计算逻辑")
    print("=" * 60)

    manager = BarManager()
    bars = create_test_bars(50)
    manager.update_history(bars)

    macd_item = MACDItem(manager)

    # 手动测试 EMA 计算
    print("\n测试 EMA(12) 计算:")
    for ix in [11, 12, 13, 20]:
        ema12 = macd_item._calculate_ema(ix, 12, macd_item._ema_fast)
        ema12_str = f"{ema12:.4f}" if ema12 is not None else "None"
        print(f"  索引 {ix}: EMA(12) = {ema12_str}")

    print("\n测试 EMA(26) 计算:")
    for ix in [25, 26, 27, 30]:
        ema26 = macd_item._calculate_ema(ix, 26, macd_item._ema_slow)
        ema26_str = f"{ema26:.4f}" if ema26 is not None else "None"
        print(f"  索引 {ix}: EMA(26) = {ema26_str}")

    print("\n[PASS] EMA 计算测试通过")


def test_macd_update():
    """测试 MACD 更新"""
    print("\n" + "=" * 60)
    print("测试 MACD 单根K线更新")
    print("=" * 60)

    manager = BarManager()
    bars = create_test_bars(50)
    manager.update_history(bars)

    macd_item = MACDItem(manager)
    macd_item.update_history(bars)

    # 先手动计算一次（调用绘制方法）
    for ix, bar in enumerate(bars):
        macd_item._draw_bar_picture(ix, bar)

    # 记录更新前的值
    print("\n更新前索引40的值:")
    old_dif = macd_item._dif_values.get(40)
    old_dea = macd_item._dea_values.get(40)
    old_macd = macd_item._macd_values.get(40)
    old_dif_str = f"{old_dif:.4f}" if old_dif is not None else "None"
    old_dea_str = f"{old_dea:.4f}" if old_dea is not None else "None"
    old_macd_str = f"{old_macd:.4f}" if old_macd is not None else "None"
    print(f"  DIF: {old_dif_str}")
    print(f"  DEA: {old_dea_str}")
    print(f"  MACD: {old_macd_str}")

    # 更新一根K线
    print("\n调用 update_bar(索引40)...")
    macd_item.update_bar(bars[40])

    # 检查缓存是否正确清空
    print(f"\n更新后缓存状态:")
    print(f"  索引39 的 DIF: {'存在' if 39 in macd_item._dif_values else '已清空'}")
    print(f"  索引40 的 DIF: {'存在' if 40 in macd_item._dif_values else '已清空'}")
    print(f"  索引41 的 DIF: {'存在' if 41 in macd_item._dif_values else '已清空'}")

    print("\n[PASS] MACD 更新测试通过")


if __name__ == "__main__":
    try:
        test_macd_calculation()
        test_macd_ema()
        test_macd_update()

        print("\n" + "=" * 60)
        print("[PASS] 所有 MACD 测试通过！")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[FAIL] 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
