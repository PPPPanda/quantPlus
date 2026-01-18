"""手动测试 MAItem 的使用方式"""

import sys
from datetime import datetime, timedelta
import pyqtgraph as pg
from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval
from vnpy.chart.manager import BarManager
from vnpy.chart.widget import ChartWidget

# 确保导入路径正确
sys.path.insert(0, "E:/work/quant/quantPlus/src")

from qp.apps.enhanced_chart.items import MAItem

def create_test_bars(count: int = 100) -> list[BarData]:
    """创建测试用的K线数据"""
    bars = []
    base_time = datetime(2024, 1, 1, 9, 0)
    base_price = 100.0

    for i in range(count):
        bar = BarData(
            gateway_name="TEST",
            symbol="TEST",
            exchange=Exchange.LOCAL,
            datetime=base_time + timedelta(minutes=i),
            interval=Interval.MINUTE,
            volume=1000,
            turnover=100000,
            open_interest=5000,
            open_price=base_price + i * 0.1,
            high_price=base_price + i * 0.1 + 0.5,
            low_price=base_price + i * 0.1 - 0.3,
            close_price=base_price + i * 0.1 + 0.2,
        )
        bars.append(bar)

    return bars

def test_ma_item_basic():
    """测试 MAItem 基本功能"""
    print("=" * 60)
    print("测试 1: MAItem 基本初始化")
    print("=" * 60)

    # 创建 BarManager
    manager = BarManager()

    # 创建测试数据
    bars = create_test_bars(100)
    print(f"创建了 {len(bars)} 根测试K线")

    # 更新数据到 manager
    manager.update_history(bars)
    print(f"BarManager 中的K线数量: {len(manager.get_all_bars())}")

    # 创建 MAItem
    ma_item = MAItem(manager, period=20, color="yellow")
    print(f"创建 MAItem: period={ma_item.period}, color={ma_item.color}")

    # 检查 _bar_picutures 是否为空
    print(f"初始 _bar_picutures 长度: {len(ma_item._bar_picutures)}")

    # 调用 update_history
    print("\n调用 update_history...")
    ma_item.update_history(bars)
    print(f"update_history 后 _bar_picutures 长度: {len(ma_item._bar_picutures)}")

    # 检查 MA 计算
    print("\n测试 MA 计算:")
    for ix in [0, 10, 19, 20, 50]:
        ma_value = ma_item._calculate_ma(ix)
        print(f"  索引 {ix}: MA = {ma_value}")

    print("\n[PASS] 基本功能测试通过")
    return manager, ma_item


def test_chart_widget_integration():
    """测试与 ChartWidget 的集成"""
    print("\n" + "=" * 60)
    print("测试 2: ChartWidget 集成")
    print("=" * 60)

    # 创建 BarManager
    manager = BarManager()
    bars = create_test_bars(100)
    manager.update_history(bars)

    # 模拟 ChartWidget 的 add_item 行为
    print("模拟 add_item 流程:")

    # Step 1: 创建 item (这是 ChartWidget.add_item 内部做的)
    item = MAItem(manager)  # 注意：只传入 manager
    print(f"  1. 创建 item: period={item.period}, color={item.color} (默认值)")

    # Step 2: 手动设置参数
    item.period = 30
    item.color = "cyan"
    item._pen = pg.mkPen(color="cyan", width=2)
    print(f"  2. 设置参数: period={item.period}, color={item.color}")

    # Step 3: 调用 update_history
    item.update_history(bars)
    print(f"  3. update_history: _bar_picutures 长度 = {len(item._bar_picutures)}")

    # Step 4: 测试绘制
    print("\n  4. 测试绘制前几根K线:")
    for ix in range(min(5, len(bars))):
        bar = manager.get_bar(ix)
        if bar:
            picture = item._draw_bar_picture(ix, bar)
            print(f"     索引 {ix}: picture 对象 = {picture}")

    print("\n[PASS] ChartWidget 集成测试通过")
    return item


def test_items_dict_access():
    """测试 _items 字典访问方式"""
    print("\n" + "=" * 60)
    print("测试 3: _items 字典访问")
    print("=" * 60)

    # 模拟 ChartWidget._items 字典
    _items = {}

    manager = BarManager()
    bars = create_test_bars(100)
    manager.update_history(bars)

    # 模拟 add_item
    item_name = "ma_20"
    item = MAItem(manager)
    _items[item_name] = item
    print(f"添加 item: '{item_name}' -> {item}")

    # 测试获取
    ma_item = _items.get("ma_20")
    print(f"获取 item: _items.get('ma_20') = {ma_item}")

    # 错误的访问方式（原代码的问题）
    print("\n错误的访问方式:")
    try:
        # 这是错误的！_items 不是按 plot_name 分组的
        items = _items.get("candle", {})
        print(f"  _items.get('candle', {{}}) = {items}")
        result = items.get("ma_20")
        print(f"  items.get('ma_20') = {result}")
    except AttributeError as e:
        print(f"  [ERROR] {e}")

    # 正确的访问方式
    print("\n正确的访问方式:")
    ma_item = _items.get("ma_20")
    print(f"  _items.get('ma_20') = {ma_item}")

    print("\n[PASS] 字典访问测试通过")


if __name__ == "__main__":
    try:
        test_ma_item_basic()
        test_chart_widget_integration()
        test_items_dict_access()

        print("\n" + "=" * 60)
        print("[PASS] 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[FAIL] 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
