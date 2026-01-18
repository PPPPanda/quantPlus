"""测试 MACD 模块导入"""

def test_macd_item_import():
    """测试 MACDItem 导入"""
    from qp.apps.enhanced_chart.items import MACDItem
    print("[OK] MACDItem 导入成功")
    return MACDItem


def test_macd_in_items():
    """测试 items 模块包含 MACDItem"""
    from qp.apps.enhanced_chart import items
    assert hasattr(items, 'MACDItem'), "items 模块应该包含 MACDItem"
    assert hasattr(items, 'MAItem'), "items 模块应该包含 MAItem"
    print("[OK] items 模块导出正确")


def test_widget_imports():
    """测试 Widget 的导入"""
    from qp.apps.enhanced_chart.ui.widget import EnhancedChartWizardWidget
    print("[OK] EnhancedChartWizardWidget 导入成功")

    # 检查是否导入了 MACDItem
    import inspect
    source = inspect.getsource(EnhancedChartWizardWidget)
    assert 'MACDItem' in source, "EnhancedChartWizardWidget 应该导入 MACDItem"
    assert 'create_chart' in source, "EnhancedChartWizardWidget 应该重写 create_chart"
    print("[OK] Widget 包含 MACD 相关代码")


def test_create_chart_method():
    """测试 create_chart 方法"""
    from qp.apps.enhanced_chart.ui.widget import EnhancedChartWizardWidget
    import inspect

    # 检查是否有 create_chart 方法
    assert hasattr(EnhancedChartWizardWidget, 'create_chart'), "应该有 create_chart 方法"

    # 检查方法内容
    source = inspect.getsource(EnhancedChartWizardWidget.create_chart)
    assert '"macd"' in source, "create_chart 应该创建 macd plot"
    assert 'MACDItem' in source, "create_chart 应该添加 MACDItem"
    assert 'maximum_height=150' in source, "应该设置 MACD 高度为 150"

    print("[OK] create_chart 方法实现正确")


if __name__ == "__main__":
    print("=" * 60)
    print("测试 MACD 模块导入")
    print("=" * 60)

    try:
        test_macd_item_import()
        print()

        test_macd_in_items()
        print()

        test_widget_imports()
        print()

        test_create_chart_method()
        print()

        print("=" * 60)
        print("[PASS] 所有导入测试通过！")
        print("=" * 60)
        print()
        print("MACD 功能已完整集成，可以测试运行：")
        print("  uv run python -m qp.runtime.trader_app --gateway tts --profile all")

    except Exception as e:
        print()
        print("=" * 60)
        print(f"[FAIL] 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
