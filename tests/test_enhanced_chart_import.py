"""测试增强 K 线图表模块的导入"""

def test_app_import():
    """测试 App 导入"""
    from qp.apps.enhanced_chart import EnhancedChartWizardApp
    print("[OK] EnhancedChartWizardApp 导入成功")

    app = EnhancedChartWizardApp()
    print(f"[OK] App 实例化成功: {app.display_name}")
    return app


def test_items_import():
    """测试 Items 导入"""
    from qp.apps.enhanced_chart.items import MAItem
    print("[OK] MAItem 导入成功")
    return MAItem


def test_ui_import():
    """测试 UI 导入"""
    from qp.apps.enhanced_chart.ui import EnhancedChartWizardWidget, MAConfigDialog
    print("[OK] EnhancedChartWizardWidget 导入成功")
    print("[OK] MAConfigDialog 导入成功")
    return EnhancedChartWizardWidget, MAConfigDialog


def test_profiles_integration():
    """测试 profiles 集成"""
    from qp.ui.profiles import _try_import_enhanced_chart

    app_cls = _try_import_enhanced_chart()
    if app_cls:
        print(f"[OK] profiles 集成成功: {app_cls.__name__}")
        return True
    else:
        print("[FAIL] profiles 集成失败")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("测试增强 K 线图表模块导入")
    print("=" * 60)

    try:
        test_app_import()
        print()

        test_items_import()
        print()

        test_ui_import()
        print()

        test_profiles_integration()
        print()

        print("=" * 60)
        print("[PASS] 所有导入测试通过！")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"[FAIL] 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
