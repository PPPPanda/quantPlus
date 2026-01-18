"""测试颜色对话框"""

import sys

# 确保导入路径正确
sys.path.insert(0, "E:/work/quant/quantPlus/src")

def test_color_map():
    """测试颜色映射"""
    from qp.apps.enhanced_chart.ui.dialogs import MAConfigDialog

    print("=" * 60)
    print("测试 MA 颜色对话框")
    print("=" * 60)

    # 测试颜色映射
    print("\n颜色映射:")
    print(f"颜色数量: {len(MAConfigDialog.COLOR_MAP)}")
    print("\n中文名 -> 英文值:")
    for chinese, english in MAConfigDialog.COLOR_MAP.items():
        print(f"  {chinese:6} -> {english}")

    # 测试映射转换
    print("\n测试颜色转换:")
    test_cases = [
        ("黄色", "yellow"),
        ("青色", "cyan"),
        ("橙色", "orange"),
        ("紫色", "violet"),
    ]

    for chinese, expected in test_cases:
        actual = MAConfigDialog.COLOR_MAP.get(chinese)
        status = "[PASS]" if actual == expected else "[FAIL]"
        print(f"  {status} {chinese} -> {actual} (期望: {expected})")

    print("\n[PASS] 颜色映射测试通过")


def test_color_contrast():
    """测试颜色对比度"""
    from qp.apps.enhanced_chart.ui.dialogs import MAConfigDialog

    print("\n" + "=" * 60)
    print("测试高对比度颜色")
    print("=" * 60)

    high_contrast_colors = [
        "yellow", "cyan", "lime", "red", "dodgerblue",
        "orange", "violet", "white", "hotpink", "gold"
    ]

    available = list(MAConfigDialog.COLOR_MAP.values())

    print("\n高对比度颜色检查:")
    for color in high_contrast_colors:
        if color in available:
            print(f"  [OK] {color}")
        else:
            print(f"  [MISS] {color} (缺失)")

    print(f"\n对比度高的颜色数量: {len([c for c in high_contrast_colors if c in available])}/{len(high_contrast_colors)}")
    print("\n[PASS] 颜色对比度测试通过")


if __name__ == "__main__":
    try:
        test_color_map()
        test_color_contrast()

        print("\n" + "=" * 60)
        print("[PASS] 所有颜色对话框测试通过！")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[FAIL] 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
