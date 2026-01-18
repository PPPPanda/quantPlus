"""测试图例功能"""

import sys
from PySide6 import QtWidgets
import pyqtgraph as pg

def test_legend_basic():
    """测试基本图例功能"""
    print("=" * 60)
    print("测试 pyqtgraph LegendItem")
    print("=" * 60)

    # 创建应用
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # 创建图表窗口
    win = pg.GraphicsLayoutWidget(show=True, title="图例测试")
    win.resize(800, 600)

    # 创建plot
    plot = win.addPlot(title="测试图表")

    # 创建图例
    legend = pg.LegendItem(offset=(70, 10))
    legend.setParentItem(plot.vb)

    # 设置样式
    legend.setBrush(pg.mkBrush(color=(0, 0, 0, 100)))  # 半透明黑色背景
    legend.setPen(pg.mkPen(color=(200, 200, 200), width=1))

    # 添加一些示例数据和图例项
    colors = ["yellow", "cyan", "magenta", "green"]
    periods = [5, 10, 20, 60]

    for color, period in zip(colors, periods):
        # 创建线条样本
        sample_line = pg.PlotDataItem(pen=pg.mkPen(color=color, width=2))
        legend.addItem(sample_line, f"MA{period}")
        print(f"添加图例项: MA{period} ({color})")

    print("\n图例已创建，应该显示在图表右上角")
    print("包含 4 个MA指标: MA5, MA10, MA20, MA60")
    print("\n关闭窗口以结束测试")

    # 运行应用
    if sys.flags.interactive != 1 or not hasattr(QtWidgets.QApplication, 'exec'):
        app.exec()

if __name__ == "__main__":
    test_legend_basic()
