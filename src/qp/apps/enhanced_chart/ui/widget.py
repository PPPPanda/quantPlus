"""增强 K 线图表 Widget"""

from datetime import datetime, timedelta
from tzlocal import get_localzone_name
import pyqtgraph as pg

from vnpy.event import EventEngine
from vnpy.chart import ChartWidget
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import QtWidgets, QtCore
from vnpy.trader.constant import Interval
from vnpy.trader.utility import ZoneInfo
from vnpy.trader.object import ContractData

# 继承官方 Widget
from vnpy_chartwizard.ui.widget import ChartWizardWidget

# 导入自定义 Item 和对话框
from ..items import MAItem, MACDItem, OpenInterestItem
from .dialogs import MAConfigDialog

# 导入官方的 Item
from vnpy.chart.item import CandleItem, VolumeItem


class EnhancedChartWizardWidget(ChartWizardWidget):
    """
    增强 K 线图表 Widget

    扩展功能：
    - 工具栏（周期选择、指标菜单）
    - MA 指标添加
    - 多周期支持
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """构造函数"""
        # 先调用父类构造函数
        super().__init__(main_engine, event_engine)

        # 当前周期（默认1分钟）
        self.current_interval: Interval = Interval.MINUTE

        # 跟踪每个图表的图例 {vt_symbol: legend_item}
        self.legends: dict[str, object] = {}  # K线图的MA图例
        self.macd_legends: dict[str, object] = {}  # MACD图的DIF/DEA图例

        # 添加工具栏
        self.add_toolbar()

    def add_toolbar(self) -> None:
        """添加工具栏"""
        # 创建工具栏
        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar.setLayout(toolbar_layout)

        # 周期选择
        toolbar_layout.addWidget(QtWidgets.QLabel("K线周期:"))
        self.interval_combo = QtWidgets.QComboBox()
        self.interval_combo.addItems(["1分钟", "5分钟", "15分钟", "60分钟"])
        self.interval_combo.setCurrentText("1分钟")
        self.interval_combo.currentTextChanged.connect(self.on_interval_changed)
        toolbar_layout.addWidget(self.interval_combo)

        # 指标菜单按钮
        self.indicator_button = QtWidgets.QPushButton("添加指标")
        self.indicator_menu = QtWidgets.QMenu(self.indicator_button)  # 设置父对象
        self.indicator_menu.addAction("均线 MA", self.add_ma_indicator)
        # 预留：MACD、BOLL 等
        # self.indicator_menu.addAction("MACD", self.add_macd_indicator)
        # self.indicator_menu.addAction("布林带 BOLL", self.add_boll_indicator)
        self.indicator_button.setMenu(self.indicator_menu)
        toolbar_layout.addWidget(self.indicator_button)

        toolbar_layout.addStretch()

        # 插入到主布局顶部（在原有的 symbol_line + button 之上）
        main_layout: QtWidgets.QVBoxLayout = self.layout()
        main_layout.insertWidget(0, toolbar)

    def on_interval_changed(self, text: str) -> None:
        """周期切换事件"""
        interval_map = {
            "1分钟": Interval.MINUTE,
            "5分钟": (Interval.MINUTE, 5),   # 5分钟需要特殊处理
            "15分钟": (Interval.MINUTE, 15),
            "60分钟": Interval.HOUR,
        }

        new_interval = interval_map.get(text, Interval.MINUTE)

        # 如果是元组，表示需要合成（暂时简化，使用1分钟）
        if isinstance(new_interval, tuple):
            self.current_interval = new_interval[0]
            # TODO: 实现 K 线合成逻辑
            QtWidgets.QMessageBox.information(
                self,
                "提示",
                f"{text} 周期暂未实现，将使用 1 分钟数据"
            )
        else:
            self.current_interval = new_interval

    def add_ma_indicator(self) -> None:
        """添加均线指标"""
        print("[DEBUG] add_ma_indicator 被调用")

        # 弹出配置对话框
        dialog = MAConfigDialog(self)
        if not dialog.exec():
            return  # 用户取消

        # 获取参数
        params = dialog.get_params()
        period = params["period"]
        color = params["color"]

        # 获取当前活跃的图表
        current_chart = self.get_current_chart()
        if not current_chart:
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "请先创建一个图表"
            )
            return

        # 获取当前合约代码
        current_index = self.tab.currentIndex()
        vt_symbol = self.tab.tabText(current_index)

        # 添加 MA 指标到图表
        try:
            # MAItem 需要传入 manager, period, color
            current_chart.add_item(
                MAItem,
                f"ma_{period}",  # item_name
                "candle",        # plot_name (添加到蜡烛图区域)
            )

            # 需要手动设置参数（vnpy 的 add_item 不支持直接传参）
            # 获取刚添加的 item (直接从 _items 字典获取)
            ma_item = current_chart._items.get(f"ma_{period}")
            print(f"[DEBUG] 获取到的 ma_item: {ma_item}")

            if ma_item:
                ma_item.period = period
                ma_item.color = color
                ma_item._pen = pg.mkPen(color=color, width=2)

                # 必须调用 update_history 初始化 _bar_picutures
                bars = current_chart._manager.get_all_bars()
                print(f"[DEBUG] 历史K线数量: {len(bars)}")
                ma_item.update_history(bars)

                # 添加到图例
                legend = self.get_or_create_legend(current_chart, vt_symbol)

                # 创建一个简单的线条作为图例样本
                sample_line = pg.PlotDataItem(pen=pg.mkPen(color=color, width=2))
                legend.addItem(sample_line, f"MA{period}")
                print(f"[DEBUG] 已添加 MA{period} 到图例")

            QtWidgets.QMessageBox.information(
                self,
                "成功",
                f"已添加 MA{period} 指标（{color}）"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"添加指标失败: {e}"
            )

    def get_or_create_legend(self, chart: ChartWidget, vt_symbol: str) -> pg.LegendItem:
        """获取或创建图表的图例（K线图MA图例）"""
        if vt_symbol in self.legends:
            return self.legends[vt_symbol]

        # 创建图例（增加x偏移避免与日期信息框重合）
        legend = pg.LegendItem(offset=(180, 10))  # 右上角偏移，向右移动更多
        legend.setParentItem(chart._first_plot.vb)  # 添加到主图表区域

        # 设置样式
        legend.setBrush(pg.mkBrush(color=(0, 0, 0, 100)))  # 半透明黑色背景
        legend.setPen(pg.mkPen(color=(200, 200, 200), width=1))

        # 保存引用
        self.legends[vt_symbol] = legend

        print(f"[DEBUG] 为 {vt_symbol} 创建K线图例 (offset=180)")
        return legend

    def _add_macd_legend(self, chart: ChartWidget, vt_symbol: str) -> None:
        """为MACD区域添加DIF和DEA图例"""
        if vt_symbol in self.macd_legends:
            return  # 已经添加过

        # 获取MACD的plot
        macd_plot = chart._plots.get("macd")
        if not macd_plot:
            return

        # 创建MACD图例
        legend = pg.LegendItem(offset=(70, 10))  # 右上角偏移
        legend.setParentItem(macd_plot.vb)  # 添加到MACD图表区域

        # 设置样式
        legend.setBrush(pg.mkBrush(color=(0, 0, 0, 100)))  # 半透明黑色背景
        legend.setPen(pg.mkPen(color=(200, 200, 200), width=1))

        # 添加DIF线图例（黄色）
        dif_line = pg.PlotDataItem(pen=pg.mkPen(color="yellow", width=1))
        legend.addItem(dif_line, "DIF")

        # 添加DEA线图例（青色）
        dea_line = pg.PlotDataItem(pen=pg.mkPen(color="cyan", width=1))
        legend.addItem(dea_line, "DEA")

        # 保存引用
        self.macd_legends[vt_symbol] = legend

        print(f"[DEBUG] 为 {vt_symbol} 创建MACD图例 (DIF/DEA)")


    def get_current_chart(self) -> ChartWidget | None:
        """获取当前活跃的图表"""
        current_index = self.tab.currentIndex()
        print(f"[DEBUG] 当前tab索引: {current_index}")
        if current_index < 0:
            print("[DEBUG] 没有打开的图表")
            return None

        vt_symbol = self.tab.tabText(current_index)
        print(f"[DEBUG] 当前合约: {vt_symbol}")
        chart = self.charts.get(vt_symbol)
        print(f"[DEBUG] 获取到的图表: {chart}")
        return chart

    def create_chart(self) -> ChartWidget:
        """
        重写：创建图表对象（调整布局顺序）.

        布局顺序：
        1. candle (K线图) - 隐藏x轴
        2. volume (成交量 + 持仓量曲线) - 隐藏x轴，上移
        3. macd (MACD指标) - 显示x轴，新增

        Returns:
            配置好的 ChartWidget
        """
        chart = ChartWidget()

        # 1. K线图区域
        chart.add_plot("candle", hide_x_axis=True)
        chart.add_item(CandleItem, "candle", "candle")

        # 2. 成交量区域（上移，隐藏x轴）
        chart.add_plot("volume", maximum_height=150, hide_x_axis=True)
        chart.add_item(VolumeItem, "volume", "volume")
        # 叠加持仓量曲线（橙色线条）
        chart.add_item(OpenInterestItem, "open_interest", "volume")

        # 3. MACD 区域（新增，显示x轴）
        chart.add_plot("macd", maximum_height=150, hide_x_axis=False)
        chart.add_item(MACDItem, "macd", "macd")

        # 添加光标
        chart.add_cursor()

        print("[DEBUG] 创建增强图表: candle + volume(+OI) + macd")
        return chart

    def close_tab(self, index: int) -> None:
        """重写：关闭标签页时清理图例"""
        vt_symbol = self.tab.tabText(index)

        # 清理K线图例引用
        if vt_symbol in self.legends:
            print(f"[DEBUG] 清理 {vt_symbol} 的K线图例")
            self.legends.pop(vt_symbol)

        # 清理MACD图例引用
        if vt_symbol in self.macd_legends:
            print(f"[DEBUG] 清理 {vt_symbol} 的MACD图例")
            self.macd_legends.pop(vt_symbol)

        # 调用父类方法
        super().close_tab(index)

    def new_chart(self) -> None:
        """
        重写：创建新图表（支持多周期）

        注：当前简化实现，仅使用 Interval.MINUTE
        """
        # 获取合约代码
        vt_symbol: str = self.symbol_line.text()
        if not vt_symbol:
            return

        if vt_symbol in self.charts:
            return

        if "LOCAL" not in vt_symbol:
            contract: ContractData | None = self.main_engine.get_contract(vt_symbol)
            if not contract:
                return

        # 调用父类的 new_chart（会创建图表并查询历史数据）
        # 这里暂时使用 Interval.MINUTE，未来可扩展
        super().new_chart()

        # 为新创建的图表添加 MACD 图例
        chart = self.charts.get(vt_symbol)
        if chart:
            self._add_macd_legend(chart, vt_symbol)

        # 可以在这里根据 self.current_interval 重新查询数据
        # if chart and self.current_interval != Interval.MINUTE:
        #     # TODO: 实现多周期数据查询
        #     pass
