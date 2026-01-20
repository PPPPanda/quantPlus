"""
增强版回测K线图表对话框

替换 vnpy_ctabacktester 的 CandleChartDialog，提供：
- 增强图表布局（K线 + MACD + 成交量+持仓量）
- 交易信号标记（买卖点、盈亏连线）
- MA 均线支持
"""

from typing import Optional

import pyqtgraph as pg

from vnpy.trader.ui import QtWidgets, QtCore, QtGui
from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.chart import ChartWidget
from vnpy.chart.item import CandleItem, VolumeItem

from ..items import MACDItem, OpenInterestItem, MAItem
from .dialogs import MAConfigDialog


def generate_trade_pairs(trades: list) -> list:
    """
    将交易配对（开仓-平仓）

    Args:
        trades: TradeData 列表

    Returns:
        配对后的交易列表
    """
    long_trades: list = []
    short_trades: list = []
    trade_pairs: list = []

    for trade in trades:
        if trade.direction == Direction.LONG:
            long_trades.append(trade)
        else:
            short_trades.append(trade)

        # 尝试配对
        if long_trades and short_trades:
            # 找到相反方向的交易进行配对
            if trade.direction == Direction.LONG and short_trades:
                # 当前是买入，找之前的卖出（平多）
                pass
            elif trade.direction == Direction.SHORT and long_trades:
                # 当前是卖出，找之前的买入（平空）
                pass

    # 简化配对逻辑：按时间顺序，相邻的买卖配对
    i = 0
    while i < len(trades) - 1:
        current = trades[i]
        next_trade = trades[i + 1]

        # 检查是否可以配对
        if current.direction != next_trade.direction:
            if current.direction == Direction.LONG:
                # 做多：买入开仓 -> 卖出平仓
                trade_pairs.append({
                    "open_dt": current.datetime,
                    "open_price": current.price,
                    "close_dt": next_trade.datetime,
                    "close_price": next_trade.price,
                    "direction": Direction.LONG,
                    "volume": min(current.volume, next_trade.volume),
                })
            else:
                # 做空：卖出开仓 -> 买入平仓
                trade_pairs.append({
                    "open_dt": current.datetime,
                    "open_price": current.price,
                    "close_dt": next_trade.datetime,
                    "close_price": next_trade.price,
                    "direction": Direction.SHORT,
                    "volume": min(current.volume, next_trade.volume),
                })
            i += 2
        else:
            i += 1

    return trade_pairs


class EnhancedCandleChartDialog(QtWidgets.QDialog):
    """
    增强版回测K线图表对话框

    兼容 vnpy_ctabacktester.CandleChartDialog 接口
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # 数据映射
        self.dt_ix_map: dict = {}      # datetime -> index
        self.ix_bar_map: dict = {}     # index -> BarData

        # 价格范围
        self.high_price: float = 0
        self.low_price: float = 0
        self.price_range: float = 0

        # 绘图元素
        self.items: list = []

        # MA 图例
        self.ma_legend: Optional[pg.LegendItem] = None

        # 更新标志
        self._updated: bool = False

        self.init_ui()

    def init_ui(self) -> None:
        """初始化界面"""
        self.setWindowTitle("增强版回测K线图表")
        self.resize(1400, 900)

        # 主布局
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        # 图例说明
        legend_layout = QtWidgets.QHBoxLayout()

        label1 = QtWidgets.QLabel("红色虚线 ── 盈利交易")
        label1.setStyleSheet("color: red;")
        legend_layout.addWidget(label1)

        label2 = QtWidgets.QLabel("绿色虚线 ── 亏损交易")
        label2.setStyleSheet("color: #00FF00;")
        legend_layout.addWidget(label2)

        legend_layout.addSpacing(20)

        label3 = QtWidgets.QLabel("黄色↑ ── 买入开仓")
        label3.setStyleSheet("color: yellow;")
        legend_layout.addWidget(label3)

        label4 = QtWidgets.QLabel("黄色↓ ── 卖出平仓")
        label4.setStyleSheet("color: yellow;")
        legend_layout.addWidget(label4)

        legend_layout.addSpacing(20)

        label5 = QtWidgets.QLabel("品红↓ ── 卖出开仓")
        label5.setStyleSheet("color: magenta;")
        legend_layout.addWidget(label5)

        label6 = QtWidgets.QLabel("品红↑ ── 买入平仓")
        label6.setStyleSheet("color: magenta;")
        legend_layout.addWidget(label6)

        legend_layout.addStretch()

        # 添加指标按钮
        self.indicator_button = QtWidgets.QPushButton("添加指标")
        self.indicator_menu = QtWidgets.QMenu(self.indicator_button)
        self.indicator_menu.addAction("均线 MA", self.add_ma_indicator)
        self.indicator_button.setMenu(self.indicator_menu)
        legend_layout.addWidget(self.indicator_button)

        main_layout.addLayout(legend_layout)

        # 创建图表
        self.chart = ChartWidget()

        # 1. K线图区域
        self.chart.add_plot("candle", hide_x_axis=True)
        self.chart.add_item(CandleItem, "candle", "candle")

        # 2. 成交量区域（叠加持仓量）
        self.chart.add_plot("volume", maximum_height=150, hide_x_axis=True)
        self.chart.add_item(VolumeItem, "volume", "volume")
        self.chart.add_item(OpenInterestItem, "open_interest", "volume")

        # 3. MACD 区域
        self.chart.add_plot("macd", maximum_height=150, hide_x_axis=False)
        self.chart.add_item(MACDItem, "macd", "macd")

        # 添加光标
        self.chart.add_cursor()

        main_layout.addWidget(self.chart)

        # 添加 MACD 图例
        self._add_macd_legend()

    def _add_macd_legend(self) -> None:
        """为 MACD 区域添加图例"""
        macd_plot = self.chart._plots.get("macd")
        if not macd_plot:
            return

        legend = pg.LegendItem(offset=(70, 10))
        legend.setParentItem(macd_plot.vb)
        legend.setBrush(pg.mkBrush(color=(0, 0, 0, 100)))
        legend.setPen(pg.mkPen(color=(200, 200, 200), width=1))

        dif_line = pg.PlotDataItem(pen=pg.mkPen(color="yellow", width=1))
        legend.addItem(dif_line, "DIF")

        dea_line = pg.PlotDataItem(pen=pg.mkPen(color="cyan", width=1))
        legend.addItem(dea_line, "DEA")

    def is_updated(self) -> bool:
        """检查是否已更新数据"""
        return self._updated

    def update_history(self, history: list) -> None:
        """
        更新历史K线数据

        Args:
            history: BarData 列表
        """
        self.dt_ix_map.clear()
        self.ix_bar_map.clear()
        self.items.clear()

        if not history:
            return

        # 建立映射
        for ix, bar in enumerate(history):
            self.dt_ix_map[bar.datetime] = ix
            self.ix_bar_map[ix] = bar

        # 计算价格范围
        self.high_price = max(bar.high_price for bar in history)
        self.low_price = min(bar.low_price for bar in history)
        self.price_range = self.high_price - self.low_price

        # 更新图表
        self.chart.update_history(history)

        self._updated = True

    def update_trades(self, trades: list) -> None:
        """
        更新交易数据（在图表上显示买卖信号）

        Args:
            trades: TradeData 列表
        """
        if not trades or not self.dt_ix_map:
            return

        # 清除旧的交易标记
        candle_plot = self.chart.get_plot("candle")
        for item in self.items:
            candle_plot.removeItem(item)
        self.items.clear()

        # 生成交易配对
        trade_pairs = generate_trade_pairs(trades)

        scatter_data: list = []
        y_adjustment = self.price_range * 0.001

        for pair in trade_pairs:
            open_dt = pair["open_dt"]
            close_dt = pair["close_dt"]

            # 获取索引
            open_ix = self.dt_ix_map.get(open_dt)
            close_ix = self.dt_ix_map.get(close_dt)

            if open_ix is None or close_ix is None:
                continue

            open_bar = self.ix_bar_map.get(open_ix)
            close_bar = self.ix_bar_map.get(close_ix)

            if not open_bar or not close_bar:
                continue

            open_price = pair["open_price"]
            close_price = pair["close_price"]

            # 绘制盈亏连线
            x = [open_ix, close_ix]
            y = [open_price, close_price]

            if pair["direction"] == Direction.LONG and close_price >= open_price:
                color = "r"  # 盈利
            elif pair["direction"] == Direction.SHORT and close_price <= open_price:
                color = "r"  # 盈利
            else:
                color = "g"  # 亏损

            pen = pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine)
            line = pg.PlotCurveItem(x, y, pen=pen)
            self.items.append(line)
            candle_plot.addItem(line)

            # 确定标记颜色和位置
            if pair["direction"] == Direction.LONG:
                scatter_color = "yellow"
                open_symbol = "t1"   # 向上箭头
                close_symbol = "t"   # 向下箭头
                open_y = open_bar.low_price - y_adjustment
                close_y = close_bar.high_price + y_adjustment
            else:
                scatter_color = "magenta"
                open_symbol = "t"    # 向下箭头
                close_symbol = "t1"  # 向上箭头
                open_y = open_bar.high_price + y_adjustment
                close_y = close_bar.low_price - y_adjustment

            # 添加散点
            scatter_data.append({
                "pos": (open_ix, open_y),
                "size": 12,
                "pen": pg.mkPen(QtGui.QColor(scatter_color)),
                "brush": pg.mkBrush(QtGui.QColor(scatter_color)),
                "symbol": open_symbol
            })
            scatter_data.append({
                "pos": (close_ix, close_y),
                "size": 12,
                "pen": pg.mkPen(QtGui.QColor(scatter_color)),
                "brush": pg.mkBrush(QtGui.QColor(scatter_color)),
                "symbol": close_symbol
            })

            # 添加交易量标签
            volume = pair["volume"]
            text_color = QtGui.QColor(scatter_color)

            open_text = pg.TextItem(f"[{volume}]", color=text_color, anchor=(0.5, 0.5))
            open_text.setPos(open_ix, open_y - y_adjustment * 3)
            self.items.append(open_text)
            candle_plot.addItem(open_text)

            close_text = pg.TextItem(f"[{volume}]", color=text_color, anchor=(0.5, 0.5))
            close_text.setPos(close_ix, close_y + y_adjustment * 3 if pair["direction"] == Direction.LONG else close_y - y_adjustment * 3)
            self.items.append(close_text)
            candle_plot.addItem(close_text)

        # 创建散点图
        if scatter_data:
            scatter = pg.ScatterPlotItem(scatter_data)
            self.items.append(scatter)
            candle_plot.addItem(scatter)

    def clear(self) -> None:
        """清除数据"""
        self.clear_data()

    def clear_data(self) -> None:
        """清除数据（兼容原版接口）"""
        self.dt_ix_map.clear()
        self.ix_bar_map.clear()

        candle_plot = self.chart.get_plot("candle")
        for item in self.items:
            candle_plot.removeItem(item)
        self.items.clear()

        self._updated = False

    def add_ma_indicator(self) -> None:
        """添加均线指标"""
        if not self._updated:
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "请先加载K线数据"
            )
            return

        # 弹出配置对话框
        dialog = MAConfigDialog(self)
        if not dialog.exec():
            return

        # 获取参数
        params = dialog.get_params()
        period = params["period"]
        color = params["color"]

        try:
            # 添加 MAItem 到图表
            self.chart.add_item(
                MAItem,
                f"ma_{period}",
                "candle"
            )

            # 获取刚添加的 item 并设置参数
            ma_item = self.chart._items.get(f"ma_{period}")
            if ma_item:
                ma_item.period = period
                ma_item.color = color
                ma_item._pen = pg.mkPen(color=color, width=2)

                # 获取历史数据并更新
                bars = self.chart._manager.get_all_bars()
                ma_item.update_history(bars)

                # 添加到图例
                legend = self._get_or_create_ma_legend()
                sample_line = pg.PlotDataItem(pen=pg.mkPen(color=color, width=2))
                legend.addItem(sample_line, f"MA{period}")

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

    def _get_or_create_ma_legend(self) -> pg.LegendItem:
        """获取或创建 MA 图例"""
        if self.ma_legend:
            return self.ma_legend

        # 创建图例
        legend = pg.LegendItem(offset=(180, 10))
        legend.setParentItem(self.chart._first_plot.vb)
        legend.setBrush(pg.mkBrush(color=(0, 0, 0, 100)))
        legend.setPen(pg.mkPen(color=(200, 200, 200), width=1))

        self.ma_legend = legend
        return legend
