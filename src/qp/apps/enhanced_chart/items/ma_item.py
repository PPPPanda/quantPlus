"""
均线指标 (Moving Average) Item

在 K 线图上绘制移动平均线。
"""

from typing import Optional
import pyqtgraph as pg

from vnpy.trader.ui import QtCore, QtGui
from vnpy.trader.object import BarData
from vnpy.chart.item import ChartItem  # 修复：从 vnpy.chart.item 导入
from vnpy.chart.manager import BarManager


class MAItem(ChartItem):
    """均线指标Item"""

    def __init__(
        self,
        manager: BarManager,
        period: int = 20,
        color: str = "yellow"
    ) -> None:
        """
        构造函数.

        Args:
            manager: BarManager 实例
            period: MA 周期
            color: 线条颜色
        """
        super().__init__(manager)

        self.period: int = period
        self.color: str = color
        self._pen: QtGui.QPen = pg.mkPen(color=color, width=2)

        # 缓存 MA 值
        self._ma_values: dict[int, Optional[float]] = {}

    def _draw_bar_picture(self, ix: int, bar: BarData) -> QtGui.QPicture:
        """
        绘制单根 K 线对应的 MA 点（连线由多个点组成）.

        Args:
            ix: K 线索引
            bar: K 线数据

        Returns:
            QPicture 对象
        """
        picture: QtGui.QPicture = QtGui.QPicture()
        painter: QtGui.QPainter = QtGui.QPainter(picture)

        # 计算当前点的 MA 值
        ma_value: Optional[float] = self._calculate_ma(ix)
        self._ma_values[ix] = ma_value

        # 只有当前点和前一点都有值时才绘制连线
        if ix > 0 and ma_value is not None:
            prev_ma: Optional[float] = self._ma_values.get(ix - 1)
            if prev_ma is not None:
                painter.setPen(self._pen)
                painter.drawLine(
                    QtCore.QPointF(ix - 1, prev_ma),
                    QtCore.QPointF(ix, ma_value)
                )

        painter.end()
        return picture

    def _calculate_ma(self, ix: int) -> Optional[float]:
        """
        计算指定索引处的 MA 值.

        Args:
            ix: K 线索引

        Returns:
            MA 值，如果数据不足则返回 None
        """
        # 需要至少 period 根 K 线
        if ix < self.period - 1:
            return None

        # 获取最近 period 根 K 线的收盘价
        total_close: float = 0.0
        for i in range(ix - self.period + 1, ix + 1):
            bar: Optional[BarData] = self._manager.get_bar(i)
            if bar is None:
                return None
            total_close += bar.close_price

        return total_close / self.period

    def boundingRect(self) -> QtCore.QRectF:
        """获取边界矩形（与蜡烛图一致）"""
        min_price, max_price = self._manager.get_price_range()
        rect: QtCore.QRectF = QtCore.QRectF(
            0,
            min_price,
            len(self._manager.get_all_bars()),
            max_price - min_price
        )
        return rect

    def get_y_range(
        self,
        min_ix: Optional[int] = None,
        max_ix: Optional[int] = None
    ) -> tuple[float, float]:
        """
        获取 Y 轴范围.

        Args:
            min_ix: 最小索引
            max_ix: 最大索引

        Returns:
            (最小值, 最大值)
        """
        # 使用蜡烛图的价格范围
        return self._manager.get_price_range()

    def get_info_text(self, ix: int) -> str:
        """
        获取光标信息文本.

        Args:
            ix: K 线索引

        Returns:
            信息文本
        """
        ma_value: Optional[float] = self._ma_values.get(ix)
        if ma_value is not None:
            return f"MA{self.period}: {ma_value:.2f}"
        return f"MA{self.period}: -"

    def update_history(self, history: list[BarData]) -> None:
        """更新历史数据（清空缓存）"""
        self._ma_values.clear()
        super().update_history(history)

    def update_bar(self, bar: BarData) -> None:
        """更新单根 K 线（重新计算受影响的MA值）"""
        ix: Optional[int] = self._manager.get_index(bar.datetime)
        if ix is None:
            return

        # 清空当前点及之后的缓存（因为MA是滚动计算的）
        keys_to_remove: list[int] = [k for k in self._ma_values if k >= ix]
        for k in keys_to_remove:
            self._ma_values.pop(k, None)

        super().update_bar(bar)
