"""
持仓量指标 (Open Interest) Item

在成交量图表上叠加显示市场持仓量曲线。
持仓量会自动缩放到与成交量相同的 Y 轴范围，便于对比观察。
"""

from typing import Optional
import pyqtgraph as pg

from vnpy.trader.ui import QtCore, QtGui
from vnpy.trader.object import BarData
from vnpy.chart.item import ChartItem
from vnpy.chart.manager import BarManager


class OpenInterestItem(ChartItem):
    """
    持仓量曲线 Item

    显示市场总持仓量的变化曲线，叠加在成交量图表上。
    用于分析市场参与热度和资金流向。

    特性：
    - 持仓量自动缩放到成交量 Y 轴范围
    - 橙色曲线，与红绿成交量柱状图区分
    - 光标悬停显示原始持仓量数值
    """

    def __init__(self, manager: BarManager) -> None:
        """
        构造函数.

        Args:
            manager: BarManager 实例
        """
        super().__init__(manager)

        # 持仓量曲线样式（橙色，与成交量柱状图区分）
        self._pen: QtGui.QPen = pg.mkPen(color="orange", width=2)

        # 缓存原始持仓量值（用于显示）
        self._oi_values: dict[int, Optional[float]] = {}

        # 缩放后的持仓量值（用于绘制）
        self._scaled_oi_values: dict[int, Optional[float]] = {}

        # 缓存范围值（避免重复计算）
        self._cached_oi_range: Optional[tuple[float, float]] = None
        self._cached_vol_range: Optional[tuple[float, float]] = None

    def _draw_bar_picture(self, ix: int, bar: BarData) -> QtGui.QPicture:
        """
        绘制单根 K 线对应的持仓量点（连线由多个点组成）.

        持仓量会缩放到成交量 Y 轴范围内。

        Args:
            ix: K 线索引
            bar: K 线数据

        Returns:
            QPicture 对象
        """
        picture: QtGui.QPicture = QtGui.QPicture()
        painter: QtGui.QPainter = QtGui.QPainter(picture)

        # 计算当前点的缩放值
        scaled_value: Optional[float] = self._calculate_scaled_oi(ix, bar)
        self._scaled_oi_values[ix] = scaled_value

        # 只有当前点和前一点都有值时才绘制连线
        if ix > 0 and scaled_value is not None:
            prev_scaled: Optional[float] = self._scaled_oi_values.get(ix - 1)
            if prev_scaled is not None:
                painter.setPen(self._pen)
                painter.drawLine(
                    QtCore.QPointF(ix - 1, prev_scaled),
                    QtCore.QPointF(ix, scaled_value)
                )

        painter.end()
        return picture

    def _calculate_scaled_oi(self, ix: int, bar: BarData) -> Optional[float]:
        """
        计算缩放后的持仓量值.

        Args:
            ix: K 线索引
            bar: K 线数据

        Returns:
            缩放后的持仓量值
        """
        oi_value = bar.open_interest
        self._oi_values[ix] = oi_value

        if oi_value <= 0:
            return None

        # 获取范围（缓存）
        if self._cached_oi_range is None:
            self._cached_oi_range = self._get_oi_range()
        if self._cached_vol_range is None:
            self._cached_vol_range = self._manager.get_volume_range()

        oi_min, oi_max = self._cached_oi_range
        vol_min, vol_max = self._cached_vol_range

        # 计算缩放值
        if oi_max <= oi_min:
            # 所有OI值相同，画在成交量范围中间
            return (vol_max + vol_min) / 2
        elif vol_max <= vol_min:
            # 成交量范围无效，使用原始值
            return oi_value
        else:
            normalized = (oi_value - oi_min) / (oi_max - oi_min)
            scaled = normalized * (vol_max - vol_min) + vol_min
            return scaled

    def boundingRect(self) -> QtCore.QRectF:
        """获取边界矩形（与成交量一致）"""
        min_volume, max_volume = self._manager.get_volume_range()
        rect: QtCore.QRectF = QtCore.QRectF(
            0,
            min_volume,
            len(self._bar_picutures),
            max_volume - min_volume if max_volume > min_volume else 1
        )
        return rect

    def _get_oi_range(
        self,
        min_ix: Optional[int] = None,
        max_ix: Optional[int] = None
    ) -> tuple[float, float]:
        """
        获取持仓量的范围.

        Args:
            min_ix: 最小索引
            max_ix: 最大索引

        Returns:
            (最小值, 最大值)
        """
        bars = self._manager.get_all_bars()
        if not bars:
            return 0, 1

        # 确定范围
        if min_ix is None:
            min_ix = 0
        if max_ix is None:
            max_ix = len(bars) - 1

        # 计算范围内的持仓量最值
        oi_values = []
        for i in range(min_ix, min(max_ix + 1, len(bars))):
            bar = self._manager.get_bar(i)
            if bar and bar.open_interest > 0:
                oi_values.append(bar.open_interest)

        if not oi_values:
            return 0, 1

        return min(oi_values), max(oi_values)

    def get_y_range(
        self,
        min_ix: Optional[int] = None,
        max_ix: Optional[int] = None
    ) -> tuple[float, float]:
        """
        获取 Y 轴范围.

        注意：由于叠加在成交量图上，需要与成交量共用 Y 轴范围。
        这里返回成交量范围，持仓量会自动缩放。

        Args:
            min_ix: 最小索引
            max_ix: 最大索引

        Returns:
            (最小值, 最大值)
        """
        # 返回成交量范围（与 VolumeItem 保持一致）
        min_volume, max_volume = self._manager.get_volume_range(min_ix, max_ix)
        return min_volume, max_volume

    def get_info_text(self, ix: int) -> str:
        """
        获取光标信息文本.

        Args:
            ix: K 线索引

        Returns:
            信息文本
        """
        bar: Optional[BarData] = self._manager.get_bar(ix)

        if bar and bar.open_interest > 0:
            return f"持仓: {bar.open_interest:.0f}"
        return "持仓: -"

    def update_history(self, history: list[BarData]) -> None:
        """更新历史数据（清空缓存）"""
        self._oi_values.clear()
        self._scaled_oi_values.clear()
        self._cached_oi_range = None
        self._cached_vol_range = None
        super().update_history(history)

    def update_bar(self, bar: BarData) -> None:
        """更新单根 K 线"""
        ix: Optional[int] = self._manager.get_index(bar.datetime)
        if ix is None:
            return

        # 清空当前点及之后的缓存
        keys_to_remove: list[int] = [k for k in self._oi_values if k >= ix]
        for k in keys_to_remove:
            self._oi_values.pop(k, None)
            self._scaled_oi_values.pop(k, None)

        # 清空范围缓存（新数据可能改变范围）
        self._cached_oi_range = None
        self._cached_vol_range = None

        super().update_bar(bar)
