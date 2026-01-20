"""
MACD 指标 (Moving Average Convergence Divergence) Item

在独立的副图中绘制 MACD 指标，包括：
- DIF 线（快线）= EMA(12) - EMA(26)
- DEA 线（慢线/信号线）= EMA(DIF, 9)
- MACD 柱状图（Histogram）= 2 * (DIF - DEA)
"""

from typing import Optional
import pyqtgraph as pg

from vnpy.trader.ui import QtCore, QtGui
from vnpy.trader.object import BarData
from vnpy.chart.item import ChartItem
from vnpy.chart.manager import BarManager


class MACDItem(ChartItem):
    """MACD指标Item"""

    def __init__(
        self,
        manager: BarManager,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> None:
        """
        构造函数.

        Args:
            manager: BarManager 实例
            fast_period: 快线周期（默认12）
            slow_period: 慢线周期（默认26）
            signal_period: 信号线周期（默认9）
        """
        super().__init__(manager)

        self.fast_period: int = fast_period
        self.slow_period: int = slow_period
        self.signal_period: int = signal_period

        # 线条颜色和画笔
        self._dif_pen: QtGui.QPen = pg.mkPen(color="yellow", width=1)
        self._dea_pen: QtGui.QPen = pg.mkPen(color="cyan", width=1)
        self._up_brush: QtGui.QBrush = pg.mkBrush(color=(255, 0, 0, 100))  # 红色半透明
        self._down_brush: QtGui.QBrush = pg.mkBrush(color=(0, 255, 0, 100))  # 绿色半透明

        # 缓存计算结果
        self._dif_values: dict[int, Optional[float]] = {}
        self._dea_values: dict[int, Optional[float]] = {}
        self._macd_values: dict[int, Optional[float]] = {}

        # EMA缓存
        self._ema_fast: dict[int, Optional[float]] = {}
        self._ema_slow: dict[int, Optional[float]] = {}
        self._ema_signal: dict[int, Optional[float]] = {}

    def _draw_bar_picture(self, ix: int, bar: BarData) -> QtGui.QPicture:
        """
        绘制单根 K 线对应的 MACD 图形.

        Args:
            ix: K 线索引
            bar: K 线数据

        Returns:
            QPicture 对象
        """
        picture: QtGui.QPicture = QtGui.QPicture()
        painter: QtGui.QPainter = QtGui.QPainter(picture)

        # 计算 MACD 值
        dif, dea, macd = self._calculate_macd(ix)
        self._dif_values[ix] = dif
        self._dea_values[ix] = dea
        self._macd_values[ix] = macd

        # 绘制 MACD 柱状图
        if macd is not None:
            # 设置无描边
            painter.setPen(QtCore.Qt.PenStyle.NoPen)

            if macd > 0:
                painter.setBrush(self._up_brush)
            else:
                painter.setBrush(self._down_brush)

            # 柱状图从0开始
            painter.drawRect(
                QtCore.QRectF(
                    ix - 0.3,  # 左边界
                    0,         # 底部（0轴）
                    0.6,       # 宽度
                    macd       # 高度（可能为负）
                )
            )

        # 绘制 DIF 线（连接前一点）
        if ix > 0 and dif is not None:
            prev_dif = self._dif_values.get(ix - 1)
            if prev_dif is not None:
                painter.setPen(self._dif_pen)
                painter.drawLine(
                    QtCore.QPointF(ix - 1, prev_dif),
                    QtCore.QPointF(ix, dif)
                )

        # 绘制 DEA 线（连接前一点）
        if ix > 0 and dea is not None:
            prev_dea = self._dea_values.get(ix - 1)
            if prev_dea is not None:
                painter.setPen(self._dea_pen)
                painter.drawLine(
                    QtCore.QPointF(ix - 1, prev_dea),
                    QtCore.QPointF(ix, dea)
                )

        painter.end()
        return picture

    def _calculate_ema(
        self,
        ix: int,
        period: int,
        cache: dict[int, Optional[float]]
    ) -> Optional[float]:
        """
        计算 EMA (指数移动平均) - 使用迭代避免递归深度超限.

        Args:
            ix: K 线索引
            period: EMA 周期
            cache: EMA 缓存字典

        Returns:
            EMA 值，如果数据不足则返回 None
        """
        if ix < period - 1:
            return None

        # 如果已缓存，直接返回
        if ix in cache:
            return cache[ix]

        # 找到第一个未缓存的索引（向前查找）
        start_ix = ix
        while start_ix >= period - 1 and start_ix not in cache:
            start_ix -= 1

        # 如果没有找到任何缓存，从第一个有效点开始
        if start_ix < period - 1:
            start_ix = period - 1

        # 从 start_ix 迭代计算到 ix
        multiplier = 2.0 / (period + 1)

        for i in range(start_ix, ix + 1):
            if i in cache:
                continue

            bar = self._manager.get_bar(i)
            if bar is None:
                return None

            close_price = bar.close_price

            # 第一个有效点：使用简单移动平均
            if i == period - 1:
                total = 0.0
                for j in range(i - period + 1, i + 1):
                    b = self._manager.get_bar(j)
                    if b is None:
                        return None
                    total += b.close_price
                ema = total / period
                cache[i] = ema
            else:
                # 后续点：使用 EMA 公式
                prev_ema = cache.get(i - 1)
                if prev_ema is None:
                    return None
                ema = (close_price - prev_ema) * multiplier + prev_ema
                cache[i] = ema

        return cache.get(ix)

    def _calculate_macd(self, ix: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        计算 MACD 指标值 - 使用迭代计算DEA避免缓存依赖.

        Args:
            ix: K 线索引

        Returns:
            (DIF, DEA, MACD) 元组，如果数据不足则相应值为 None
        """
        # 计算 EMA(fast) 和 EMA(slow)
        ema_fast = self._calculate_ema(ix, self.fast_period, self._ema_fast)
        ema_slow = self._calculate_ema(ix, self.slow_period, self._ema_slow)

        if ema_fast is None or ema_slow is None:
            return None, None, None

        # 计算 DIF = EMA(fast) - EMA(slow)
        dif = ema_fast - ema_slow

        # 计算 DEA = EMA(DIF, signal_period)
        # 需要至少 slow_period + signal_period - 1 根K线
        min_bars = self.slow_period + self.signal_period - 1
        if ix < min_bars - 1:
            return dif, None, None

        # 如果已缓存DEA，直接使用
        if ix in self._ema_signal:
            dea = self._ema_signal[ix]
            macd = 2 * (dif - dea)
            return dif, dea, macd

        # 找到第一个未缓存的DEA索引
        start_ix = ix
        while start_ix >= min_bars - 1 and start_ix not in self._ema_signal:
            start_ix -= 1

        # 如果没有找到缓存，从第一个有效点开始
        if start_ix < min_bars - 1:
            start_ix = min_bars - 1

        # 从 start_ix 迭代计算到 ix
        multiplier = 2.0 / (self.signal_period + 1)

        for i in range(start_ix, ix + 1):
            if i in self._ema_signal:
                continue

            # 计算当前索引的DIF
            ema_f = self._calculate_ema(i, self.fast_period, self._ema_fast)
            ema_s = self._calculate_ema(i, self.slow_period, self._ema_slow)
            if ema_f is None or ema_s is None:
                return dif, None, None

            current_dif = ema_f - ema_s

            # 第一个有效点：使用DIF的简单移动平均
            if i == min_bars - 1:
                total_dif = 0.0
                for j in range(i - self.signal_period + 1, i + 1):
                    ema_f2 = self._calculate_ema(j, self.fast_period, self._ema_fast)
                    ema_s2 = self._calculate_ema(j, self.slow_period, self._ema_slow)
                    if ema_f2 is None or ema_s2 is None:
                        return dif, None, None
                    total_dif += (ema_f2 - ema_s2)
                dea = total_dif / self.signal_period
                self._ema_signal[i] = dea
            else:
                # 后续点：使用EMA公式
                prev_dea = self._ema_signal.get(i - 1)
                if prev_dea is None:
                    return dif, None, None
                dea = (current_dif - prev_dea) * multiplier + prev_dea
                self._ema_signal[i] = dea

        # 获取最终的DEA值
        dea = self._ema_signal.get(ix)
        if dea is None:
            return dif, None, None

        # 计算 MACD = 2 * (DIF - DEA)
        macd = 2 * (dif - dea)

        return dif, dea, macd

    def boundingRect(self) -> QtCore.QRectF:
        """获取边界矩形"""
        min_value, max_value = self.get_y_range()
        rect = QtCore.QRectF(
            0,
            min_value,
            len(self._manager.get_all_bars()),
            max_value - min_value
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
        # 确定范围
        all_bars = self._manager.get_all_bars()
        if not all_bars:
            return -1.0, 1.0

        start_ix = min_ix if min_ix is not None else 0
        end_ix = max_ix if max_ix is not None else len(all_bars) - 1

        # 收集可见范围内的所有值
        values = []

        for ix in range(start_ix, end_ix + 1):
            # 如果缓存中没有，主动计算
            if ix not in self._dif_values:
                bar = self._manager.get_bar(ix)
                if bar:
                    dif, dea, macd = self._calculate_macd(ix)
                    self._dif_values[ix] = dif
                    self._dea_values[ix] = dea
                    self._macd_values[ix] = macd

            dif = self._dif_values.get(ix)
            dea = self._dea_values.get(ix)
            macd = self._macd_values.get(ix)

            if dif is not None:
                values.append(dif)
            if dea is not None:
                values.append(dea)
            if macd is not None:
                values.append(macd)

        if not values:
            return -1.0, 1.0

        min_value = min(values)
        max_value = max(values)

        # 确保范围有效且包含0轴
        if min_value > 0:
            min_value = 0
        if max_value < 0:
            max_value = 0

        # 添加边距（至少10%）
        range_val = max_value - min_value
        if range_val < 0.001:
            range_val = 1.0
        margin = range_val * 0.15

        return min_value - margin, max_value + margin

    def get_info_text(self, ix: int) -> str:
        """
        获取光标信息文本.

        Args:
            ix: K 线索引

        Returns:
            信息文本
        """
        dif = self._dif_values.get(ix)
        dea = self._dea_values.get(ix)
        macd = self._macd_values.get(ix)

        parts = []
        if dif is not None:
            parts.append(f"DIF: {dif:.3f}")
        if dea is not None:
            parts.append(f"DEA: {dea:.3f}")
        if macd is not None:
            parts.append(f"MACD: {macd:.3f}")

        return "  ".join(parts) if parts else "MACD: -"

    def update_history(self, history: list[BarData]) -> None:
        """更新历史数据（清空所有缓存）"""
        self._dif_values.clear()
        self._dea_values.clear()
        self._macd_values.clear()
        self._ema_fast.clear()
        self._ema_slow.clear()
        self._ema_signal.clear()
        super().update_history(history)

    def update_bar(self, bar: BarData) -> None:
        """更新单根 K 线（重新计算受影响的值）"""
        ix = self._manager.get_index(bar.datetime)
        if ix is None:
            return

        # 清空当前点及之后的缓存
        keys_to_remove = [k for k in self._dif_values if k >= ix]
        for k in keys_to_remove:
            self._dif_values.pop(k, None)
            self._dea_values.pop(k, None)
            self._macd_values.pop(k, None)
            self._ema_fast.pop(k, None)
            self._ema_slow.pop(k, None)
            self._ema_signal.pop(k, None)

        super().update_bar(bar)
