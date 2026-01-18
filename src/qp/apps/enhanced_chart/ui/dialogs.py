"""配置对话框"""

from vnpy.trader.ui import QtWidgets, QtCore, QtGui


class MAConfigDialog(QtWidgets.QDialog):
    """均线配置对话框"""

    # 颜色映射：中文名称 -> 英文颜色值
    COLOR_MAP = {
        "黄色": "yellow",
        "青色": "cyan",
        "品红": "magenta",
        "绿色": "lime",
        "红色": "red",
        "蓝色": "dodgerblue",
        "橙色": "orange",
        "紫色": "violet",
        "白色": "white",
        "粉色": "hotpink",
        "金色": "gold",
        "深绿": "green",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("均线参数配置")
        self.resize(300, 150)

        # 周期输入
        self.period_spin = QtWidgets.QSpinBox()
        self.period_spin.setMinimum(2)
        self.period_spin.setMaximum(500)
        self.period_spin.setValue(20)

        # 颜色选择（显示中文）
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItems(list(self.COLOR_MAP.keys()))
        self.color_combo.setCurrentText("黄色")  # 默认黄色

        # 确定/取消按钮
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # 布局
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("MA 周期:", self.period_spin)
        form_layout.addRow("线条颜色:", self.color_combo)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def get_params(self) -> dict:
        """获取用户配置的参数"""
        # 获取中文颜色名
        chinese_color = self.color_combo.currentText()
        # 转换为英文颜色值
        english_color = self.COLOR_MAP.get(chinese_color, "yellow")

        return {
            "period": self.period_spin.value(),
            "color": english_color  # 返回英文颜色值
        }
