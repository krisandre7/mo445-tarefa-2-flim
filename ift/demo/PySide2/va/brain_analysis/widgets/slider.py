from PySide2.QtWidgets import QSlider
from PySide2.QtGui import QMouseEvent
from PySide2.QtCore import Qt

class Slider(QSlider):
    def __init__(self, parent, orientation=Qt.Horizontal, default_value=0):
        super().__init__(orientation, parent)

        self.default_value = default_value

        self.setFocusPolicy(Qt.StrongFocus)
        self.setTickPosition(QSlider.TicksBothSides)
        if default_value > self.maximum():
            self.setMaximum(default_value)
        self.setValue(self.default_value)

    def mouseDoubleClickEvent(self, e: QMouseEvent):
        self.setValue(self.default_value)