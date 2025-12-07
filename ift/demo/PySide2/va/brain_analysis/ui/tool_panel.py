from PySide2.QtWidgets import *
from PySide2.QtCore import *

from widgets.slider import Slider


class ToolPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setFixedWidth(280)
        # self.setStyleSheet("background-color: #f9ffc5")


        self.proj_cpanel = ProjectionCPanel(parent=self)

        layout = QVBoxLayout()
        layout.addWidget(self.proj_cpanel)
        layout.addStretch(1)
        self.setLayout(layout)



class ProjectionCPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setTitle("t-SNE parameters")

        self.perp_slider = Slider(self, Qt.Horizontal, 40)
        self.perp_slider.setTickInterval(10)
        self.perp_slider.setMinimum(0)
        self.perp_slider.setMaximum(100)
        self.perp_slider.setSingleStep(5)

        self.perp_spinbox = QSpinBox()
        self.perp_spinbox.setMinimum(0)
        self.perp_spinbox.setMaximum(100)
        self.perp_spinbox.setValue(self.perp_slider.value())

        self.maxiter_slider = Slider(self, Qt.Horizontal, 1000)
        self.maxiter_slider.setTickInterval(200)
        self.maxiter_slider.setMinimum(0)
        self.maxiter_slider.setMaximum(2000)

        self.maxiter_spinbox = QSpinBox()
        self.maxiter_spinbox.setMinimum(0)
        self.maxiter_spinbox.setMaximum(2000)
        self.maxiter_spinbox.setValue(self.maxiter_slider.value())

        self.proj_btn = QPushButton("Project")

        layout = QGridLayout()
        layout.addWidget(QLabel("Perplexity"), 0, 0)
        layout.addWidget(self.perp_slider, 0, 1)
        layout.addWidget(self.perp_spinbox, 0, 2)
        layout.addWidget(QLabel("Max. Iter."), 1, 0)
        layout.addWidget(self.maxiter_slider, 1, 1)
        layout.addWidget(self.maxiter_spinbox, 1, 2)
        layout.addWidget(self.proj_btn, 2, 0, -1, -1)
        layout.setAlignment(self.proj_btn, Qt.AlignCenter)

        self.setLayout(layout)

        self.perp_slider.valueChanged.connect(lambda:
                                              self.perp_spinbox.setValue(
                                                  self.perp_slider.value()))
        self.perp_spinbox.valueChanged.connect(lambda:
                                              self.perp_slider.setValue(
                                                  self.perp_spinbox.value()))

        self.maxiter_slider.valueChanged.connect(lambda:
                                              self.maxiter_spinbox.setValue(
                                                  self.maxiter_slider.value()))
        self.maxiter_spinbox.valueChanged.connect(lambda:
                                               self.maxiter_slider.setValue(
                                                   self.maxiter_spinbox.value()))