from PySide2.QtWidgets import QLabel, QSlider, QGroupBox, QFrame, QSpinBox, QPushButton
from PySide2.QtWidgets import QGridLayout, QVBoxLayout
from PySide2.QtCore import Qt, Signal
from PySide2.QtGui import QMouseEvent, QPainter, QColor
from PySide2.QtCharts import QtCharts

import numpy as np

import pyift.pyift as ift


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



class ToolPanel(QFrame):
    # signals
    project_dataset_signal = Signal()

    def __init__(self, parent):
        super().__init__(parent)

        self.setFixedWidth(280)
        # self.setFixedHeight(180)
        self.setObjectName("ToolPanel")
        # self.setStyleSheet("background-color: white")

        self.tsne_gbox = None
        self.tsne_perplexity_slider = None
        self.tsne_perplexity_spinbox = None
        self.tsne_maxiter_slider = None
        self.tsne_maxiter_spinbox = None

        self.reset_tsne_group_box()

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self.tsne_gbox)
        main_layout.addStretch(1)
        self.setLayout(main_layout)


    def reset_tsne_group_box(self):
        self.tsne_gbox = QGroupBox("t-SNE parameters")
        
        self.tsne_perplexity_slider = Slider(self, Qt.Horizontal, 40)
        self.tsne_perplexity_slider.setTickInterval(10)
        self.tsne_perplexity_slider.setMinimum(0)
        self.tsne_perplexity_slider.setMaximum(100)

        self.tsne_perplexity_spinbox = QSpinBox()
        self.tsne_perplexity_spinbox.setMinimum(0)
        self.tsne_perplexity_spinbox.setMaximum(100)
        self.tsne_perplexity_spinbox.setValue(self.tsne_perplexity_slider.value())

        self.tsne_maxiter_slider = Slider(self, Qt.Horizontal, 1000)
        self.tsne_maxiter_slider.setTickInterval(200)
        self.tsne_maxiter_slider.setMinimum(0)
        self.tsne_maxiter_slider.setMaximum(2000)

        self.tsne_maxiter_spinbox = QSpinBox()
        self.tsne_maxiter_spinbox.setMinimum(0)
        self.tsne_maxiter_spinbox.setMaximum(2000)
        self.tsne_maxiter_spinbox.setValue(self.tsne_maxiter_slider.value())

        self.tsne_btn = QPushButton("Project")
        self.tsne_btn.setFixedWidth(100)
        self.tsne_btn.clicked.connect(lambda: self.window().chart_view.chart().axisX().setRange(0, 2))  # this signal is captured in MainWindow
        # self.tsne_btn.clicked.connect(lambda: self.project_dataset_signal.emit())  # this signal is captured in MainWindow
        # self.tsne_btn.clicked.connect(lambda: self.window().projet_dataset())  # another way

        layout = QGridLayout()
        layout.addWidget(QLabel("Perplexity"), 0, 0)
        layout.addWidget(self.tsne_perplexity_slider, 0, 1)
        layout.addWidget(self.tsne_perplexity_spinbox, 0, 2)
        layout.addWidget(QLabel("Max. Iter."), 1, 0)
        layout.addWidget(self.tsne_maxiter_slider, 1, 1)
        layout.addWidget(self.tsne_maxiter_spinbox, 1, 2)
        layout.addWidget(self.tsne_btn, 2, 0, -1, -1)
        layout.setAlignment(self.tsne_btn, Qt.AlignCenter)
        self.tsne_gbox.setLayout(layout)

        self.tsne_perplexity_slider.valueChanged.connect(lambda: self.tsne_perplexity_spinbox.setValue(self.tsne_perplexity_slider.value()))
        self.tsne_maxiter_slider.valueChanged.connect(lambda: self.tsne_maxiter_spinbox.setValue(self.tsne_maxiter_slider.value()))
        # another way to connect signal and slot
        # QObject.connect(self.tsne_perplexity_slider, SIGNAL('valueChanged(int)'), lambda: self.tsne_perplexity_spinbox.setValue(self.tsne_perplexity_slider.value()))


    def get_tsne_parameters(self):
        return self.tsne_perplexity_slider.value(), self.tsne_maxiter_slider.value()


class ChartView(QtCharts.QChartView):
    def __init__(self, parent):
        super().__init__(QtCharts.QChart(), parent)

        self.setRenderHint(QPainter.Antialiasing)
        self.reset_chart_view()
        # self.setRubberBand(QtCharts.QChartView.HorizontalRubberBand)

    def reset_chart_view(self):
        self.chart().removeAllSeries()

        # an empty serie only to draw the axis and grids from the chart
        series = QtCharts.QScatterSeries()
        self.chart().addSeries(series)
        self.chart().createDefaultAxes()  # creates default axes and grid for the serie
        
        # axis_x = QtCharts.QValueAxis()
        # axis_x.setRange(0, 1.0)
        # axis_x.setTickCount(11)
        # axis_x.setLabelFormat("%.2f")
        
        # axis_y = QtCharts.QValueAxis()
        # axis_y.setRange(0, 1.0)
        # axis_y.setTickCount(11)
        # axis_y.setLabelFormat("%.2f")

        # self.chart().setAxisX(axis_x, series)
        # self.chart().setAxisY(axis_y, series)
        self.chart().legend().setVisible(False)  # hide the legend
        self.chart().removeAllSeries()  # remove the empty serie but keep the axes and grid


    def plot_dataset(self, proj_data, true_labels=None):
        print("plot_dataset")
        self.reset_chart_view()

        if true_labels is None:
            # setColor
            series = QtCharts.QScatterSeries()
            series.setColor(QColor(33, 33, 33, 200))
            series.setMarkerSize(10.0)
            series.setBorderColor(Qt.black)

            for sample in proj_data:
                series.append(sample[0], sample[1])
        else:
            unique_true_labels = np.unique(true_labels)

            max_true_label = int(unique_true_labels.max())  # the parameter for CategoricalColorTable must be 'int' and not 'numpy.int'
            cmap = ift.CategoricalColorTable(max_true_label)  # YCbCr Color Map
            cmap = ift.ConvertYCbCrColorTableToRGBColorTable(cmap, 255).AsNumPy()  # RGB Color Map
            cmap = np.insert(cmap, 0, [0, 0, 0], axis=0)  # adding the black color for the truelabel 0

            colors = np.array([list(cmap[truelabel]) for truelabel in true_labels])


        min_x, max_x = proj_data[:, 0].min(), proj_data[:, 0].max()
        min_y, max_y = proj_data[:, 1].min(), proj_data[:, 1].max()

        chart = self.chart()
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.axisX().setRange(min_x - 0.1, max_x + 0.1)
        chart.axisY().setRange(min_y - 0.1, max_y + 0.1)
        # chart.axisX().setRange(-1, 7)
        # self.chart().axisY().setRange(-1, 7)
