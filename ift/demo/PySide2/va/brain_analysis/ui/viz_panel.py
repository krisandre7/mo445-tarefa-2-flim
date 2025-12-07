from PySide2.QtWidgets import QFrame
from PySide2.QtGui import QPainter

from PySide2.QtCharts import QtCharts


class VizPanel(QtCharts.QChartView):
    def __init__(self, parent=None):
        super().__init__(QtCharts.QChart(), parent=parent)

        self.setMinimumWidth(700)
        self.setRenderHint(QPainter.Antialiasing)

        self.reset_chart_view()


    def reset_chart_view(self):
        self.chart().removeAllSeries()

        # an empty serie only to draw the axis and grids from the chart
        series = QtCharts.QScatterSeries()
        self.chart().addSeries(series)
        self.chart().createDefaultAxes()  # creates default axes and grid for the serie

        self.chart().legend().setVisible(False)  # hide the legend
        self.chart().removeAllSeries()  # remove the empty serie but keep the axes and grid
