from PySide2.QtWidgets import *

from ui.tool_panel import ToolPanel
from ui.viz_panel import VizPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.tool_panel = ToolPanel(self)
        self.viz_panel = VizPanel(self)

        self.setWindowTitle("VA")
        self.setMinimumSize(900, 700)

        self.setCentralWidget(QWidget())
        self.centralWidget().setContentsMargins(0, 0, 0, 0)

        main_layout = QGridLayout(self.centralWidget())
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)  # space between the widgets of the layout
        main_layout.addWidget(self.tool_panel, 0, 0)
        main_layout.addWidget(self.viz_panel, 0, 1)
