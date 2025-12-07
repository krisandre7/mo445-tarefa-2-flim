import sys
from PySide2.QtWidgets import QApplication

from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    UI = MainWindow()
    UI.show()
    sys.exit(app.exec_())
