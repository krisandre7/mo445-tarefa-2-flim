import os

import pyift.pyift as ift

from PySide2.QtWidgets import QMainWindow, QWidget, QGridLayout, QFrame, QAction, QFileDialog
from PySide2.QtWidgets import QDialog, QMessageBox
from PySide2.QtGui import QKeySequence, QIcon
from PySide2.QtCore import QSize, Slot, QStandardPaths, QDir

from ui.main_panel import ToolPanel, ChartView
# CONVENTION: Google Python Style Guide
# module_name, package_name, ClassName, method_name, ExceptionName,
# function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name,
# function_parameter_name, local_var_name
# camelCase only to conform to pre-existing conventions





class Data():
    def __init__(self):
        self.dataset = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.first_dialog_main_image = True

        self.data = Data()
        self.tool_panel = ToolPanel(self)
        self.chart_view = ChartView(self)
        # self.chart_view.setStyleSheet("background-color:red;")

        self.setCentralWidget(QWidget())
        self.load_ui()
        self.load_menu()

        # connecting signals and slots
        self.tool_panel.project_dataset_signal.connect(self.project_dataset)


    def load_ui(self):
        main_layout = QGridLayout(self.centralWidget())
        main_layout.addWidget(self.tool_panel, 0, 0)
        main_layout.addWidget(self.chart_view, 0, 1)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.setWindowTitle("Caos Visualizer")
        self.setGeometry(150, 45, 1000, 700)
        self.setMinimumSize(1000, 500)
        self.statusBar().showMessage('Ready')

    def load_menu(self):
        '''
        On Mac OS, some names for menu actions/items are reserved, such as:
        'about', 'config', 'options', 'setup', 'setting', 'preferences', 'quit', 'exit
        If anyone is used, its action will be incorporated at the menu with the application's name.
        The text used for the application name in the menu bar is obtained from the value set in the Info.plist file in the application's bundle.
        http://ftp.ics.uci.edu/pub/centos0/ics-custom-build/BUILD/PyQt-x11-gpl-4.7.2/doc/html/qmenubar.html
        '''
        menu_bar = self.menuBar()

        print("dir: %s" % os.getcwd())
        open_dataset_act = QAction(QIcon("ui/icons/folder.png"), 'Open DataSet', menu_bar)
        open_dataset_act.setShortcut(QKeySequence.Open)
        open_dataset_act.triggered.connect(self.open_dataset)

        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(open_dataset_act)

        # menu_bar.addMenu(file_menu)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.setStyleSheet("background-color: white")
        self.toolbar.addAction(open_dataset_act)
        self.toolbar.setIconSize(QSize(32, 32))


    @Slot()
    def open_dataset(self):
        '''
        By default, the QFileDialog open the file picker from the directory
        where the python main program is called.
        After the first dialog file picker, the default becomes the directory
        of the last file uploaded.
        '''
        file_dialog = QFileDialog(self, "Open File")

        '''
        We could define a fixed initial directory for the File Dialog. But here,
        we are going to set the Pictures folder as the directory used during the
        first file dialog.
        For that, we created the attribute first_dialog_main_image which is declared as True.
        After the first file dialog, the directory used by the QFileDialog
        will be the directory of the last file picked.
        '''
        if self.first_dialog_main_image:
            # Returns (a list) with the default DataSet dir from the current OS
            # dataset_dir_list = QStandardPaths.standardLocations(
            #     QStandardPaths.HomeLocation)
            dataset_dir_list = ["/Users/samuel/workspace/exps"]
            if dataset_dir_list:
                finding_dir = dataset_dir_list[0]
            else:
                finding_dir = QDir.currentPath()

            file_dialog.setDirectory(finding_dir)
            self.first_dialog_main_image = False

        filters = ["DataSet (*.zip)", "Any files (*)"]
        file_dialog.setNameFilters(filters)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog)


        # execute/open the File Dialog. It returns the action executed by the
        #  User. If the button OK was pressed
        if (file_dialog.exec() == QDialog.Accepted):
            dataset_path = file_dialog.selectedFiles()[0]

            if not os.path.isdir(dataset_path):
                self.data.dataset = ift.ReadDataSet(dataset_path)

                if self.data.dataset.projection:
                    proj_data = self.data.dataset.GetProjection()
                    true_labels = self.data.dataset.GetTrueLabels()
                    self.chart_view.plot_dataset(proj_data, true_labels)
                    print(proj_data.shape)
                else:
                    self.project_dataset()


    @Slot()
    def project_dataset(self):
        perplexity, max_iters = self.tool_panel.get_tsne_parameters()
        Z = self.data.dataset

        if Z:
            self.chart_view.reset_chart_view()

            print("projecting")
            print(perplexity, max_iters)
            ift.DimReductionByTSNE(Z, 2, perplexity, max_iters)
            proj_data = Z.GetProjection()
            print(proj_data.shape)
            true_labels = Z.GetTrueLabels()
            self.chart_view.plot_dataset(proj_data, true_labels)



















