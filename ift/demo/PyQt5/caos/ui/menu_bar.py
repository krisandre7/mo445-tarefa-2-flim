import os
import pdb

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class FileDialogControl():
    first_dialog_main_image = True


class FileMenu(QMenu):
    '''
    A window is a widget that isn't visually the child of any other widget and
    that usually has a frame and a window title.
    A window can have a parent widget. It will then be grouped with its parent
    and deleted when the parent is deleted, minimized when the parent is
    minimized etc.
    If supported by the window manager, it will also have a common taskbar
    entry with its parent.

    QMenu is a window by default, even if a parent widget is specified in the
    constructor.

    In our case, the FileMenu (which is QMenu) has the a QMenuBar as parent.
    self.parent() gives us the QMenuBar object.
    We would expect that self.window() returns the MainWindow object, which is
    parent of the MenuBar, but it returns the own self, since QMenu is a window.
    '''

    def __init__(self, menu_bar):
        super().__init__(menu_bar)
        self.setTitle("&File")

        self.open_act = QAction("&Open Main Image", menu_bar)
        # QKeySequence.Open gets the default open shortcut of the current OS
        self.open_act.setShortcut(QKeySequence.Open)
        self.open_act.triggered.connect(self.open_main_image)
        self.addAction(self.open_act)

        self.addSeparator()

        '''
        When using PyQt on a Mac Os, the system will intercept certain
        commands contain the word 'Quit', 'Exit', 'Setting', 'Settings',
        'preferences' and probably a slew of others, and remove them from your
        menubar because they are reserved labels.
        If a menubar header has no items, it will not display, making it
        appear as if you haven't modified the menubar.
        '''
        exit_act = self.addAction("&Close Caos",
                                       QApplication.instance().quit,
                                       QKeySequence.Close)

    @pyqtSlot()
    def open_main_image(self):
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
        For that, we created the class FileDialogControl which has the static
        attribute first_dialog_main_image which is declared as True by the
        class.
        After the first file dialog, the directory used by the QFileDialog
        will be the directory of the last file picked.
        '''
        if FileDialogControl.first_dialog_main_image:
            # Returns (a list) with the default Pictures dir from the current OS
            pictures_dir_list = QStandardPaths.standardLocations(
                QStandardPaths.PicturesLocation)
            if pictures_dir_list:
                finding_dir = pictures_dir_list[0]
            else:
                finding_dir = QDir.currentPath()

            file_dialog.setDirectory(finding_dir)
            # setting the static attribute of the class FileDialogControl
            FileDialogControl.first_dialog_main_image = False

        file_dialog.setNameFilter("Images (*.png *.jpeg *.jpg *.ppm *.pgm)")

        # execute/open the File Dialog. It returns the action executed by the
        #  User. If the button OK was pressed
        if (file_dialog.exec() == QDialog.Accepted):
            main_image_path = file_dialog.selectedFiles()[0]
            if not os.path.isdir(main_image_path):
                menu_bar = self.parent()
                main_window = menu_bar.window()

                main_window.image_panel.load_main_image(main_image_path)
                menu_bar.view_menu.update_actions()


class ViewMenu(QMenu):
    def __init__(self, menu_bar):
        super().__init__(menu_bar)
        main_window = menu_bar.window()

        self.setTitle("&View")
        self.zoom_in_act = self.addAction("&Zoom In (25%)",
                                          self.zoom_in,
                                          QKeySequence.ZoomIn)
        self.zoom_out_act = self.addAction("&Zoom Out (25%)",
                                           self.zoom_out,
                                           QKeySequence.ZoomOut)
        self.normal_size_act = self.addAction("&Normal Size",
                                              main_window.image_panel.normal_size,
                                              "Ctrl+Shift+K")
        self.normal_size_act.setEnabled(False)

        self.addSeparator()

        self.fit_to_window_act = self.addAction("&Fit to Window",
                                                main_window.image_panel.fit_to_window,
                                                "Ctrl+Shift+J")
        self.fit_to_window_act.setEnabled(False)



    @pyqtSlot()
    def zoom_in(self):
        main_window = self.parent().window()
        main_window.image_panel.regular_scale(1.25)
        self.update_actions()

    @pyqtSlot()
    def zoom_out(self):
        main_window = self.parent().window()
        main_window.image_panel.regular_scale(0.75)
        self.update_actions()

    def update_actions(self):
        main_window = self.parent().window()
        abs_scale_factor = main_window.image_panel.abs_scale_factor

        # switch (enable/disable) the action manipulation
        self.normal_size_act.setEnabled(not self.normal_size_act.isChecked())
        self.fit_to_window_act.setEnabled(not self.fit_to_window_act.isChecked())
        self.zoom_in_act.setEnabled(abs_scale_factor < 10.0)
        self.zoom_out_act.setEnabled(abs_scale_factor > 0.05)


class MenuBar(QMenuBar):
    def __init__(self, main_window):
        # the parent is the main_window
        super().__init__(main_window)

        self.file_menu = FileMenu(self)
        self.view_menu = ViewMenu(self)

        self.addMenu(self.file_menu)
        self.addMenu(self.view_menu)


    def update_actions(self):
        self.view_menu.update_actions()