from PySide2 import QtGui, QtWidgets
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog, QMainWindow, QDialog, QLabel, QSpacerItem, QSizePolicy
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from PySide2.QtGui import QIcon
import os, shutil, time
from threading import Thread
from PySide2.QtCore import Qt
from generate_image import *
import winsound
from qt_material import apply_stylesheet
import qt
import anotherwindow


class StartWindow():

    def __init__(self):
        super().__init__()
        qfile_stats = QFile('Display/start.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)
        self.ui.first.clicked.connect(self.f1)
        self.ui.second.clicked.connect(self.f2)

    def f2(self):
        global add
        add = qt.Window()
        qt.Mywindow = add
        Mwindow.ui.close()
        qt.Mywindow.ui.show()

    def f1(self):
        global add
        add = anotherwindow.LoginGui()
        Mwindow.ui.close()
        add.ui.show()

if __name__ == '__main__':
    app = QApplication([])
    add = QMainWindow()
    Mwindow = StartWindow()
    Mwindow.ui.show()
    app.exec_()
