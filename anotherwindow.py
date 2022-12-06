from PySide2 import QtGui
from PySide2.QtGui import QPixmap
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QLabel
from PySide2.QtCore import QFile

import tkinter as tk
from  tkinter import filedialog

import shutil

from BackEnd import back_end
from GetImage import get_image

"""
Press 'esc' to save and exit
Press 'r' to restore
Press 'n' to update
Press '1' to choose sure background
Press '2' to choose sure foreground
Press '3' to choose probable background
Press '4' to choose probable foreground

Use the right mouse to select the area using rectangle
Use the left  mouse to select the area using a brush

For the first time, you need to select the area using the right mouse
and it is set for defult as mode '2' --- choosing sure foreground
"""

class LoginGui(object):

    def __init__(self):
        # 对ui文件进行加载
        self.ui = QUiLoader().load('gui.ui')
        # 按键连接
        self.ui.startButton.clicked.connect(self.startButton)
        self.ui.photoButton.clicked.connect(self.photoButton)
        self.ui.saveButton.clicked.connect(self.saveButton)

    def photoButton(self):  # 选择图片
        label_1 = self.ui.label_1
        self.pic = get_image()
        pixmap = QtGui.QPixmap(self.pic)  # 导入本地图片，（）里填图片路径！！！
        label_1.setPixmap(pixmap)  # 设置图片到label
        label_1.setScaledContents(True)  # 图片自适应
        label_1.show()

    def displayLabel(self):  # 展示图片
        label_2 = self.ui.label_2
        print("test for display1")
        pixmap = QtGui.QPixmap(self.result)  # 抠图后的图片路径
        print("test for display2:"+self.result)
        label_2.setPixmap(pixmap)  # 设置图片到label
        label_2.setScaledContents(True)  # 图片自适应
        label_2.show()

    def startButton(self):
        # 抠图程序启动
        self.result = back_end(self.pic)
        if self.result != None:
            self.displayLabel()

    def saveButton(self):
        # 保存功能
        save_to_path = filedialog.askdirectory()
        if save_to_path != '':
            shutil.move(self.result, save_to_path)

if __name__ == '__main__':
    app = QApplication([])
    stats = LoginGui()
    stats.ui.show()
    app.exec_()
