import sys
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtGui import QFont
from PySide2.QtWidgets import (QApplication, QWidget, QLineEdit, QFrame, QLabel, QVBoxLayout)


class MyLabel(QLabel):
    def __init__(self, title, parent=None):
        super(MyLabel, self).__init__(title, parent)
        # 接受拖入操作
        self.setAcceptDrops(True)
        # 加边框
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        # 文字居中显示
        self.setAlignment(Qt.AlignCenter)
        # 字体加大显示
        self.setFont(QFont(self.font().family(), 24))

    # 拖动进入事件
    def dragEnterEvent(self, event):
        if (event.mimeData().hasFormat('text/plain')):
            event.accept()
        else:
            event.ignore()

    # 放置事件
    def dropEvent(self, event):
        self.setText(event.mimeData().text())


class DemoDragDropEvent(QWidget):
    def __init__(self, parent=None):
        super(DemoDragDropEvent, self).__init__(parent)

        # 设置窗口标题
        self.setWindowTitle('实战PySide2: 拖放操作演示')
        # 设置窗口大小
        self.resize(400, 160)

        self.initUi()

    def initUi(self):
        mainLayout = QVBoxLayout()

        editBox = QLineEdit('拖动选中的文字')
        editBox.setDragEnabled(True)

        textBox = MyLabel('文字拖放到这里')

        mainLayout.addWidget(editBox)
        mainLayout.addWidget(textBox)
        self.setLayout(mainLayout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DemoDragDropEvent()
    window.show()
    app.exec_()