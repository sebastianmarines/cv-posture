import sys

import cv2
from PyQt5 import QtWidgets, QtCore, QtGui

from ui.mainwindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)

        # self.ui.video.s
        # self.ui.label.setPicture()

        self.cap = cv2.VideoCapture(1)

        self.ui.pushButton.clicked.connect(self.show_image)

    @QtCore.pyqtSlot()
    def show_image(self):
        success, image = self.cap.read()
        self.image = image
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0],
                                  QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(self.image))


app = QtWidgets.QApplication([])

application = MainWindow()

application.show()

sys.exit(app.exec())
