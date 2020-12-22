import sys

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QImage, QPixmap

import settings
from threads import MPThread
from ui.MainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)

        self.setFixedSize(self.size())

        self.thread = QThread()
        # Step 3: Create a mpworker object
        self.mpworker = MPThread(debug=settings.DEBUG)
        # Step 4: Move mpworker to the thread
        self.mpworker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        # noinspection PyUnresolvedReferences
        self.thread.started.connect(self.mpworker.run)
        self.mpworker.new_frame.connect(self.show_image)
        self.mpworker.data.connect(self.store_data)
        # Step 6: Start the thread
        self.thread.start()

    @QtCore.pyqtSlot(QImage)
    def show_image(self, image):
        self.ui.webcam.setPixmap(QPixmap.fromImage(image))

    @QtCore.pyqtSlot(tuple)
    def store_data(self, data):
        print(data)


app = QtWidgets.QApplication([])

application = MainWindow()

application.show()

sys.exit(app.exec())
