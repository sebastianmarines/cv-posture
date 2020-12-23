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
        self.data: tuple = ()
        self.start_mediapipe()
        self.ui.send_data.clicked.connect(self.send_data)

    def start_mediapipe(self) -> None:
        # self.mp_thread = QThread()
        # Step 3: Create a mpworker object
        self.mpworker = MPThread(debug=settings.DEBUG)
        # Step 4: Move mpworker to the thread
        # self.mpworker.moveToThread(self.mp_thread)
        # Step 5: Connect signals and slots
        # noinspection PyUnresolvedReferences
        # self.mp_thread.started.connect(self.mpworker.run)
        self.mpworker.new_frame.connect(self.show_image)
        self.mpworker.data.connect(self.store_data)
        # Step 6: Start the thread
        self.mpworker.start()

    def send_data(self) -> None:
        print(self.data)
        # Step 2: Create a QThread object
        self.data_thread = QThread()
        # Step 3: Create a worker object
        self.data_worker = DataSave(self.data)
        # # Step 4: Move worker to the thread
        # self.data_worker.moveToThread(self.data_thread)
        # # Step 5: Connect signals and slots
        # # noinspection PyUnresolvedReferences
        # self.data_thread.started.connect(self.data_worker.run)
        # # Step 6: Start the thread
        # self.data_thread.start()
        #
        # self.ui.send_data.setEnabled(False)
        # self.data_thread.finished.connect(
        #     lambda: self.ui.send_data.setEnabled(True)
        # )

    @QtCore.pyqtSlot(QImage)
    def show_image(self, image):
        self.ui.webcam.setPixmap(QPixmap.fromImage(image))

    @QtCore.pyqtSlot(tuple)
    def store_data(self, data):
        self.data = data


app = QtWidgets.QApplication([])

application = MainWindow()

application.show()

sys.exit(app.exec())
