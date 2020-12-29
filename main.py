import sys

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QImage, QPixmap

from threads import MPThread, DataSave
from ui.MainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(self.size())
        self.data: tuple = ()
        self.button_state = True
        self.start_mediapipe()
        self.ui.send_data.clicked.connect(self.send_data)

    def start_mediapipe(self) -> None:
        self.mp_thread = QThread()
        # Step 3: Create a mpworker object
        self.mpworker = MPThread()
        # Step 4: Move mpworker to the thread
        self.mpworker.moveToThread(self.mp_thread)
        # Step 5: Connect signals and slots
        # noinspection PyUnresolvedReferences
        self.mp_thread.started.connect(self.mpworker.run)
        self.mpworker.new_frame.connect(self.show_image)
        self.mpworker.data.connect(self.store_data)
        # Step 6: Start the thread
        self.mp_thread.start()

    def send_data(self) -> None:
        self.data_thread = QThread()
        self.data_worker = DataSave(self.data)
        self.data_worker.moveToThread(self.data_thread)
        self.data_thread.started.connect(self.data_worker.run)
        self.data_worker.message.connect(self.print_msg)
        self.data_worker.finished.connect(self.toggle_button)
        self.data_thread.start()

    @QtCore.pyqtSlot()
    def toggle_button(self):
        self.data_thread.quit()
        self.data_thread.wait()
        self.ui.send_data.setEnabled(self.button_state)
        self.button_state = not self.button_state

    @QtCore.pyqtSlot(str)
    def print_msg(self, message: str) -> None:
        print(message)

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
