from PyQt5.QtCore import pyqtSlot, QThread
from PyQt5.QtGui import QImage, QPixmap

from ui.MainWindow import Ui_MainWindow


class SlotsMixin:
    data_thread: QThread
    ui: Ui_MainWindow
    button_state: bool
    data: tuple
    thread_finished: bool

    @pyqtSlot(int)
    def update_counter(self, time):
        self.ui.contador.setText(str(time))

    @pyqtSlot()
    def toggle_button(self):
        self.data_thread.quit()
        self.data_thread.wait()
        self.ui.send_data.setEnabled(self.button_state)
        self.button_state = not self.button_state

    @pyqtSlot(str)
    def print_msg(self, message: str) -> None:
        print(message)

    @pyqtSlot(QImage)
    def show_image(self, image):
        self.ui.webcam.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(tuple)
    def store_data(self, data):
        self.data = data

    @pyqtSlot()
    def finish_thread(self):
        self.thread_finished = True
