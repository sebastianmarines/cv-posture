import sys

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QImage, QPixmap

from threads import MPThread, DataSave, Counter
from ui.MainWindow import Ui_MainWindow
from utils import resource_path


class MainWindow(QtWidgets.QMainWindow):
    poses = (
        ('./posture-images/encorvado.jpg', 'Encorvado'),
        ('./posture-images/encorvado-codo.jpg', 'Recargado en codo'),
        ('./posture-images/sumido.png', 'Sumido')
    )

    def __init__(self):
        super().__init__()

        self.data: tuple = ()
        self.button_state = True
        self.thread_finished = False
        self.poses_index = 0

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(self.size())

        self.ui.current_image.setPixmap(
            QtGui.QPixmap(
                resource_path("posture-images/encorvado.jpg")
            )
        )

        self.ui.send_data.clicked.connect(self.send_data)
        self.start_mediapipe()
        self.counter()

    # Threads

    def start_mediapipe(self) -> None:
        self.mp_thread = QThread(parent=self)
        # Step 3: Create a mpworker object
        self.mpworker = MPThread()
        # Step 4: Move mpworker to the thread
        self.mpworker.moveToThread(self.mp_thread)
        # Step 5: Connect signals and slots
        # noinspection PyUnresolvedReferences
        self.mp_thread.started.connect(self.mpworker.run)
        self.mpworker.new_frame.connect(self.show_image)
        self.mpworker.finished.connect(self.finish_thread)
        self.mpworker.data.connect(self.store_data)
        # Step 6: Start the thread
        self.mp_thread.start()

    def send_data(self) -> None:
        self.data_thread = QThread(parent=self)
        self.data_worker = DataSave(self.data, extras=[])
        self.data_worker.moveToThread(self.data_thread)
        # noinspection PyUnresolvedReferences
        self.data_thread.started.connect(self.data_worker.run)
        self.data_worker.message.connect(self.print_msg)
        self.data_worker.finished.connect(self.toggle_button)
        self.data_thread.start()

    def counter(self, time: int = 5):
        self.counter_thread = QThread(parent=self)
        self.counter_worker = Counter(time=time)
        self.counter_worker.moveToThread(self.counter_thread)
        # noinspection PyUnresolvedReferences
        self.counter_thread.started.connect(self.counter_worker.run)
        self.counter_worker.second.connect(self.update_counter)
        self.counter_worker.finished.connect(self.send_data)
        self.counter_thread.start()

    # Signal slots

    @QtCore.pyqtSlot(int)
    def update_counter(self, time):
        self.ui.contador.setText(str(time))

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

    @QtCore.pyqtSlot()
    def finish_thread(self):
        self.thread_finished = True


app = QtWidgets.QApplication([])

application = MainWindow()

application.show()

sys.exit(app.exec())
