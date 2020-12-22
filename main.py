import sys

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QImage, QPixmap

from threads import MPThread
from ui.MainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)

        self.setFixedSize(self.size())

        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = MPThread()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        # noinspection PyUnresolvedReferences
        self.thread.started.connect(self.worker.run)
        self.worker.new_frame.connect(self.show_image)
        # Step 6: Start the thread
        self.thread.start()

    @QtCore.pyqtSlot(QImage)
    def show_image(self, image):
        self.ui.webcam.setPixmap(QPixmap.fromImage(image))


app = QtWidgets.QApplication([])

application = MainWindow()

application.show()

sys.exit(app.exec())
