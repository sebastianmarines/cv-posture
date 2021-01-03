import sys

from PyQt5 import QtWidgets, QtGui

from mixins import ThreadsMixin, SlotsMixin
from ui.MainWindow import Ui_MainWindow
from utils import resource_path


class MainWindow(QtWidgets.QMainWindow, SlotsMixin, ThreadsMixin):
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
        self.poses_index = 1
        self.started = False

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(self.size())

        self.setup()

    def setup(self):
        self.start_mediapipe()
        self.ui.counter.setText("--")
        self.ui.start.clicked.connect(self.start)

    def start(self):
        self.ui.start.setEnabled(False)
        self.handle_pose_images()
        self.counter()

    def handle_pose_images(self):
        if self.poses_index > len(self.poses):
            # TODO Handle exit
            return
        _curr_image, _img_description = self.poses[self.poses_index]
        self.ui.current_image.setPixmap(
            QtGui.QPixmap(resource_path(_curr_image))
        )
        self.poses_index += 1


app = QtWidgets.QApplication([])

application = MainWindow()

application.show()

sys.exit(app.exec())
