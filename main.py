import sys

from PyQt5 import QtWidgets, QtGui

from mixins import ThreadsMixin, SlotsMixin
from ui.Alert import Ui_Dialog
from ui.MainWindow import Ui_MainWindow
from utils import resource_path


class MainWindow(QtWidgets.QMainWindow, SlotsMixin, ThreadsMixin):
    poses = (
        ('./posture-images/derecho.jpg', 'Derecho'),
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
        self.started = False
        self.active = False
        self.current_pose = ""

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setFixedSize(self.size())
        self.setWindowIcon(
            QtGui.QIcon(
                resource_path("logo.png")
            )
        )
        self.setWindowTitle("Stance")

        self.dialog = QtWidgets.QDialog()
        self.alert = Ui_Dialog()
        self.alert.setupUi(Dialog=self.dialog)

        self.setup()

    def setup(self):
        self.start_mediapipe()
        self.ui.counter.setText("--")
        self.ui.start.clicked.connect(self.start)

    def start(self):
        self.active = True
        self.ui.start.setEnabled(False)
        self.ui.instructions.setText("Por favor sientate como se muestra en la imagen")
        self.handle_pose_images()

    def handle_pose_images(self):
        if self.poses_index > len(self.poses) - 1:
            self.active = False
            self.next()
            return
        _curr_image, _img_description = self.poses[self.poses_index]
        self.ui.current_image.setPixmap(
            QtGui.QPixmap(resource_path(_curr_image))
        )
        self.current_pose = _img_description
        self.ui.pose.setText(self.current_pose)
        self.poses_index += 1
        self.counter()

    def next(self):
        if self.active:
            self.handle_pose_images()
        else:
            self.ui.counter.setText("--")
            self.dialog.show()


app = QtWidgets.QApplication([])

application = MainWindow()

application.show()

sys.exit(app.exec())
