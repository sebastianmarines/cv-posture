import sys

import cv2
import mediapipe as mp
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from predict import get_points
from ui.MainWindow import Ui_MainWindow
from utils import normalized_to_pixel_coordinates
from utils.images import cv2_to_qimage

DEBUG = True


class Worker(QThread):
    new_frame = pyqtSignal(QtGui.QImage)

    def run(self) -> None:
        cap = cv2.VideoCapture(1)
        mp_face_mesh = mp.solutions.holistic
        holistic = mp_face_mesh.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            upper_body_only=True
        )
        while True:
            success, frame = cap.read()
            if success:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                status, face_landmarks, keypoints, keypoints_coords, pose_landmarks = get_points(rgb_image,
                                                                                                 model=holistic)
                if DEBUG and status:
                    image_rows, image_cols, _ = rgb_image.shape
                    for landmark in face_landmarks:
                        landmark_px = normalized_to_pixel_coordinates(
                            landmark[0],
                            landmark[1],
                            image_cols,
                            image_rows
                        )
                        cv2.circle(rgb_image, landmark_px, 1,
                                   (255, 0, 0), 1)

                    for index, landmark in enumerate(pose_landmarks):
                        landmark_px = normalized_to_pixel_coordinates(
                            landmark[0],
                            landmark[1],
                            image_cols,
                            image_rows
                        )
                        cv2.circle(rgb_image, landmark_px, 1,
                                   (0, 0, 255), 5)

                image = cv2_to_qimage(rgb_image)
                self.new_frame.emit(image)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)

        self.setFixedSize(self.size())

        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker()
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
