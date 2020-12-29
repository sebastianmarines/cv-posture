import cv2
import mediapipe as mp
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, QObject

from predict import get_points
from utils import normalized_to_pixel_coordinates, cv2_to_qimage


class MPThread(QObject):
    new_frame = pyqtSignal(QtGui.QImage)
    data = pyqtSignal(tuple)
    debug = True

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
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
                if status and self.debug:
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

                self.data.emit((keypoints, pose_landmarks))
