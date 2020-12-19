from __future__ import annotations

import time
from math import acos, degrees
from typing import Tuple

import pretty_errors
import cv2
import mediapipe as mp
import numpy as np
from numpy.linalg import norm

from utils import normalized_to_pixel_coordinates

POINTS = (
    10,  # Forehead
    152,  # Chin
    93,  # Right
    323  # Left
)


def get_points(target_img: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes an RGB image and returns the face landmarks on each detected face.

    :param target_img: An RGB image represented as a numpy array.
    :return: A tuple containing: status, 468 landmark points, 4 keypoints, and the keypoints
    coordinates in pixels.
    """
    _image_rows, _image_cols, _ = image.shape
    result = face_mesh.process(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

    _points = np.zeros((468, 3))
    _keypoints = np.zeros((4, 3))
    _keypoints_abs_coordinates = np.zeros((4, 2))
    _status = False

    if result.multi_face_landmarks:
        _status = True
        for index, point in enumerate(result.multi_face_landmarks[0].landmark):
            _point = np.array([point.x, point.y, point.z])
            _points[index] = _point

        for index, point in enumerate(POINTS):
            _keypoints[index] = _points[point]
            _landmark_px = normalized_to_pixel_coordinates(
                _keypoints[index, 0],
                _keypoints[index, 1],
                image_cols,
                image_rows
            )
            _keypoints_abs_coordinates[index] = _landmark_px

    return _status, _points, _keypoints, _keypoints_abs_coordinates


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> int:
    cos_a = np.sum(v1 * v2) / (norm(v1) * norm(v2))
    angle = degrees(acos(cos_a))
    return round(angle)


def get_roll(face_points: np.ndarray) -> int:
    forehead, chin, *_ = face_points
    face_vector = chin - forehead
    ref_point = np.array([0, forehead[1], 0])
    ref_vector = ref_point - forehead

    return angle_between_vectors(face_vector, ref_vector)


def get_yaw(face_points: np.ndarray) -> int:
    *_, right_ear, left_ear = face_points
    face_vector = left_ear - right_ear
    ref_point = np.array([right_ear[0], 0, 0])
    ref_vector = ref_point - right_ear

    return angle_between_vectors(face_vector, ref_vector)


def get_pitch(face_points: np.ndarray) -> int:
    forehead, chin, *_ = face_points
    face_vector = chin - forehead
    ref_point = np.array([0, forehead[1], forehead[2] + 5])  # arbitrary number just to make a vector
    ref_vector = ref_point - forehead

    return angle_between_vectors(face_vector, ref_vector)


def face_area(face_points: np.ndarray) -> int:
    """
    Calculate the area as if it was a diamond

    :param face_points: A numpy array containing 4 keypoints
    :return: Area in pixels
    """
    forehead, chin, right, left = face_points
    dp = norm(chin - forehead)
    ds = norm(left - right)

    if not np.isnan(dp) and not np.isnan(ds):
        return round((dp * ds) / 2)
    return 0


if __name__ == "__main__":
    pretty_errors.replace_stderr()
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(1)

    while cap.isOpened():

        start_time = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue

        image.flags.writeable = False
        status, face, keypoints, keypoints_coords = get_points(image)

        if status is not False:
            roll = get_roll(keypoints)
            yaw = get_yaw(keypoints)
            pitch = get_pitch(keypoints)
            area = face_area(keypoints_coords)
            # Roll
            roll_position = (lambda: "Hombro derecho" if roll < 90 else "Hombro izquierdo" if roll > 90 else "Derecho")
            cv2.putText(
                image,
                f"Roll: {roll}",
                (10, 200),
                cv2.FONT_ITALIC,
                0.7,
                (255, 0, 0),
                2
            )
            cv2.putText(
                image,
                roll_position(),
                (20, 225),
                cv2.FONT_ITALIC,
                0.7,
                (0, 0, 255),
                2
            )
            # Yaw
            yaw_position = (lambda: "Izquierda" if yaw < 90 else "Derecha" if yaw > 90 else "Frente")
            cv2.putText(
                image,
                f"Yaw: {yaw}",
                (10, 250),
                cv2.FONT_ITALIC,
                0.7,
                (255, 0, 0),
                2
            )
            cv2.putText(
                image,
                yaw_position(),
                (20, 275),
                cv2.FONT_ITALIC,
                0.7,
                (0, 0, 255),
                2
            )
            # Pitch
            pitch_position = (lambda: "Abajo" if pitch < 90 else "Arriba" if pitch > 90 else "Frente")
            cv2.putText(
                image,
                f"Pitch: {pitch}",
                (10, 300),
                cv2.FONT_ITALIC,
                0.7,
                (255, 0, 0),
                2
            )
            cv2.putText(
                image,
                pitch_position(),
                (20, 325),
                cv2.FONT_ITALIC,
                0.7,
                (0, 0, 255),
                2
            )
            # Area
            cv2.putText(
                image,
                f"Area: {area}",
                (10, 350),
                cv2.FONT_ITALIC,
                0.7,
                (255, 0, 0),
                2
            )

        if status is not False:
            image_rows, image_cols, _ = image.shape
            for face_landmarks in face:
                landmark_px = normalized_to_pixel_coordinates(
                    face_landmarks[0],
                    face_landmarks[1],
                    image_cols,
                    image_rows
                )
                cv2.circle(image, landmark_px, 1,
                           (255, 0, 0), 1)

        fps = f"FPS: {round(1.0 / (time.time() - start_time))}"
        cv2.putText(
            image,
            fps,
            (10, 20),
            cv2.FONT_ITALIC,
            0.7,
            (50, 205, 50),
            2
        )
        cv2.imshow('MediaPipe FaceMesh', image)
        # time.sleep(0.5)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    face_mesh.close()
    cap.release()
