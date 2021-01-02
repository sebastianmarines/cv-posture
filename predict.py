from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from numpy.linalg import norm

from utils import normalized_to_pixel_coordinates, angle_between_vectors

FACE_POINTS = (
    10,  # Forehead
    152,  # Chin
    93,  # Right
    323  # Left
)

POSE_POINTS = (
    11,  # Shoulders
    12,
    13,  # Elbows
    14,
    15,  # Wrists
    16
)


def get_points(target_img: np.ndarray, model: Holistic) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes an RGB image and returns the face landmarks on each detected face.

    :param model: mp holistic model
    :param target_img: An RGB image represented as a numpy array.
    :return: A tuple containing: status, 468 landmark points, 4 keypoints, the keypoints
    coordinates in pixels, and pose landmarks.
    """
    target_img = np.copy(target_img)
    _image_rows, _image_cols, _ = target_img.shape
    result = model.process(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

    _keypoints = np.zeros((4, 3))
    _keypoints_abs_coordinates = np.zeros((4, 2))
    _status = False
    _points = np.zeros((0,))
    _pose_landmarks = np.zeros((6, 3))

    if result.face_landmarks and result.pose_landmarks:
        _status = True
        _points = np.array([[point.x, point.y, point.z] for point in result.face_landmarks.landmark])

        for _index, point in enumerate(FACE_POINTS):
            _keypoints[_index] = _points[point]
            _landmark_px = normalized_to_pixel_coordinates(
                _keypoints[_index, 0],
                _keypoints[_index, 1],
                _image_cols,
                _image_rows
            )
            _keypoints_abs_coordinates[_index] = _landmark_px

        for _index, point in enumerate(POSE_POINTS):
            point = result.pose_landmarks.landmark[point]
            _pose_landmarks[_index] = np.array([point.x, point.y, point.z])

    return _status, _points, _keypoints, _keypoints_abs_coordinates, _pose_landmarks


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

    :param face_points: A numpy array containing 4 pixel coordinates of keypoints
    :return: Area in pixels
    """
    forehead, chin, right, left = face_points
    dp = norm(chin - forehead)
    ds = norm(left - right)

    if not np.isnan(dp) and not np.isnan(ds):
        return round((dp * ds) / 2)
    return 0


def face_distance(target_img: np.ndarray, face_coordinates_px: np.ndarray) -> Tuple:
    """
    Calculate the face distance from (0, 0).

    :param target_img: An RGB image represented as a numpy array.
    :param face_coordinates_px: Face coordinates in pixels represented as a numpy array.
    :return: A tuple containing the X and Y values of the distance.
    """

    _image_rows, _image_cols, _ = target_img.shape
    forehead, _, right, _ = face_coordinates_px

    height_delta = norm(forehead[1])
    width_delta = norm(forehead[0])

    return width_delta if not np.isnan(width_delta) else 0, height_delta if not np.isnan(height_delta) else 0
