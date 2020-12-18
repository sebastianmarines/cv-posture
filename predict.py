from __future__ import annotations

import time
from math import sqrt, acos, degrees
from typing import List, Union, NamedTuple

import cv2
import mediapipe as mp

_mp_face_mesh = mp.solutions.face_mesh

points = (
    10,  # Frente
    152,  # Barbilla
    # 1,  # Nariz
    93,  # Derecha
    323  # Izquierda
)


class Point(NamedTuple):
    x: float
    y: float
    z: float

    def to_mathematica_vector_representation(self, max_decimals=10) -> str:
        return f"{{{round(self.x, max_decimals)}, {round(self.y, max_decimals)}, {round(self.z, max_decimals)}}}"


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_two_points(cls, p1: Point, p2: Point) -> Vector:
        return cls(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)

    def magnitude(self) -> float:
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __mul__(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


def get_points(target_img, show_all: bool = False) -> Union[bool, List[Point]]:
    result = _face_mesh.process(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return False
    _points: List[Point] = []
    if show_all:
        for landmark in result.multi_face_landmarks[0].landmark:
            _point = Point(landmark.x, landmark.y, landmark.z)
            _points.append(_point)
    else:
        for point in points:
            landmark = result.multi_face_landmarks[0].landmark[point]
            _point = Point(landmark.x, landmark.y, landmark.z)
            _points.append(_point)

    return _points


def list_to_vector(_points: List[Point]) -> str:
    _vector = "{"
    for _point in _points:
        _vector += _point.to_mathematica_vector_representation() + ","
    _vector = _vector.rstrip(",")
    _vector += "}"
    return _vector


def get_roll(_points: List[Point]) -> float:
    forehead, chin, *_ = _points
    # Calcular vector de la frente a la barbilla
    face_vector = Vector.from_two_points(chin, forehead)
    # Calcular vector de referencia
    ref_point = Point(0, forehead.y, 0)
    ref_vector = Vector.from_two_points(ref_point, forehead)

    cos_a = (face_vector * ref_vector) / (face_vector.magnitude() * ref_vector.magnitude())
    angle = degrees(acos(cos_a))
    return angle


def get_yaw(_points: List[Point]) -> float:
    *_, right_ear, left_ear = _points
    face_vector = Vector.from_two_points(left_ear, right_ear)
    ref_point = Point(right_ear.x, 0, 0)
    ref_vector = Vector.from_two_points(ref_point, right_ear)

    cos_a = (face_vector * ref_vector) / (face_vector.magnitude() * ref_vector.magnitude())
    angle = degrees(acos(cos_a))
    return angle


def get_pitch(_points: List[Point]) -> float:
    forehead, chin, *_ = _points
    face_vector = Vector.from_two_points(chin, forehead)
    ref_point = Point(0, forehead.y, forehead.z + 5)  # arbitrary number just to make a vector
    ref_vector = Vector.from_two_points(ref_point, forehead)
    cos_a = (face_vector * ref_vector) / (face_vector.magnitude() * ref_vector.magnitude())
    angle = degrees(acos(cos_a))
    return angle


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    _face_mesh = _mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(1)

    ROLL_REFERENCE = 90
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():

        start_time = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = _face_mesh.process(image)
        key_points = get_points(image)
        if key_points:
            roll = round(get_roll(key_points))
            yaw = round(get_yaw(key_points))
            pitch = round(get_pitch(key_points))
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

        if results.multi_face_landmarks:
            image_rows, image_cols, _ = image.shape
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )
                for landmark in face_landmarks.landmark:
                    landmark_px = mp_drawing._normalized_to_pixel_coordinates(
                        landmark.x,
                        landmark.y,
                        image_cols,
                        image_rows
                    )
                    cv2.circle(image, (landmark_px[0]+round(640/480), landmark_px[1]), 1,
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

    _face_mesh.close()
    cap.release()
