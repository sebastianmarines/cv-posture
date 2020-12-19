from math import degrees, acos
import numpy as np
from numpy.linalg import norm


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> int:
    """
    Calculate the angle between 2 vectors

    :param v1: A numpy array
    :param v2: A numpy array
    :return: The angle between the vectors
    """
    cos_a = np.sum(v1 * v2) / (norm(v1) * norm(v2))
    angle = degrees(acos(cos_a))
    return round(angle)
