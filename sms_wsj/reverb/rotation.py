"""Contains functions regarding 3D rotation matrices. Used in `scenario.py`."""
import numpy as np


def rot_x(alpha):
    """Returns rotation matrix."""
    return np.asarray(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ]
    )


def rot_y(alpha):
    """Returns rotation matrix."""
    return np.asarray(
        [
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ]
    )


def rot_z(alpha):
    """Returns rotation matrix."""
    return np.asarray(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ]
    )
