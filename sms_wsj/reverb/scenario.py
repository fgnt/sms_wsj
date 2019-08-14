"""
Helps to quickly create source and sensor positions.
Try it with the following code:

>>> import numpy as np
>>> from mpl_toolkits.mplot3d import proj3d
>>> import paderbox.reverb.scenario as scenario
>>> src = scenario.generate_random_source_positions(dims=2, sources=1000)
>>> src[1, :] = np.abs(src[1, :])
>>> mic = scenario.generate_sensor_positions(shape='linear', scale=0.1, number_of_sensors=6)
>>> scenario.simple_plot(room=None, sources=src, sensors=mic)
>>> plt.show()
"""

import numpy as np

from paderbox.visualization import matplotlib_fix

import matplotlib.pyplot as plt
import random
import itertools
from paderbox.visualization.new_cm import viridis_hex
from paderbox.utils.deprecated import deprecated
from paderbox.math.rotation import rot_x, rot_y, rot_z
from paderbox.math.directional import minus as directional_minus

################################################################################
# Register Axes3D as a 'projection' object available for use just like any axes
################################################################################
from mpl_toolkits.mplot3d import Axes3D


def sample_from_random_box(center, edge_lengths, rng=np.random):
    """ Sample from a random box to get somewhat random locations.

    >>> points = np.asarray([sample_from_random_box(
    ...     [[10], [20], [30]], [[1], [2], [3]]
    ... ) for _ in range(1000)])
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> _ = ax.scatter(points[:, 0, 0], points[:, 1, 0], points[:, 2, 0])
    >>> _ = plt.show()

    Args:
        center: Original center (mean).
        edge_lengths: Edge length of the box to be sampled from.

    Returns:

    """
    center = np.asarray(center)
    edge_lengths = np.asarray(edge_lengths)
    return center + rng.uniform(
        low=-edge_lengths / 2,
        high=edge_lengths / 2
    )


def generate_sensor_positions(
        shape='cube',
        center=np.zeros((3, 1), dtype=np.float),
        scale=0.01,
        number_of_sensors=None,
        jitter=None,
        rng=np.random,
        rotate_x=0, rotate_y=0, rotate_z=0
):
    """ Generate different sensor configurations.

    Sensors are index counter-clockwise starting with the 0th sensor below
    the x axis. This is done, such that the first two sensors point towards
    the x axis.

    :param shape: A shape, i.e. 'cube', 'triangle', 'linear' or 'circular'.
    :param center: Numpy array with shape (3, 1)
        which holds coordinates x, y and z.
    :param scale: Scalar responsible for scale of the array. See individual
        implementations, if it is used as radius or edge length.
    :param jitter: Add random Gaussian noise with standard deviation ``jitter``
        to sensor positions.
    :return: Numpy array with shape (3, number_of_sensors).
    """

    center = np.array(center)
    if center.ndim == 1:
        center = center[:, None]

    if shape == 'cube':
        b = scale / 2
        sensor_positions = np.array([
            [-b, -b, -b],
            [-b, -b, b],
            [-b, b, -b],
            [-b, b, b],
            [b, -b, -b],
            [b, -b, b],
            [b, b, -b],
            [b, b, b]
        ]).T

    elif shape == 'triangle':
        assert number_of_sensors == 3, (
            "triangle is only defined for 3 sensors",
            number_of_sensors)
        sensor_positions = generate_sensor_positions(
            shape='circular', scale=scale, number_of_sensors=3, rng=rng
        )

    elif shape == 'linear':
        sensor_positions = np.zeros((3, number_of_sensors), dtype=np.float)
        sensor_positions[1, :] = scale * np.arange(number_of_sensors)
        sensor_positions -= np.mean(sensor_positions, keepdims=True, axis=1)

    elif shape == 'circular':
        if number_of_sensors == 1:
            sensor_positions = np.zeros((3, 1), dtype=np.float)
        else:
            radius = scale
            delta_phi = 2 * np.pi / number_of_sensors
            phi_0 = delta_phi / 2
            phi = np.arange(0, number_of_sensors) * delta_phi - phi_0
            sensor_positions = np.asarray([
                radius * np.cos(phi),
                radius * np.sin(phi),
                np.zeros(phi.shape)
            ])

    elif shape == 'chime3':
        assert scale == None, scale
        assert (
            number_of_sensors == None or number_of_sensors == 6
        ), number_of_sensors

        sensor_positions = np.asarray(
            [
                [-0.1, 0, 0.1, -0.1, 0, 0.1],
                [0.095, 0.095, 0.095, -0.095, -0.095, -0.095],
                [0, -0.02, 0, 0, 0, 0]
            ]
        )

    else:
        raise NotImplementedError('Given shape is not implemented.')

    sensor_positions = rot_x(rotate_x) @ sensor_positions
    sensor_positions = rot_y(rotate_y) @ sensor_positions
    sensor_positions = rot_z(rotate_z) @ sensor_positions

    if jitter is not None:
        sensor_positions += rng.normal(
            0., jitter, size=sensor_positions.shape
        )

    return np.asarray(sensor_positions + center)


def get_max_sensor_distance(sensors):
    """ Use this function to check for alias effects.

    Args:
        sensors: As generated by ``generate_sensor_positions()``.

    Returns:

    """
    maximum = 0.
    for i in range(sensors.shape[1]):
        for j in range(i - 1):
            maximum = np.maximum(
                maximum,
                np.linalg.norm(sensors[:, i] - sensors[:, j])
            )
    return maximum


def generate_random_source_positions(
        center=np.zeros((3, 1)),
        sources=1,
        distance_interval=(1, 2),
        dims=2,
        minimum_angular_distance=None,
        maximum_angular_distance=None,
        rng=np.random
):
    """ Generates random positions on a hollow sphere or circle.

    Samples are drawn from a uniform distribution on a hollow sphere with
    inner and outer radius according to distance_interval.

    The idea is to sample from an angular centric Gaussian distribution.

    Params:
        center
        sources
        distance_interval
        dims
        minimum_angular_distance: In randiant or None.
        maximum_angular_distance: In randiant or None.
        rng: Random number generator, if you need to set the seed.
    """
    enforce_angular_constrains = (
        minimum_angular_distance is not None or
        maximum_angular_distance is not None
    )

    if not dims == 2 and enforce_angular_constrains:
        raise NotImplementedError(
            'Only implemented distance constraints for 2D.'
        )

    accept = False
    while not accept:
        x = rng.normal(size=(3, sources))
        if dims == 2:
            x[2, :] = 0

        if enforce_angular_constrains:
            if not sources == 2:
                raise NotImplementedError
            angle = np.arctan2(x[1, :], x[0, :])
            difference = directional_minus(angle[None, :], angle[:, None])
            difference = difference[np.triu_indices_from(difference, k=1)]
            distance = np.abs(difference)
            if (
                minimum_angular_distance is not None and
                minimum_angular_distance > np.min(distance)
            ):
                continue
            if (
                maximum_angular_distance is not None and
                maximum_angular_distance < np.max(distance)
            ):
                continue
        accept = True

    x /= np.linalg.norm(x, axis=0)

    radius = rng.uniform(
        distance_interval[0] ** dims,
        distance_interval[1] ** dims,
        size=(1, sources)
    ) ** (1 / dims)

    x *= radius
    return np.asarray(x + center)


def generate_source_positions_on_circle(
        center=np.zeros((3, 1)),
        azimuth_angles=np.deg2rad(np.linspace(0, 360, 360, endpoint=False)),
        radius=1
):
    K = azimuth_angles.shape[0]
    positions = np.zeros((3, K))
    positions[0] = radius * np.cos(azimuth_angles)
    positions[1] = radius * np.sin(azimuth_angles)

    return positions + center


@deprecated
def generate_deterministic_source_positions(
        center=np.zeros((3, 1)),
        n=1,
        azimuth_angles=None,
        elevation_angles=None,
        radius=1,
        dims=3
):
    """
    Generate positions aligned at predefined angles on a sphere's surface or
    on a circle's boundary.

    :param center: The sphere's/circle's center coordinates in accord with dim.
    :param n: The amount of source positions to be sampled.
    :param azimuth_angles: List or tuple of azimuth angles pointing to samples.
     Size must equal n.
    :param elevation_angles: List or tuple of elevation angles pointing to
     samples. Size must equal n.
    :param radius: The sphere's/circle's radius to sample positions at.
    :param dims: 2 for circle, 3 for sphere
    :return: pos: (n times 3) numpy array of calculated positions

    Example:

    >>> center = [3,3,3]
    >>> n = 32
    >>> deltaAngle = np.pi/16
    >>> azimuth_angles = np.arange(0, 2*np.pi, deltaAngle)
    >>> elevation_angles = np.arange(0, np.pi, deltaAngle/2)
    >>> radius = 2
    >>> dims = 3
    >>> source_positions = generate_deterministic_source_positions(
    ...     center, n, azimuth_angles, elevation_angles, radius, dims
    ... )

    """
    if not 2 <= dims <= 3:
        raise NotImplementedError("Dims out of implemented range. Please choose"
                                  "2 or 3.")
    if azimuth_angles is None:
        raise NotImplementedError("Please provide azimuth and elevation angles")
    if not len(azimuth_angles) == n or not len(elevation_angles) == n:
        raise EnvironmentError("length of azimuth angles and elevation angles "
                               "must be equal n")

    if elevation_angles is None:
        elevation_angles = np.zeros_like(azimuth_angles)

    x = center
    if dims == 2:
        x[2, :] = 0
    pos = np.array([x] * n, dtype=np.float64).reshape((n, 3))
    z = np.array([(np.cos(elevation_angles) * np.cos(azimuth_angles)),
                  (np.cos(elevation_angles) * np.sin(azimuth_angles)),
                  np.sin(elevation_angles)]).T
    pos += radius * z
    return pos


def is_inside_room(dimensions, x):
    """
    Treats x as 3-dim vector and determines whether it's inside the
    room dimensions.

    :param dimensions: 3-object-sequence. Denotes the room dimensions.
    :param x: 3-object-sequence. Denotes the point to verify.
    :return: True for x being inside the room dimensions and False otherwise.
    """
    positive = all([elem > 0 for elem in x])  # all elements shall be greater 0
    return positive and np.all(np.subtract(dimensions, x))


def generate_uniformly_random_sources_and_sensors(
        dimensions,
        sources,
        sensors,
        rng=np.random
):
    """
    Returns two lists with random sources and sensors
    within the room dimensions.

    :param dimensions: 1x3 list; room dimensions in meters e.g. [9,7,3]
    :param sources: Integer; Number of desired source positions e.g. 3
    :param sensors: Integer; Number of desired sensor positions e.g. 1
    :return: source_list, sensor_list:
        Each is a list of 3-element-lists denoting the positions' coordinates

    >>> room = (4.5, 5, 3)
    >>> sources = generate_random_source_positions()
    >>> sensors = generate_sensor_positions(shape='triangle', scale=0.1)
    >>> from paderbox.visualization import figure_context
    >>> with figure_context():
    ...     simple_plot(room, sources, sensors)
    >>> plt.show()

    """
    source_list = []
    sensor_list = []

    for _ in range(sources):
        x = rng.uniform(10 ** -3, dimensions[0])
        y = rng.uniform(10 ** -3, dimensions[1])
        z = rng.uniform(10 ** -3, dimensions[2])
        source_list.append([x, y, z])
    for _ in range(sensors):
        x = rng.uniform(10 ** -3, dimensions[0])
        y = rng.uniform(10 ** -3, dimensions[1])
        z = rng.uniform(10 ** -3, dimensions[2])
        sensor_list.append([x, y, z])
    return source_list, sensor_list


def plot(room=None, sources=None, sensors=None, ax=None):
    if np.ndim(room) == 1:
        room = np.reshape(room, (-1, 1))
    if np.ndim(sources) == 1:
        sources = np.reshape(sources, (-1, 1))
    if np.ndim(sensors) == 1:
        sensors = np.reshape(sensors, (-1, 1))
    for parameter in (room, sources, sensors):
        assert parameter is None or np.shape(parameter)[0] == 3

    if ax is None:
        _, ax = plt.subplots()

    for axis in 'x y'.split():
        ax.locator_params(axis=axis, nbins=7)

    if room is not None:
        room = np.asarray(room)
        assert room.size == 3
        room = room.squeeze()
        ranges = np.asarray([0 * room, room]).T
        setup = {'alpha': 0.2, 'c': 'b'}
        for a in range(2):
            ax.plot(ranges[0, [a, a]], ranges[1], **setup)
            ax.plot(ranges[0], ranges[1, [a, a]], **setup)

    if sources is not None:
        sources = np.asarray(sources)
        ax.scatter(sources[0, :], sources[1, :])

    if sensors is not None:
        sensors = np.asarray(sensors)
        setup = {'c': 'r'}
        ax.scatter(sensors[0, :], sensors[1, :], **setup)

    plt.axis('equal')
