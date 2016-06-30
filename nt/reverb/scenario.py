"""
Helps to quickly create source and sensor positions.
Try it with the following code:

>>> from mpl_toolkits.mplot3d import proj3d
>>> import nt.reverb.scenario as scenario
>>> src = scenario.generate_random_source_positions(dims=2, sources=1000)
>>> src[1, :] = numpy.abs(src[1, :])
>>> mic = scenario.generate_sensor_positions(shape='linear', scale=0.1)
>>> scenario.plot(src, mic)
"""

import numpy
import matplotlib.pyplot as plt
import random
import itertools
from nt.visualization.new_cm import viridis_hex

################################################################################
# Register Axes3D as a 'projection' object available for use just like any  axes
################################################################################
from mpl_toolkits.mplot3d import Axes3D


def generate_sensor_positions(
        shape='cube',
        center=numpy.zeros((3, 1)),
        scale=0.01
):
    """
    Generate different sensor configurations as known from the Matlab
    implementations.

    :param shape:
    :param center:
    :param scale:
    :return:
    """
    # TODO: Allow different dims for shape=linear
    if shape == 'cube':
        b = scale / 2
        sensor_positions = numpy.array([
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
        # b is the radius here
        b = scale
        sensor_positions = numpy.array([
            [b/2, -numpy.sqrt(3)/2 * b, 0],
            [b/2, numpy.sqrt(3)/2 * b, 0],
            [-b, 0, 0],
        ]).T
        sensor_positions -= numpy.mean(sensor_positions, axis=1, keepdims=True)
    elif shape == 'linear':
        b = scale / 2
        sensor_positions = numpy.array([
            [b, 0, 0],
            [-b, 0, 0]
        ]).T
    else:
        raise NotImplementedError('Given shape is not implemented.')

    return numpy.asarray(sensor_positions + center)


def generate_random_source_positions(
        center=numpy.zeros((3, 1)),
        sources=1,
        distance_interval=(1, 2),
        dims=3
):
    """ Generates random positions on a hollow sphere or circle.

    Samples are drawn from a uniform distribution on a hollow sphere with
    inner and outer radius according to distance_interval.

    The idea is to sample from an angular centric Gaussian distribution.
    """
    x = numpy.random.normal(size=(3, sources))
    if dims == 2:
        x[2, :] = 0

    x /= numpy.linalg.norm(x, axis=0)

    radius = numpy.random.uniform(
        distance_interval[0] ** dims,
        distance_interval[1] ** dims,
        size=(1, sources)
    ) ** (1 / dims)

    x *= radius
    return numpy.asarray(x + center)


def generate_source_positions_on_circle(
        center=numpy.zeros((3, 1)),
        azimuth_angles=numpy.deg2rad(numpy.arange(0, 360, 1)),
        radius=1
):
    return numpy.asarray(
        generate_deterministic_source_positions(
            center=center,
            n=azimuth_angles.shape[0],
            azimuth_angles=azimuth_angles,
            elevation_angles=numpy.zeros_like(azimuth_angles),
            radius=radius,
            dims=3
        )
    ).T


def generate_deterministic_source_positions(
        center=numpy.zeros((3, 1)),
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
    >>> deltaAngle = numpy.pi/16
    >>> azimuth_angles = numpy.arange(0, 2*numpy.pi, deltaAngle)
    >>> elevation_angles = numpy.arange(0, numpy.pi, deltaAngle/2)
    >>> radius = 2
    >>> dims = 3
    >>> source_positions = generate_deterministic_source_positions(center, n, azimuth_angles, elevation_angles, radius, dims)
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
        elevation_angles = numpy.zeros_like(azimuth_angles)

    x = center
    if dims == 2:
        x[2, :] = 0
    pos = numpy.array([x] * n, dtype=numpy.float64).reshape((n, 3))
    z = numpy.array([(numpy.cos(elevation_angles) * numpy.cos(azimuth_angles)),
                     (numpy.cos(elevation_angles) * numpy.sin(azimuth_angles)),
                     numpy.sin(elevation_angles)]).T
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
    return positive and numpy.all(numpy.subtract(dimensions, x))


def generate_uniformly_random_sources_and_sensors(
        dimensions,
        sources,
        sensors
):
    """
    Returns two lists with random sources and sensors
    within the room dimensions.

    :param dimensions: 1x3 list; room dimensions in meters e.g. [9,7,3]
    :param sources: Integer; Number of desired source positions e.g. 3
    :param sensors: Integer; Number of desired sensor positions e.g. 1
    :return: source_list, sensor_list:
        Each is a list of 3-element-lists denoting the positions' coordinates
    """
    source_list = []
    sensor_list = []
    # todo: avoid sensors/sources in the room's center
    for s in range(sources):
        x = random.uniform(10 ** -3, dimensions[0])
        y = random.uniform(10 ** -3, dimensions[1])
        z = random.uniform(10 ** -3, dimensions[2])
        source_list.append([x, y, z])
    for m in range(sensors):
        x = random.uniform(10 ** -3, dimensions[0])
        y = random.uniform(10 ** -3, dimensions[1])
        z = random.uniform(10 ** -3, dimensions[2])
        sensor_list.append([x, y, z])
    return source_list, sensor_list


def plot(room=None, sources=None, sensors=None, dictionary=None):
    """ Plot a given room with possible sources and sensors.

    All positions and distances in meters.

    TODO: This function has way to many hard coded numbers.

    >>> room = (4.5, 5, 3)
    >>> sources = generate_random_source_positions()
    >>> sensors = generate_sensor_positions(shape='triangle', scale=0.1)
    >>> from nt.visualization import context_manager
    >>> with context_manager():
    ...     plot(room, sources, sensors)
    >>> plt.show()

    :param room: Tuple or array of room dimensions with shape (3,)
    :param sources: Array of K source positions with shape (3, K)
    :param sensors: Array of D sensor positions with shape (3, D)
    :return:
    """
    room = numpy.asarray(room) if room is not None else None
    sources = numpy.asarray(sources) if sources is not None else None
    sensors = numpy.asarray(sensors) if sensors is not None else None

    for parameter in (room, sources, sensors):
        assert parameter is None or parameter.shape[0] == 3

    fig = plt.figure(figsize=(8, 3.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    for axis in 'x y z'.split():
        ax1.locator_params(axis=axis, nbins=5)
    for axis in 'x y'.split():
        ax2.locator_params(axis=axis, nbins=7)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax1.set_xlim3d((-1, room[0]+1))
    ax1.set_ylim3d((-1, room[1]+1))
    ax1.set_zlim3d((-1, room[2]+1))
    ax2.set_xlim(-1, room[0]+1)
    ax2.set_ylim(-1, room[1]+1)

    if room is not None:
        room = numpy.asarray(room)
        ranges = numpy.asarray([0*room, room]).T
        setup = {'alpha': 0.2, 'c': 'b'}
        for a, b in itertools.product(range(2), repeat=2):
            ax1.plot(ranges[0], ranges[1, [a, a]], ranges[2, [b, b]], **setup)
            ax1.plot(ranges[0, [a, a]], ranges[1], ranges[2, [b, b]], **setup)
            ax1.plot(ranges[0, [a, a]], ranges[1, [b, b]], ranges[2], **setup)

        for a in range(2):
            ax2.plot(ranges[0, [a, a]], ranges[1], **setup)
            ax2.plot(ranges[0], ranges[1, [a, a]], **setup)

    if sources is not None:
        ax1.scatter(sources[0, :], sources[1, :], sources[2, :])
        ax2.scatter(sources[0, :], sources[1, :])

    if sensors is not None:
        setup = {'c': 'r'}
        ax1.scatter(sensors[0, :], sensors[1, :], sensors[2, :], **setup)
        ax2.scatter(sensors[0, :], sensors[1, :], **setup)

    if dictionary is not None:
        colors = viridis_hex[::len(viridis_hex) // len(dictionary)]
        for (key, value), color in zip(dictionary.items(), colors):
            ax1.scatter(value[0, :], value[1, :], value[2, :], c=color)
            ax2.scatter(value[0, :], value[1, :], color=color, label=key)
            ax2.legend()

    plt.subplots_adjust(right=1.2)
    ax1.xaxis.pane.set_edgecolor('black')
    ax1.yaxis.pane.set_edgecolor('black')

    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    ax1.w_xaxis.line.set_color((0.0, 0.0, 0.0, 0.2))
    ax1.w_yaxis.line.set_color((0.0, 0.0, 0.0, 0.2))
    ax1.w_zaxis.line.set_color((0.0, 0.0, 0.0, 0.2))