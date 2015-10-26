"""
Helps to quickly create source and sensor positions. Try it with the following code:

>>> from mpl_toolkits.mplot3d import proj3d
>>> import nt.reverb.scenario as scenario
>>> src = scenario.generate_random_source_positions(dims=2, n=1000)
>>> src[1, :] = numpy.abs(src[1, :])
>>> mic = scenario.generate_sensor_positions(shape='linear', scale=0.1)
>>> scenario.plot(src, mic)
"""

import numpy
import matplotlib.pyplot as plt


def generate_sensor_positions(shape='cube', center=numpy.zeros((3, 1)), scale=0.01):
    """
    Generate different sensor configurations as known from the Matlab implementations.

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
            [-b, -b,  b],
            [-b,  b, -b],
            [-b,  b,  b],
            [ b, -b, -b],
            [ b, -b,  b],
            [ b,  b, -b],
            [ b,  b,  b]
        ]).T
    elif shape == 'linear':
        b = scale / 2
        sensor_positions = numpy.array([
            [ b, 0, 0],
            [-b, 0, 0]
        ]).T
    else:
        raise NotImplementedError('Given shape is not implemented.')

    return sensor_positions + center


def generate_random_source_positions(center=numpy.zeros((3, 1)), n=1, distance_interval=(1, 2), dims=3):
    """ Generates random positions on a hollow sphere or circle.

    Samples are drawn from a uniform distribution on a hollow sphere with
    inner and outer radius according to distance_interval.

    The idea is to sample from a angular centric Gaussian distribution.
    """
    x = numpy.random.normal(size=(3, n))
    if dims == 2:
        x[2, :] = 0

    x /= numpy.linalg.norm(x, axis=0)

    radius = numpy.random.uniform(
        distance_interval[0]**dims,
        distance_interval[1]**dims,
        size=(1, n)
    )**(1 / dims)

    x *= radius
    return x + center


def plot(src=None, mic=None, room=None):
    fig = plt.figure(figsize=(8, 3.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    if room is None:
        ax1.set_xlim3d((-3, 3))
        ax1.set_ylim3d((-3, 3))
        ax1.set_zlim3d((-1, 1))
        ax2.set_xlim((-3, 3))
        ax2.set_ylim((-3, 3))
    else:
        ax1.set_xlim3d((-room[0], room[0]))
        ax1.set_ylim3d((-room[1], room[1]))
        ax1.set_zlim3d((-room[2], room[2]))
        ax2.set_xlim((-room[0], room[0]))
        ax2.set_ylim((-room[1], room[1]))

    if src is not None:
        ax1.scatter(src[0, :], src[1, :], src[2, :])
        ax2.scatter(src[0, :], src[1, :])

    if mic is not None:
        ax1.scatter(mic[0, :], mic[1, :], mic[2, :], c='r')
        ax2.scatter(mic[0, :], mic[1, :], c='r')
