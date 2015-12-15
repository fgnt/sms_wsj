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
from operator import add
import random

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

    The idea is to sample from an angular centric Gaussian distribution.
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

def generate_deterministic_source_positions(center=numpy.zeros((3,1)),
                                                  n=1,
                                                  azimuth_angles = None,
                                                  elevation_angles = None,
                                                  radius=1,
                                                  dims=3):
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
    :return: pos: n times 3 numpy array of calculated positions

    Example:

    >>> center = [3,3,3]
    >>> n = 32
    >>> deltaAngle = numpy.pi/16
    >>> azimuth_angles = numpy.arange(0,2*numpy.pi,deltaAngle)
    >>> elevation_angles = numpy.arange(0,numpy.pi,deltaAngle/2)
    >>> radius = 2
    >>> dims = 3
    >>> source_positions = generate_deterministic_source_positions(center,n,azimuth_angles,elevation_angles,radius,dims)
    """
    if not 2 <= dims <= 3:
        raise NotImplementedError("Dims out of implemented range. Please choose"
                                  "2 or 3.")
    if azimuth_angles is None or elevation_angles is None:
        raise NotImplementedError("Please provide azimuth and elevation angles")
    if not len(azimuth_angles)==n or not len(elevation_angles)==n:
        raise EnvironmentError("length of azimuth angles and elevation angles "
                               "must be equal n")
    x = center
    if dims == 2:
        x[2,:] = 0
    pos = numpy.array([x]*n,dtype=numpy.float64).reshape((n,3))
    z = numpy.array([(numpy.cos(elevation_angles)*numpy.cos(azimuth_angles)),
                 (numpy.cos(elevation_angles)*numpy.sin(azimuth_angles)),
                 numpy.sin(elevation_angles)]).T
    pos += radius*z
    return pos

def isInsideRoom(roomDim,x):
    """
    Treats x as 3-dim vector and determines whether it's inside the
    room dimensions.

    :param roomDim: 3-object-sequence. Denotes the room dimensions.
    :param x: 3-object-sequence. Denotes the point to verify.
    :return: True for x being inside the room dimensions and False otherwise.
    """
    positive = all([elem > 0 for elem in x]) # all elements shall be greater 0
    return positive and numpy.all(numpy.subtract(roomDim,x))


def generate_uniformly_random_sources_and_sensors(roomDim,numSources,numSensors):
    """
    Returns two lists with random sources and sensors
    within the room dimensions.

    :param roomDim: 1x3 list; room dimensions in meters e.g. [9,7,3]
    :param numSources: Integer; Number of desired source positions e.g. 3
    :param numSensors: Integer; Number of desired sensor positions e.g. 1
    :return:srcList,micList: Each is a list of 3-element-lists denoting
            the positions' coordinates
    """
    srcList = []
    micList = []
    # todo: avoid sensors/sources in the room's center
    for s in range(numSources):
        x = random.uniform(10**-3,roomDim[0])
        y = random.uniform(10**-3,roomDim[1])
        z = random.uniform(10**-3,roomDim[2])
        srcList.append([x,y,z])
    for m in range(numSensors):
        x = random.uniform(10**-3,roomDim[0])
        y = random.uniform(10**-3,roomDim[1])
        z = random.uniform(10**-3,roomDim[2])
        micList.append([x,y,z])
    return srcList,micList

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
