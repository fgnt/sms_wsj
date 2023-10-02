"""
Helps to quickly create source and sensor positions.
Try it with the following code:

>>> import numpy as np
>>> import sms_wsj.reverb.scenario as scenario
>>> src = scenario.generate_random_source_positions(dims=2, sources=1000)
>>> src[1, :] = np.abs(src[1, :])
>>> mic = scenario.generate_sensor_positions(shape='linear', scale=0.1, number_of_sensors=6)
"""

import numpy as np
from sms_wsj.reverb.rotation import rot_x, rot_y, rot_z


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
        center=np.zeros((3, 1), dtype=np.float64),
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
        sensor_positions = np.zeros((3, number_of_sensors), dtype=np.float64)
        sensor_positions[1, :] = scale * np.arange(number_of_sensors)
        sensor_positions -= np.mean(sensor_positions, keepdims=True, axis=1)

    elif shape == 'circular':
        if number_of_sensors == 1:
            sensor_positions = np.zeros((3, 1), dtype=np.float64)
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
        assert scale is None, scale
        assert (
                number_of_sensors is None or number_of_sensors == 6
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
            difference = np.angle(
                np.exp(1j * (angle[None, :], angle[:, None])))
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
