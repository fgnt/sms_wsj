"""
Offers methods for calculating room impulse responses and convolutions of these
with audio signals.
"""

import numpy
import numpy as np
import scipy

import nt.reverb.CalcRIR_Simple_C as tranVuRIR
import nt.reverb.scenario as scenario

eps = 1e-60
window_length = 256

available_rir_algorithms = ['tran_vu_python',
                            'tran_vu_cython',
                            'tran_vu_python_loopy']

def generate_rir(
        room_dimensions,
        source_positions,
        sensor_positions,
        sound_decay_time,
        sample_rate=16000,
        filter_length=2 ** 13,
        sensor_orientations=None,
        sensor_directivity=None,
        sound_velocity=343,
        algorithm=None
):
    """ Wrapper for different RIR generators. Will replace generate_RIR().

    Args:
        room_dimensions: Numpy array with shape (3, 1)
            which holds coordinates x, y and z.
        source_positions: Numpy array with shape (3, number_of_sources)
            which holds coordinates x, y and z in each column.
        sensor_positions: Numpy array with shape (3, number_of_sensors)
            which holds coordinates x, y and z in each column.
        sound_decay_time: Reverberation time in seconds.
        sample_rate: Sampling rate in Hertz.
        filter_length: Filter length, typically 2**13.
            Longer huge reverberation times.
        sensor_orientations: Numpy array with shape (2, 1)
            which holds azimuth and elevation angle in each column.
        sensor_directivity: String determining directivity for all sensors.
        sound_velocity: Set to 343 m/s.
        algorithm: The only implemented algorithm are
            'tran_vu_cython', 'tran_vu_python', 'tran_vu_python_loopy'.

    Returns: Numpy array of room impulse respones with
        shape (number_of_sources, number_of_sensors, filter_length).
    """

    assert room_dimensions.shape == (3, 1)
    assert source_positions.shape[0] == 3
    assert sensor_positions.shape[0] == 3

    number_of_sources = source_positions.shape[1]
    number_of_sensors = sensor_positions.shape[1]

    if algorithm is None:
        algorithm = 'tran_vu_cython'
    assert algorithm in [
        'tran_vu_cython', 'tran_vu_python', 'tran_vu_python_loopy'
    ], 'Unknown algorithm {}'.format(algorithm)

    if sensor_orientations is None:
        sensor_orientations = np.zeros((2, number_of_sources))

    if sensor_directivity is None:
        sensor_directivity = 'omnidirectional'

    if algorithm == 'tran_vu_cython':
        assert (
            np.all(sensor_orientations == np.zeros((2, number_of_sources))) and
            sensor_directivity == 'omnidirectional'
        ), 'Directional sensors is wrongly implemented in the Cython code.'

        # rir = generate_RIR(
        #     roomDimension=list(room_dimensions[:, 0]),
        #     sourcePositions=source_positions.T,
        #     sensorPositions=sensor_positions.T,
        #     samplingRate=sample_rate,
        #     filterLength=filter_length,
        #     soundDecayTime=sound_decay_time,
        #     algorithm='TranVu',
        #     sensorOrientations=sensor_orientations,
        #     sensorDirectivity=sensor_directivity,
        #     soundvelocity=sound_velocity
        # ).transpose((2, 1, 0))

        noiseFloor = -60
        rir = tranVuRIR.calc(
            np.ndarray.astype(room_dimensions[:, 0], dtype=np.float64),
            np.ndarray.astype(source_positions.T, dtype=np.float64),
            np.ndarray.astype(sensor_positions.T, dtype=np.float64),
            sample_rate,
            filter_length, sound_decay_time * 1000, noiseFloor,
            np.ndarray.astype(sensor_orientations, dtype=np.float64),
            1,
            sound_velocity
        ).transpose((2, 1, 0))

    elif algorithm == 'tran_vu_python':
        assert (
            np.all(sensor_orientations == np.zeros((2, number_of_sources))) and
            sensor_directivity == 'omnidirectional'
        ), 'Directional sensors are not supported with this algorithm.'

        rir = _generate_rir_tran_vu_python(
            room_dimensions=room_dimensions,
            source_positions=source_positions,
            sensor_positions=sensor_positions,
            sound_decay_time=sound_decay_time,
            filter_length=filter_length,
            sampling_rate=sample_rate
        )

    elif algorithm == 'tran_vu_python_loopy':
        assert (
            np.all(sensor_orientations == np.zeros((2, number_of_sources))) and
            sensor_directivity == 'omnidirectional'
        ), 'Directional sensors are not supported with this algorithm.'

        rir = _generate_rir_tran_vu_python_loopy(
            room_dimensions=room_dimensions,
            source_positions=source_positions,
            sensor_positions=sensor_positions,
            sound_decay_time=sound_decay_time,
            filter_length=filter_length,
            sampling_rate=sample_rate
        )
    else:
        raise NotImplementedError(
            'Algorithm "{}" is unknown.'.format(algorithm)
        )

    assert rir.shape[0] == number_of_sources
    assert rir.shape[1] == number_of_sensors
    assert rir.shape[2] == filter_length

    return rir


def generate_RIR(roomDimension,
                 sourcePositions,
                 sensorPositions,
                 samplingRate,
                 filterLength,
                 soundDecayTime,
                 algorithm="TranVu",
                 sensorOrientations=None,
                 sensorDirectivity="omnidirectional",
                 soundvelocity=343):
    """
    Generates a room impulse response.

    :param roomDimension: 3-floats-sequence; The room dimensions in meters
    :param sourcePositions: List of 3-floats-lists (#sources) containing
        the source position coordinates in meter within room dimension.
    :param sensorPositions: List of 3-floats-List (#sensors). The sensor
        positions in meter within room dimensions.
    :param samplingRate: scalar in Hz.
    :param filterLength: number of filter coefficients
    :param soundDecayTime: scalar in seconds. Reverberation time.
    :param algorithm: algorithm used for calculation. Default is "TranVu".
        Choose from: "TranVu","Habets","Lehmann","LehmannFast","AllenBerkley"
    :param sensorOrientations: List of name,value pairs (#sensors). Specifies
        orientation of each sensor using azimuth and elevation angle.
    :param sensorDirectivity: string determining directivity for all sensors.
        default:'omnidirectional'. Choose from:'omnidirectional', 'subcardioid',
        'cardioid','hypercardioid','bidirectional'
    :param soundvelocity: scalar in m/s. default: 343
    :return: RIR as Numpy matrix (filterlength x numberSensors x numberSources)

    Note that Having 1 source yields a RIR with shape (filterlength,numberSensors,1)
    whereas matlab method would return 2-dimensional matrix (filterlength,
    numberSensors)

    Example:

    >>> roomDim = (10,10,4)
    >>> sources = ((1,1,2),)
    >>> mics = ((2,3,2),(9,9,2))
    >>> sampleRate = 16000
    >>> filterLength = 2**13
    >>> T60 = 0.3
    >>> pyRIR = generate_RIR(roomDim,sources,mics,sampleRate, filterLength,T60)
    """

    # These are lists of possible picks
    algorithmList = (
        "TranVu", "Habets", "Lehmann", "LehmannFast", "AllenBerkley")
    directivityList = {"omnidirectional": 1, "subcardioid": 0.75,
                       "cardioid": 0.5,
                       "hypercardioid": 0.25, "bidirectional": 0}

    # get number of sensors and sources
    try:
        numSources = len(sourcePositions)
        numSensors = len(sensorPositions)
    except EnvironmentError:
        print("source and/or sensor positions aren't lists/tuples. Can't call"
              "len() on them.")

    # verify input for correct datatypes and values
    if not len(roomDimension) == 3:
        raise Exception("RoomDimensions needs 3 positive numbers!")
    if not (len(sourcePositions[s]) == 3 for s in range(numSources)) or \
            not all(
                scenario.is_inside_room(roomDimension, [u, v, w]) for u, v, w \
                in sourcePositions):
        raise Exception("Source positions aren't lists of positive 3-element-"
                        "lists or inside room dimensions!")
    if not (len(sensorPositions[s]) == 3 for s in range(numSensors)) or \
            not (all(scenario.is_inside_room(roomDimension, [s, t, u])) for
                 s, t, u \
                 in sensorPositions):
        raise Exception("Sensor positions aren't lists of positive 3-element-"
                        "lists or inside room dimensions!")
    if not numpy.isscalar(samplingRate):
        raise Exception("sampling rate isn't scalar!")
    if not numpy.isscalar(filterLength):
        raise Exception("Filter length isn't scalar!")
    if type(soundDecayTime) == str:
        raise Exception("sound decay time should be numeric!")
    if not any(algorithm == s for s in algorithmList):
        raise Exception("algorithm " + algorithm + " is unknown! Please choose"
                                                   "one of the following: \n" +
                        algorithmList)
    if not any(sensorDirectivity == key for key in directivityList):
        raise Exception("sensor directivity " + sensorDirectivity + " unknown!")
    if not numpy.isscalar(soundvelocity):
        raise Exception("sound velocity isn't scalar!")

    # Give optional arguments default values
    if sensorOrientations is None:
        sensorOrientations = []
        for x in range(numSensors):
            sensorOrientations.append((0, 0))

    # todo: Fall 'Lehmann' und 'omnidirectional' ausschließen!
    # todo: Fall 'LehmannFast' und sound velocity != 343 ausschließen!

    alpha = directivityList[sensorDirectivity]

    rir = numpy.zeros((filterLength, numSensors, numSources))

    # todo: Mehr Algorithmen in Betracht ziehen
    if algorithm == "TranVu":
        # TranVU method
        noiseFloor = -60
        rir = tranVuRIR.calc(numpy.asarray(roomDimension, dtype=numpy.float64),
                             numpy.asarray(sourcePositions,
                                           dtype=numpy.float64),
                             numpy.asarray(sensorPositions,
                                           dtype=numpy.float64),
                             samplingRate,
                             filterLength, soundDecayTime * 1000, noiseFloor,
                             numpy.asarray(sensorOrientations,
                                           dtype=numpy.float64),
                             alpha, soundvelocity)
    else:
        raise NotImplementedError(
            "The chosen algorithm is not implemented yet.")
    return rir


def blackman_harris_window(x):
    # Can not be replaced by from scipy.signal import blackmanharris.
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    x = np.pi * (x - window_length / 2) / window_length
    x = a0 - a1 * np.cos(2.0 * x) + a2 * np.cos(4.0 * x) - a3 * np.cos(6.0 * x)
    return np.maximum(x, 0)


def _generate_rir_tran_vu_python_loopy(
        room_dimensions,
        source_positions,
        sensor_positions,
        sound_decay_time,
        filter_length,
        sampling_rate,
        noise_threshold=-60,
        sound_velocity=343,
        dtype=np.float64
):
    print('Hello world.')
    air_coefficient = 0.9991

    norm_cut_off = 0.95
    samples_per_meter = sampling_rate / sound_velocity

    sources = source_positions.shape[1]
    sensors = sensor_positions.shape[1]

    if sound_decay_time > 0:
        room_surface = 2 * (
            room_dimensions[0] * room_dimensions[1] +
            room_dimensions[1] * room_dimensions[2] +
            room_dimensions[2] * room_dimensions[0]
        )
        room_volume = np.prod(room_dimensions)

        # Unknown parameter
        alpha = 1 - np.exp(
            (-24 * np.log(10) * room_volume) /
            (sound_velocity * sound_decay_time * room_surface)
        )
        reflection_coefficient = -np.sqrt(1 - alpha)

        image_order = int(np.maximum(
            10,
            np.ceil(noise_threshold / 10 / np.log10(1 - alpha) / 3)
        ))
    else:
        reflection_coefficient = 0.0
        image_order = 0

    rir = np.zeros((sources, sensors, filter_length), dtype=dtype)

    for s in range(sources):

        for z in range(-image_order, image_order + 1):

            image = np.zeros((3,), dtype=dtype)
            if z % 2 == 0:
                image[2] = z * room_dimensions[2] + source_positions[2, s]
            else:
                image[2] = (z + 1) * room_dimensions[2] - source_positions[2, s]

            for x in range(-image_order, image_order + 1):

                if x % 2 == 0:
                    image[0] = x * room_dimensions[0] + source_positions[0, s]
                else:
                    image[0] = (x + 1) * room_dimensions[0] - source_positions[
                        0, s]

                for y in range(-image_order, image_order + 1):
                    if y % 2 == 0:
                        image[1] = y * room_dimensions[1] + source_positions[
                            1, s]
                    else:
                        image[1] = (y + 1) * room_dimensions[1] - \
                                   source_positions[1, s]

                    for m in range(sensors):
                        difference = image - sensor_positions[:, m]
                        distance = np.linalg.norm(difference)
                        attenuation = reflection_coefficient ** (
                            np.abs(x) + np.abs(y) + np.abs(z)
                        ) * air_coefficient ** distance / (1 + distance)

                        distance_in_samples = distance * samples_per_meter
                        int_delay = int(distance_in_samples + 0.5)
                        fractional_delay = distance_in_samples - int_delay
                        count = - fractional_delay - window_length / 2

                        for t in range(window_length):
                            if int_delay + t < filter_length:
                                win_si = np.pi * norm_cut_off * count
                                if np.abs(win_si) < eps:
                                    # 4th order Taylor approximation of windowed sinc()
                                    x2 = win_si ** 2
                                    win_si = norm_cut_off * blackman_harris_window(
                                        count) * (
                                                 1.0 - x2 / 6.0 + x2 ** 2 / 120)
                                else:
                                    # Direct computation of windowed sinc()
                                    # print(norm_cut_off, blackman_harris_window(count), np.sin(win_si), win_si)
                                    win_si = norm_cut_off * blackman_harris_window(
                                        count) * np.sin(win_si) / win_si
                                a = int_delay + t
                                # print(count, attenuation, win_si)
                                rir[s, m, a] += attenuation * win_si
                                count += 1
                            else:
                                break
    return rir


def _generate_rir_tran_vu_python(
        room_dimensions,
        source_positions,
        sensor_positions,
        sound_decay_time,
        filter_length,
        sampling_rate,
        noise_threshold=-60,
        sound_velocity=343,
        dtype=np.float64
):
    air_coefficient = 0.9991

    norm_cut_off = 0.95
    samples_per_meter = sampling_rate / sound_velocity

    sources = source_positions.shape[1]
    sensors = sensor_positions.shape[1]

    if sound_decay_time > 0:
        room_surface = 2 * (
            room_dimensions[0] * room_dimensions[1] +
            room_dimensions[1] * room_dimensions[2] +
            room_dimensions[2] * room_dimensions[0]
        )
        room_volume = np.prod(room_dimensions)

        # Unknown parameter
        alpha = 1 - np.exp(
            (-24 * np.log(10) * room_volume) /
            (sound_velocity * sound_decay_time * room_surface)
        )
        reflection_coefficient = -np.sqrt(1 - alpha)

        image_order = int(np.maximum(
            10,
            np.ceil(noise_threshold / 10 / np.log10(1 - alpha) / 3)
        ))
    else:
        reflection_coefficient = 0.0
        image_order = 0

    images = np.zeros((
        3, sources,
        2 * image_order + 1, 2 * image_order + 1, 2 * image_order + 1,
    ), dtype=dtype)

    for i in range(-image_order, image_order + 1):
        images[0, :, i, :, :] = (
            (i + (i % 2)) * room_dimensions[0] +
            (-1) ** (i % 2) * source_positions[0, :, None, None]
        )
        images[1, :, :, i, :] = (
            (i + (i % 2)) * room_dimensions[1] +
            (-1) ** (i % 2) * source_positions[1, :, None, None]
        )
        images[2, :, :, :, i] = (
            (i + (i % 2)) * room_dimensions[2] +
            (-1) ** (i % 2) * source_positions[2, :, None, None]
        )

    rir = np.zeros((sources, sensors, filter_length), dtype=dtype)
    t = np.asarray(range(window_length))

    differences = images[:, :, :, :, :, None] - sensor_positions[:, None, None, None, None, :]
    distances = np.linalg.norm(differences, axis=0)

    for s in range(sources):
        for x in range(-image_order, image_order + 1):
            for y in range(-image_order, image_order + 1):
                for z in range(-image_order, image_order + 1):
                    distance = distances[s, x, y, z, :]
                    attenuation = reflection_coefficient ** (
                        np.abs(x) + np.abs(y) + np.abs(z)
                    ) * air_coefficient ** distance / (1 + distance)
                    distance_in_samples = distance * samples_per_meter
                    int_delay = [int(_d) for _d in distance_in_samples + 0.5]
                    lengths = [min(window_length, max(filter_length - int_delay[m], 0)) for m in range(sensors)]
                    for m, length in zip(range(sensors), lengths):
                        if length > 0:
                            fractional_delay = distance_in_samples[m] - int_delay[m]
                            print(fractional_delay)
                            count = - fractional_delay - window_length / 2
                            win_si = np.pi * norm_cut_off * (count + t[:length])
                            win_si = np.where(win_si == 0, 1.0e-20, win_si)
                            win_si = norm_cut_off * blackman_harris_window((count + t[:length])) * np.sin(win_si) / win_si
                            rir[s, m, int_delay[m]:int_delay[m] + length] += attenuation[m] * win_si[:length]

    return rir


def convolve(signal, impulse_resonse):
    """ Convolution of time signal with impulse response.

    Takes audio signals and the impulse responses according to their position
    and returns the convolution. The number of audio signals in x are required
    to correspond to the number of sources in the given RIR.
    Convolution is conducted through frequency domain via FFT.

    Args:
        signal: Time signal with shape (sources, samples)
        impulse_response: Shape (sources, sensors, samples)

    Returns:

    """
    sources, samples = signal.shape
    sources_, sensors, filter_length = impulse_resonse.shape
    assert sources == sources_

    x = np.zeros((sources, sensors, samples + filter_length - 1))
    for source_index in range(sources):
        for target_index in range(sensors):
            x[source_index, target_index, :] = scipy.signal.fftconvolve(
                signal[source_index, :],
                impulse_resonse[source_index, target_index, :]
            )
    return x
