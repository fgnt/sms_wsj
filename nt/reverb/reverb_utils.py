"""
Offers methods for calculating room impulse responses and convolutions of these
with audio signals.
"""

import numpy
import numpy as np
from scipy import signal

import nt.reverb.CalcRIR_Simple_C as tranVuRIR
import nt.reverb.scenario as scenario


def generate_rir(
        room,
        sources,
        sensors,
        sound_decay_time,
        sampling_rate=16000,
        filter_length=2**13,
        sensor_orientations=None,
        sensor_directivity=None,
        sound_velocity=343,
        algorithm=None
):
    """ Wrapper of Wilhelm's RIR generator with sane defaults.

    Args:
        room: Numpy array with shape (3, 1)
            which holds coordinates x, y and z.
        sources: Numpy array with shape (3, number_of_sources)
            which holds coordinates x, y and z in each column.
        sensors: Numpy array with shape (3, number_of_sensors)
            which holds coordinates x, y and z in each column.
        sound_decay_time: Reverberation time in seconds.
        sampling_rate: Sampling rate in Hertz.
        filter_length: Filter length, typically 2**13.
            Longer huge reverberation times.
        sensor_orientations: Numpy array with shape (2, 1)
            which holds azimuth and elevation angle in each column.
        sensor_directivity: String determining directivity for all sensors.
        sound_velocity: Set to 343 m/s.
        algorithm: The only implemented algorithm is 'TranVu'.

    Returns: Numpy array of room impulse respones with
        shape (number_of_sources, number_of_sensors, filter_length).
    """

    assert room.shape == (3, 1)
    assert sources.shape[0] == 3
    assert sensors.shape[0] == 3

    number_of_sources = sources.shape[1]
    number_of_sensors = sensors.shape[1]

    if sensor_orientations is None:
        sensor_orientations = np.zeros((2, number_of_sources))

    if sensor_directivity is None:
        sensor_directivity = 'omnidirectional'

    if algorithm is None:
        algorithm = 'TranVu'

    rir = generate_RIR(
        roomDimension=list(room[:, 0]),
        sourcePositions=sources.T,
        sensorPositions=sensors.T,
        samplingRate=sampling_rate,
        filterLength=filter_length,
        soundDecayTime=sound_decay_time,
        algorithm=algorithm,
        sensorOrientations=sensor_orientations,
        sensorDirectivity=sensor_directivity,
        soundvelocity=sound_velocity
    ).transpose((2, 1, 0))

    assert rir.shape[0] == number_of_sources
    assert rir.shape[1] == number_of_sensors
    assert rir.shape[2] == filter_length

    return rir


def generate_RIR(roomDimension, sourcePositions, sensorPositions, samplingRate,
                 filterLength, soundDecayTime, algorithm="TranVu",
                 sensorOrientations=None, sensorDirectivity="omnidirectional",
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


def filter():
    pass


def fft_convolve(x, impulse_response):
    """
    Takes audio signals and the impulse responses according to their position
    and returns the convolution. The number of audio signals in x are required
    to correspond to the number of sources in the given RIR.
    Convolution is conducted through frequency domain via FFT.

    :param x: Source signal with shape (number_sources, audio_signal_length)
    :param impulse_response: Impulse response
        with shape (filter_length, number_sensors, number_sources)
    :return: convolved_signal: Convoluted signal for every sensor and each source
        with shape (number_sensors, number_sources, signal_length)
    """
    _, sensors, sources = impulse_response.shape

    if not sources == x.shape[0]:
        raise Exception(
            "Number audio signals (" +
            str(x.shape[0]) +
            ") does not match source positions (" +
            str(sources) +
            ") in given impulse response!"
        )
    convolved_signal = numpy.zeros(
        [sensors, sources, x.shape[1] + len(impulse_response) - 1]
    )

    for i in range(sensors):
        for j in range(sources):
            convolved_signal[i, j, :] = signal.fftconvolve(
                x[j, :],
                impulse_response[:, i, j]
            )

    return convolved_signal


def time_convolve(x, impulse_response):
    """
    Takes audio signals and the impulse responses according to their position
    and returns the convolution. The number of audio signals in x are required
    to correspond to the number of sources in the given RIR.
    Convolution is conducted through time domain.

    :param x: [number_sources x audio_signal_length - array] the audio signal
        to convolve
    :param impulse_response:
        [filter_length x number_sensors x number_sources - numpy matrix ]
        The three dimensional impulse response.
    :return: convolved_signal:
        [number_sensors x number_sources x signal_length - numpy matrix]
        The convoluted signal for every sensor and each source
    """
    _, sensors, sources = impulse_response.shape

    if not sources == x.shape[0]:
        raise Exception(
            "Number audio signals (" +
            str(x.shape[0]) +
            ") does not match source positions (" +
            str(sources) +
            ") in given impulse response!"
        )
    convolved_signal = numpy.zeros(
        [sensors, sources, x.shape[1] + len(impulse_response) - 1]
    )

    for i in range(sensors):
        for j in range(sources):
            convolved_signal[i, j, :] = numpy.convolve(
                x[j, :],
                impulse_response[:, i, j]
            )

    return convolved_signal
