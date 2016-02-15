"""
Offers methods for calculating room impulse responses and convolutions of these
with audio signals.
"""

import numpy
from scipy import signal

import nt.reverb.CalcRIR_Simple_C as tranVuRIR
import nt.reverb.scenario as scenario


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
    if not sensorOrientations:
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


def fft_convolve(x, impulse_response):
    """
    Takes audio signals and the impulse responses according to their position
    and returns the convolution. The number of audio signals in x are required
    to correspond to the number of sources in the given RIR.
    Convolution is conducted through frequency domain via FFT.

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
