import numpy,scipy
import nt.reverb.CalcRIR_Simple_C as tranVuRIR
#import nt.reverb.Habets_RIR_C as HabetsRIR
import nt.reverb.scenario as scenario
from scipy import signal

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
    :return:RIR as Numpy matrix (filterlength x numberSensors x numberSources)

    note: Having 1 source yields a RIR with shape (filterlength,numberSensors,1)
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
    >>> pyRIR.shape
    (8192, 2, 1)
    """

    # These are lists of possible picks
    algorithmList = ("TranVu","Habets","Lehmann","LehmannFast","AllenBerkley")
    directivityList = {"omnidirectional":1,"subcardioid":0.75,"cardioid":0.5,
                       "hypercardioid":0.25,"bidirectional":0}

    # get number of sensors and sources
    try:
        numSources = len(sourcePositions)
        numSensors = len(sensorPositions)
    except EnvironmentError:
        print("source and/or sensor positions aren't lists/tuples. Can't call"
              "len() on them.")

    # verify input for correct datatypes and values
    if not len(roomDimension)==3:
        raise Exception("RoomDimensions needs 3 positive numbers!")
    if not (len(sourcePositions[s]) == 3 for s in range(numSources)) or \
            not all(scenario.isInsideRoom(roomDimension,[u,v,w]) for u,v,w \
                    in sourcePositions):
        raise Exception("Source positions aren't lists of positive 3-element-"
                        "lists or inside room dimensions!")
    if not (len(sensorPositions[s]) == 3 for s in range(numSensors)) or \
            not (all(scenario.isInsideRoom(roomDimension,[s,t,u])) for s,t,u \
            in sensorPositions):
        raise Exception("Sensor positions aren't lists of positive 3-element-"
                        "lists or inside room dimensions!")
    if not numpy.isscalar(samplingRate):
        raise Exception("sampling rate isn't scalar!")
    if not numpy.isscalar(filterLength):
        raise Exception("Filter length isn't scalar!")
    if type(soundDecayTime)==str:
        raise Exception("sound decay time should be numeric!")
    if not any(algorithm == s for s in algorithmList):
        raise Exception("algorithm "+algorithm+" is unknown! Please choose"
                                               "one of the following: \n"+
                                                algorithmList)
    if not any(sensorDirectivity == key for key in directivityList):
        raise Exception("sensor directivity "+sensorDirectivity+" unknown!")
    if not numpy.isscalar(soundvelocity):
        raise Exception("sound velocity isn't scalar!")

    # Give optional arguments default values
    if not sensorOrientations:
        sensorOrientations = []
        for x in range(numSensors):
            sensorOrientations.append((0,0))

    # todo: Fall 'Lehmann' und 'omnidirectional' ausschließen!
    # todo: Fall 'LehmannFast' und sound velocity != 343 ausschließen!

    alpha = directivityList[sensorDirectivity]

    rir = numpy.zeros((filterLength,numSensors,numSources))

    # todo: Mehr Algorithmen in Betracht ziehen
    if algorithm == "TranVu":
        # TranVU method
        noiseFloor = -60
        rir = tranVuRIR.calc(numpy.asarray(roomDimension,dtype=numpy.float64),
                       numpy.asarray(sourcePositions,dtype=numpy.float64),
                       numpy.asarray(sensorPositions,dtype=numpy.float64),
                       samplingRate,
                       filterLength,soundDecayTime*1000,noiseFloor,
                       numpy.asarray(sensorOrientations, dtype=numpy.float64 ),
                       alpha,soundvelocity)
    else:
        raise NotImplementedError("The chosen algorithm is not implemented yet.")
    return rir

def nearfield_time_of_flight(source_positions, sensor_positions, sound_velocity=343):
    """ Calculates exact time of flight in seconds without farfield assumption.

    :param source_positions: Array of 3D source position column vectors.
    :param sensor_positions: Array of 3D sensor position column vectors.
    :param sound_velocity: Speed of sound in m/s.
    :return: Time of flight in s.
    """
    # TODO: Check, if this works for any number of sources and sensors.
    difference = source_positions[:, :, numpy.newaxis] - sensor_positions[:, numpy.newaxis, :]
    difference = numpy.linalg.norm(difference, axis=0)
    return difference / sound_velocity


def steering_vector(time_of_flight, frequency):
    """ Calculates a steering vector.

    Keep in mind, that many steering vectors describe the same.

    :param time_of_flight: Time of flight in s.
    :param frequency: Vector of center frequencies as created
        by `get_stft_center_frequencies()`.
    :return:
    """
    # TODO: Check, if this works for any number of sources and sensors.
    return numpy.exp(-2j * numpy.pi
                     * frequency[numpy.newaxis, numpy.newaxis, :]
                     * time_of_flight[:, :, numpy.newaxis])

def fft_convolve(x,impulse_response):
    """
    Takes an audio signal and an impulse response and returns the convolution.
    Convolution is conducted through frequency domain via FFT.
    :param x: [1xD - array] the audio signal to convolute
    :param impulse_response: [filter_length x number_sensors x number_sources - numpy matrix ]
    The three dimensional impulse response.
    :return: convolved_signal: [number_sensors x number_sources x signal_length - numpy matrix]
    The convoluted signal for every sensor and each source
    """
    # Get number of sources and sensors
    num_sensors = impulse_response.shape[1]
    num_sources = impulse_response.shape[2]
    convolved_signal = numpy.zeros([num_sensors,
                                    num_sources,
                                    len(x)+len(impulse_response)-1])
    # fftconvolve for every sensor and source
    for i in range(num_sensors):
        for j in range(num_sources):
            convolved_signal[i,j,:] = signal.fftconvolve(x,
                                                         impulse_response[:,i,j])

    return convolved_signal

def time_convolve(x,impulse_response):
    """
    Takes an audio signal and an impulse response and returns the convolution.
    Convolution is conducted through time domain.
    :param x: [1xD - array] the audio signal to convolve
    :param impulse_response: [filter_length x number_sensors x number_sources - numpy matrix ]
    The three dimensional impulse response.
    :return: convolved_signal: [number_sensors x number_sources x signal_length - numpy matrix]
    The convoluted signal for every sensor and each source
    """
    # Get number of sources and sensors
    num_sensors = impulse_response.shape[1]
    num_sources = impulse_response.shape[2]
    convolved_signal = numpy.zeros([num_sensors,
                                    num_sources,
                                    len(x)+len(impulse_response)-1])
    # convolve for every sensor and source
    for i in range(num_sensors):
        for j in range(num_sources):
            convolved_signal[i,j,:] = numpy.convolve(x,
                                                     impulse_response[:,i,j])

    return convolved_signal

#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()

