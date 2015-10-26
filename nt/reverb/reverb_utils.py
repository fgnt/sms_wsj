import numpy as np
import nt.reverb.CalcRIR_Simple_C as tranVuRIR
import random

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
    >>> T60=0.3
    >>> pyRIR = generate_RIR(roomDim,sources,mics,sampleRate, filterLength,T60)
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
            not all(isInsideRoom(roomDimension,[u,v,w]) for u,v,w \
                    in sourcePositions):
        raise Exception("Source positions aren't lists of positive 3-element-"
                        "lists or inside room dimensions!")
    if not (len(sensorPositions[s]) == 3 for s in range(numSensors)) or \
            not (all(isInsideRoom(roomDimension,[s,t,u])) for s,t,u \
            in sensorPositions):
        raise Exception("Sensor positions aren't lists of positive 3-element-"
                        "lists or inside room dimensions!")
    if not np.isscalar(samplingRate):
        raise Exception("sampling rate isn't scalar!")
    if not np.isscalar(filterLength):
        raise Exception("Filter length isn't scalar!")
    if type(soundDecayTime)==str:
        raise Exception("sound decay time should be numeric!")
    if not any(algorithm == s for s in algorithmList):
        raise Exception("algorithm "+algorithm+" is unknown! Please choose"
                                               "one of the following: \n"+
                                                algorithmList)
    if not any(sensorDirectivity == key for key in directivityList):
        raise Exception("sensor directivity "+sensorDirectivity+" unknown!")
    if not np.isscalar(soundvelocity):
        raise Exception("sound velocity isn't scalar!")

    # Give optional arguments default values
    if not sensorOrientations:
        sensorOrientations = []
        for x in range(numSensors):
            sensorOrientations.append((0,0))

    # todo: Fall 'Lehmann' und 'omnidirectional' ausschließen!
    # todo: Fall 'LehmannFast' und sound velocity != 343 ausschließen!

    alpha = directivityList[sensorDirectivity]

    rir = np.zeros((filterLength,numSensors,numSources))

    # todo: Unterscheide zwischen allen Algorithmen
    # TranVU method
    noiseFloor = -60
    rir = tranVuRIR.calc(np.asarray(roomDimension,dtype=np.float64),
                   np.asarray(sourcePositions,dtype=np.float64),
                   np.asarray(sensorPositions,dtype=np.float64),
                   samplingRate,
                   filterLength,soundDecayTime*1000,noiseFloor,
                   np.asarray(sensorOrientations, dtype=np.float64 ),
                   alpha,soundvelocity)
    return rir



def isInsideRoom(roomDim,x):
    """
    Treats x as 3-dim vector and determines whether it's inside the
     room dimensions.
    :param roomDim: 3-object-sequence. Denotes the room dimensions.
    :param x: 3-object-sequence. Denotes the point to verify.
    :return: True for x being inside the room dimensions and False otherwise.
    """
    positive = all([elem > 0 for elem in x]) # all elements shall be greater 0
    return positive and np.all(np.subtract(roomDim,x))

def generateRandomSourcesAndSensors(roomDim,numSources,numSensors):
    """
    Returns two lists with random sources and sensors
    within the room dimensions
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


