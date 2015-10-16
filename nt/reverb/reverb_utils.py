import numpy as np
import nt.reverb.CalcRIR_Simple_C as RIR

def generate_RIR(roomDimension, sourcePositions, sensorPositions, samplingRate,
                 filterLength, soundDecayTime, algorithm="TranVu",
                 sensorOrientations=None, sensorDirectivity="omnidirectional",
                 soundvelocity=343):
    """
    Generates a room impulse response.
    :param roomDimension:
    :param sourcePositions:
    :param sensorPositions:
    :param samplingRate:
    :param filterLength:
    :param soundDecayTime:
    :param algorithm:
    :param sensorOrientations:
    :param sensorDirectivity:
    :param soundvelocity:
    :return:
    """

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
            not all(isInsideRoom(roomDimension,[s,t,u]) for s,t,u \
            in sourcePositions):
        raise Exception("Source positions aren't lists of positive 3-element-"
                        "lists or inside room dimensions!")
    if not (len(sensorPositions[s]) == 3 for s in range(numSensors)) or \
            not all(isInsideRoom(roomDimension,[s,t,u]) for s,t,u \
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
        raise Exception("sensor directivity " + sensorDirectivity + " unknown!")
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

    rir = np.empty((filterLength,numSensors,numSources))

    # todo: Unterscheide zwischen allen algorithmen
    # TranVU method
    noiseFloor = -60
    rir = RIR.calc()




def isInsideRoom(roomDim,x):
    """
    Treats x as 3-dim vector and determines whether it's inside the
     room dimensions.
    :param roomDim: 3-object-sequence. Denotes the room dimensions.
    :param x: 3-object-sequence. Denotes the point to verify.
    :return: True for x being inside the room dimensions and False otherwise.
    """
    positive = all(elem > 0 for elem in x)
    return positive & all(elem > 0 for elem in np.subtract(roomDim,x))