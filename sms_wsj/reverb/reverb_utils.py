"""
Offers methods for calculating room impulse responses and convolutions of these
with audio signals.
"""

import numpy as np
import scipy
import scipy.signal

eps = 1e-60
window_length = 256


# TODO: Refactor
def generate_rir(
        room_dimensions,
        source_positions,
        sensor_positions,
        sound_decay_time,
        sample_rate=16000,
        filter_length=2 ** 13,
        sensor_orientations=None,
        sensor_directivity=None,
        sound_velocity=343
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

    Returns: Numpy array of room impulse respones with
        shape (number_of_sources, number_of_sensors, filter_length).
    """
    import rirgen
    room_dimensions = np.array(room_dimensions)
    source_positions = np.array(source_positions)
    sensor_positions = np.array(sensor_positions)

    if np.ndim(source_positions) == 1:
        source_positions = np.reshape(source_positions, (-1, 1))
    if np.ndim(room_dimensions) == 1:
        room_dimensions = np.reshape(room_dimensions, (-1, 1))
    if np.ndim(sensor_positions) == 1:
        sensor_positions = np.reshape(sensor_positions, (-1, 1))

    assert room_dimensions.shape == (3, 1)
    assert source_positions.shape[0] == 3
    assert sensor_positions.shape[0] == 3

    number_of_sources = source_positions.shape[1]
    number_of_sensors = sensor_positions.shape[1]

    if sensor_orientations is None:
        sensor_orientations = np.zeros((2, number_of_sources))
    else:
        raise NotImplementedError(sensor_orientations)

    if sensor_directivity is None:
        sensor_directivity = 'omnidirectional'
    else:
        raise NotImplementedError(sensor_directivity)

    assert filter_length is not None
    rir = np.zeros(
        (number_of_sources, number_of_sensors, filter_length),
        dtype=np.float64
    )
    for k in range(number_of_sources):
        temp = rirgen.generate_rir(
            room_measures=room_dimensions[:, 0],
            source_position=source_positions[:, k],
            receiver_positions=sensor_positions.T,
            reverb_time=sound_decay_time,
            sound_velocity=sound_velocity,
            fs=sample_rate,
            n_samples=filter_length
        )
        rir[k, :, :] = np.asarray(temp)

    assert rir.shape[0] == number_of_sources
    assert rir.shape[1] == number_of_sensors
    assert rir.shape[2] == filter_length

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


def convolve(signal, impulse_response, truncate=False):
    """ Convolution of time signal with impulse response.

    Takes audio signals and the impulse responses according to their position
    and returns the convolution. The number of audio signals in x are required
    to correspond to the number of sources in the given RIR.
    Convolution is conducted through frequency domain via FFT.

    x = h conv s

    Args:
        signal: Time signal with shape (..., samples)
        impulse_response: Shape (..., sensors, filter_length)
        truncate: Truncates result to input signal length if True.

    Alternative args:
        signal: Time signal with shape (samples,)
        impulse_response: Shape (filter_length,)

    Returns: Convolution result with shape (..., sensors, length) or (length,)

    >>> signal = np.asarray([1, 2, 3])
    >>> impulse_response = np.asarray([1, 1])
    >>> print(convolve(signal, impulse_response))
    [1. 3. 5. 3.]

    >>> K, T, D, filter_length = 2, 12, 3, 5
    >>> signal = np.random.normal(size=(K, T))
    >>> impulse_response = np.random.normal(size=(K, D, filter_length))
    >>> convolve(signal, impulse_response).shape
    (2, 3, 16)

    >>> signal = np.random.normal(size=(T,))
    >>> impulse_response = np.random.normal(size=(D, filter_length))
    >>> convolve(signal, impulse_response).shape
    (3, 16)
    """
    signal = np.array(signal)
    impulse_response = np.array(impulse_response)

    if impulse_response.ndim == 1:
        x = convolve(signal, impulse_response[None, ...], truncate=truncate)
        x = np.squeeze(x, axis=0)
        return x

    *independent, samples = signal.shape
    *independent_, sensors, filter_length = impulse_response.shape
    assert independent == independent_, f'signal.shape {signal.shape} does' \
        f' not match impulse_response.shape {impulse_response.shape}'

    x = scipy.signal.fftconvolve(
        signal[..., None, :],
        impulse_response,
        axes=-1
    )

    return x[..., :samples] if truncate else x


def get_rir_start_sample(h, level_ratio=1e-1):
    """Finds start sample in a room impulse response.

    Selects that index as start sample where the first time
    a value larger than `level_ratio * max_abs_value`
    occurs.

    If you intend to use this heuristic, test it on simulated and real RIR
    first. This heuristic is developed on MIRD database RIRs and on some
    simulated RIRs but may not be appropriate for your database.

    If you want to use it to shorten impulse responses, keep the initial part
    of the room impulse response intact and just set the tail to zero.

    Params:
        h: Room impulse response with Shape (num_samples,)
        level_ratio: Ratio between start value and max value.

    >>> get_rir_start_sample(np.array([0, 0, 1, 0.5, 0.1]))
    2
    """
    assert level_ratio < 1, level_ratio
    if h.ndim > 1:
        assert h.shape[0] < 20, h.shape
        h = np.reshape(h, (-1, h.shape[-1]))
        return np.min(
            [get_rir_start_sample(h_, level_ratio=level_ratio) for h_ in h]
        )

    abs_h = np.abs(h)
    max_index = np.argmax(abs_h)
    max_abs_value = abs_h[max_index]
    # +1 because python excludes the last value
    larger_than_threshold = abs_h[:max_index + 1] > level_ratio * max_abs_value

    # Finds first occurrence of max
    rir_start_sample = np.argmax(larger_than_threshold)
    return rir_start_sample


if __name__ == "__main__":
    import doctest

    doctest.testmod()
