
from hashlib import md5

import numpy as np
from scipy.signal import fftconvolve
from sms_wsj.reverb.reverb_utils import get_rir_start_sample

__all__ = [
    'scenario_map_fn',
]


def _example_id_to_rng(example_id):
    """
    >>> _example_id_to_rng('example_id').get_state()[1][0]
    2915193065
    """
    hash_value = md5(example_id.encode())
    hash_value = int(hash_value.hexdigest(), 16)
    hash_value -= 1  # legacy operation
    hash_value = hash_value % 2 ** 32
    return np.random.RandomState(hash_value)


def extract_piece(x, offset, target_length):
    """
    >>> extract_piece(np.arange(4), -1, 5)
    array([1, 2, 3, 0, 0])

    >>> extract_piece(np.arange(6), -1, 5)
    array([1, 2, 3, 4, 5])

    >>> extract_piece(np.arange(2), -2, 5)
    array([0, 0, 0, 0, 0])

    >>> extract_piece(np.arange(2), 1, 5)
    array([0, 0, 1, 0, 0])

    >>> extract_piece(np.arange(4), 1, 5)
    array([0, 0, 1, 2, 3])

    >>> extract_piece(np.arange(2), 5, 5)
    array([0, 0, 0, 0, 0])


    Args:
        x:
        offset:
            If negative, cut left side.
            If positive: pad left side.
        target_length:

    Returns:

    """
    def pad_axis(array, pad_width, axis=-1):
        array = np.asarray(array)

        npad = np.zeros([array.ndim, 2], dtype=int)
        npad[axis, :] = pad_width
        return np.pad(array, pad_width=npad, mode='constant')

    if offset < 0:
        x = x[..., -offset:]
    else:
        x = pad_axis(x, (offset, 0), axis=-1)

    if x.shape[-1] < target_length:
        x = pad_axis(x, (0, target_length - x.shape[-1]), axis=-1)
    else:
        x = x[..., :target_length]

    return x


def get_white_noise_for_signal(
            time_signal,
            *,
            snr,
            rng_state: np.random.RandomState = np.random
    ):
        """
        Args:
            time_signal:
            snr: SNR or single speaker SNR.
            rng_state: A random number generator object or np.random
        """
        noise_signal = rng_state.normal(size=time_signal.shape)

        power_time_signal = np.mean(time_signal ** 2, keepdims=True)
        power_noise_signal = np.mean(noise_signal ** 2, keepdims=True)
        current_snr = 10 * np.log10(power_time_signal / power_noise_signal)

        factor = 10 ** (-(snr - current_snr) / 20)

        noise_signal *= factor
        return noise_signal


def synchronize_speech_source(original_source, offset, T):
    """
    >>> from sms_wsj.database.database import SmsWsj, AudioReader
    >>> ds = SmsWsj().get_dataset('cv_dev93')
    >>> example = ds[0]
    >>> original_source = AudioReader._rec_audio_read(
    ...     example['audio_path']['original_source'])
    >>> [s.shape for s in original_source]
    [(103650,), (93411,)]
    >>> synchronize_speech_source(
    ...     original_source,
    ...     example['offset'],
    ...     T=example['num_samples']['observation'],
    ... ).shape
    (2, 103650)
    """
    return np.array([
        extract_piece(x_, offset_, T)
        for x_, offset_ in zip(
            original_source,
            offset,
        )
    ])


def scenario_map_fn(
        example,
        *,
        snr_range: tuple = (20, 30),

        sync_speech_source=True,

        add_speech_image=True,
        add_speech_reverberation_early=True,
        add_speech_reverberation_tail=True,
        add_noise_image=True,

        early_rir_samples: int = int(8000 * 0.05),  # 50 milli seconds
        channel_slice: [None, slice, tuple, list] = None,

        details=False,
):
    """
    This will care for convolution with RIR and also generate noise.
    The random noise generator is fixed based on example ID. It will
    therefore generate the same SNR and same noise sequence the next time
    you use this DB.

    Args:
        example: Example dictionary.
        snr_range: required for noise generation
        sync_speech_source: Legacy option. The new convention is, that
            original_source is the unpadded signal, while speech_source is the
            padded signal.
            pad and/or cut the source signal to match the length of the
            observations. Considers the offset.
        add_speech_image:
            The speech_image is always computed, but when it is not needed,
            this option can reduce the memory consumption.
        add_speech_reverberation_early:
            Calculate the speech_reverberation_early signal, i.e., the speech
            source (padded original source) convolved with the early part of
            the RIR.
        add_speech_reverberation_tail:
            Calculate the speech_reverberation_tail signal, i.e., the speech
            source (padded original source) convolved with the tail part of
            the RIR.
        add_noise_image:
            If True, add the noise_image the returned example.
            This option has no effect to the computation time or the peak
            memory consumption.
        early_rir_samples:
            The number of samples that we count as the early RIR, default 50ms.
            The remaining part of the RIR we call tail.
            Note: The length of the early RIR is the time of flight plus this
            value.
        channel_slice: e.g. None (All channels), [4] (Single channel), ...
            Warning: Use this only for training. It will change the scale of
            the data and the added white noise.
            For the scale the standard deviation is estimated and the generated
            noise shape changes, hence also the values.
            With this option you can select the interested channels.
            All RIRs are used to estimate the time of flight, but only the
            interested channels are convolved with the original/speech source.
            This reduces computation time and memory consumption.

    Returns:

    """
    h = example['audio_data']['rir']  # Shape (speaker, channel, sample)

    # Estimate start sample first, to make it independent of channel_mode
    # Calculate one rir_start_sample (i.e. time of flight) for each speaker.
    rir_start_sample = np.array([get_rir_start_sample(h_k) for h_k in h])

    if channel_slice is not None:
        assert h.ndim == 3, h.shape
        h = h[:, channel_slice, :]
        assert h.ndim == 3, h.shape

    _, D, rir_length = h.shape

    # TODO: SAMPLE_RATE not defined
    # rir_stop_sample = rir_start_sample + int(SAMPLE_RATE * 0.05)
    # Use 50 milliseconds as early rir part, excluding the propagation delay
    #    (i.e. "rir_start_sample")
    assert isinstance(early_rir_samples, int), (type(early_rir_samples), early_rir_samples)
    rir_stop_sample = rir_start_sample + early_rir_samples

    log_weights = example['log_weights']

    # The two sources have to be cut to same length
    K = example['num_speakers']
    T = example['num_samples']['observation']
    if 'original_source' not in example['audio_data']:
        # legacy code
        example['audio_data']['original_source'] = example['audio_data']['speech_source']
    if 'original_source' not in example['num_samples']:
        # legacy code
        example['num_samples']['original_source'] = example['num_samples']['speech_source']
    s = example['audio_data']['original_source']

    def get_convolved_signals(h):
        assert len(s) == h.shape[0], (len(s), h.shape)
        x = [fftconvolve(s_[..., None, :], h_, axes=-1)
             for s_, h_ in zip(s, h)]

        assert len(x) == len(example['num_samples']['original_source']), (len(x), len(example['num_samples']['original_source']))
        for x_, T_ in zip(x, example['num_samples']['original_source']):
            assert x_.shape == (D, T_ + rir_length - 1), (
                x_.shape, D, T_ + rir_length - 1)

        # This is Jahn's heuristic to be able to still use WSJ alignments.
        offset = [
            offset_ - rir_start_sample_
            for offset_, rir_start_sample_ in zip(
                example['offset'], rir_start_sample)
        ]

        assert len(x) == len(offset)
        x = [extract_piece(x_, offset_, T) for x_, offset_ in zip(x, offset)]
        x = np.stack(x, axis=0)
        assert x.shape == (K, D, T), x.shape
        return x

    x = get_convolved_signals(h)

    # Note: scale depends on channel mode
    std = np.maximum(
        np.std(x, axis=(-2, -1), keepdims=True),
        np.finfo(x.dtype).tiny,
    )

    # Rescale such that invasive SIR is as close as possible to `log_weights`.
    scale = (10 ** (np.asarray(log_weights)[:, None, None] / 20)) / std
    # divide by 71 to ensure that all values are between -1 and 1
    scale /= 71

    x *= scale
    if add_speech_image:
        example['audio_data']['speech_image'] = x

    clean_mix = np.sum(x, axis=0)
    del x  # Reduce memory consumption for the case of `not add_speech_image`

    if add_speech_reverberation_early:
        h_early = h.copy()
        # Replace this with advanced indexing
        for i in range(h_early.shape[0]):
            h_early[i, ..., rir_stop_sample[i]:] = 0
        x_early = get_convolved_signals(h_early)
        x_early *= scale
        example['audio_data']['speech_reverberation_early'] = x_early

        if details:
            example['audio_data']['rir_early'] = h_early

    if add_speech_reverberation_tail:
        h_tail = h.copy()
        for i in range(h_tail.shape[0]):
            h_tail[i, ..., :rir_stop_sample[i]] = 0
        x_tail = get_convolved_signals(h_tail)
        x_tail *= scale
        example['audio_data']['speech_reverberation_tail'] = x_tail

        if details:
            example['audio_data']['rir_tail'] = h_tail

    if sync_speech_source:
        example['audio_data']['speech_source'] = synchronize_speech_source(
            example['audio_data']['original_source'],
            offset=example['offset'],
            T=T,
        )
    else:
        # legacy code
        example['audio_data']['speech_source'] = \
            example['audio_data']['original_source']

    rng = _example_id_to_rng(example['example_id'])
    snr = rng.uniform(*snr_range)
    example["snr"] = snr

    rng = _example_id_to_rng(example['example_id'])

    n = get_white_noise_for_signal(clean_mix, snr=snr, rng_state=rng)
    if add_noise_image:
        example['audio_data']['noise_image'] = n

    observation = clean_mix
    observation += n  # Inplace to reduce memory consumption
    example['audio_data']['observation'] = observation

    return example


def get_valid_mird_rirs(mird_dir, rng=np.random):
    import scipy.io

    def minus_with_wrap(angle1, angle2):
        return np.angle(np.exp(1j * (angle1 - angle2)))

    K = 2
    t60 = rng.choice(['0.160', '0.360', '0.610'])
    spacing = rng.choice(['3-3-3-8-3-3-3', '4-4-4-8-4-4-4', '8-8-8-8-8-8-8'])
    distance = rng.choice(['1', '2'], size=2, replace=True)

    angular_distance_ok = False
    while not angular_distance_ok:
        angle_degree = rng.choice([
            '000',
            '015', '030', '045', '060', '075', '090',
            '270', '285', '300', '315', '330', '345'
        ], size=2, replace=False)
        angular_distance = np.abs(minus_with_wrap(
            float(angle_degree[1]) / 180 * np.pi,
            float(angle_degree[0]) / 180 * np.pi,
        ) / np.pi * 180)
        if angular_distance > 37.5:
            angular_distance_ok = True

    rirs = np.stack([
        scipy.io.loadmat(str(
            mird_dir /
            f'Impulse_response_Acoustic_Lab_Bar-Ilan_University_('
            f'Reverberation_{t60}s)_{spacing}_{distance[k]}m_'
            f'{angle_degree[k]}.mat'
        ))['impulse_response'].T
        for k in range(K)
    ])

    return scipy.signal.resample_poly(rirs, up=1, down=6, axis=-1)
