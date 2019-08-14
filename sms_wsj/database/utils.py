from pathlib import Path

import numpy as np

from paderbox.database import JsonDatabase
from paderbox.database.database import HybridASRKaldiDatabaseTemplate
from paderbox.database.database import HybridASRJSONDatabaseTemplate
from paderbox.io.data_dir import kaldi_root
from paderbox.io.data_dir import database_jsons
from paderbox.database.keys import *
import hashlib
from paderbox.utils.numpy_utils import pad_axis
from hashlib import md5
from paderbox.reverb.reverb_utils import convolve
from paderbox.speech_enhancement.noise.generator import NoiseGeneratorWhite
from paderbox.reverb.reverb_utils import get_rir_start_sample

JSON_PATH = database_jsons / "wsj_bss.json"


__all__ = [
    'WsjBss',
    'WsjBssKaldiDatabase',
    'scenario_map_fn',
]


class WsjBss(HybridASRJSONDatabaseTemplate):
    """
    For the simulated database, we artificially generated 30000, 500 and 1500
    six-channel mixtures with a sampling rate of 8 kHz with source signals
    obtained from three non-overlapping WSJ sets (train: si284, develop: dev93,
    test: eval92.

    We padded or cut the second speaker to match the length of the first
    speaker. Room impulse responses were generated with the Image Method, where
    the room dimensions, the position of the circular array with radius 10 cm
    and the position of two concurring speakers were randomly sampled. The
    minimum angular distance was set to 15 degree. The reverberation time (T60)
    was uniformly sampled between 200 and 500 ms. White Gaussian noise with 20
    to 30 dB SNR was added to the mixture. We here deviated from the file lists
    provided by since the speakers of their training set and development set
    overlap and although their training set consists of 20000 mixtures it only
    includes 6842 unique utterances from si84 which turned out to be
    insufficient when training an acoustic model on that list.

    v1:
    At the time of creating v1, there was no publicly available simulated
    database for multi-channel BSS. We did not build this on `merl_mixtures`,
    because `merl_mixtures` did not have enough variability in the underlying
    speech data to train a good ASR system. This database is closer to the
    single speaker WSJ database and therefore allows to use the WSJ recipes in
    Kaldi more easily.

    There currently exists a spatialized (reverberated) version of
    `merl_mixtures` at [1].

    v2:
    In v1 the channels were normalized independently. This led to a spatial
    distortion. Now all channels are normalized with the same value. This
    implies, that you need to load all channels to have deterministic output.
    This is crucial for development and test.

    The start sample is now calculated per speaker. In v1 the propagation delay
    was calculated for all speakers and therefore the propagation delay was too
    small for the furthest speaker. Now, it is more likely that you can use
    clean speech alignments to train an ASR system on this database.

    In v1 you were able to truncate the RIR to obtain, e.g., early alignments.
    Now, v2 calculates the early image (direct) and the tail image which allows
    to even train a system to predict the early arriving speech.

    The SNR was changed from 20-30 dB to 15-25 dB to address the critique of
    a reviewer. We decided to keep it somewhat high since the noise type is
    AWGN and is not very realistic anyway.

    [1] http://www.merl.com/demos/deep-clustering
    """
    def __init__(
            self,
            json_path: [str, Path]=JSON_PATH,
            datasets_train=None,
            datasets_eval=None,
            datasets_test=None,
    ):
        super().__init__(json_path=json_path, lfr=False)

        if datasets_train is None:
            self._datasets_train = 'train_si284'
        else:
            self._datasets_train = datasets_train

        if datasets_eval is None:
            self._datasets_eval = 'cv_dev93'
        else:
            self._datasets_eval = datasets_eval

        if datasets_test is None:
            self._datasets_test = 'test_eval92'
        else:
            self._datasets_test = datasets_test

    @property
    def datasets_train(self):
        return self._datasets_train

    @property
    def datasets_eval(self):
        return self._datasets_eval

    @property
    def datasets_test(self):
        return self._datasets_test

    @property
    def lang_path(self):
        return kaldi_root / 'egs' / 'wsj_8k' / 's5' / 'data' / 'lang'

    @property
    def hclg_path_ffr(self):
        """Path to HCLG directory created by Kaldi."""
        return kaldi_root / 'egs' / 'wsj_8k' / 's5' / 'exp' \
            / 'tri4b' / 'graph_tgpr'

    @property
    def hclg_path_lfr(self):
        """Path to HCLG directory created by Kaldi."""
        return kaldi_root / 'egs' / 'wsj_8k' / 's5' / 'exp' \
            / 'tri4b_ali_si284_lfr' / 'graph_tgpr'

    @property
    def ali_path_train_ffr(self):
        """Path containing the kaldi alignments for train data."""
        return kaldi_root / 'egs' / 'wsj_8k' / 's5' / 'exp' / 'tri4b_ali_si284'

    @property
    def ali_path_eval_ffr(self):
        """Path containing the kaldi alignments for dev data."""
        return kaldi_root / 'egs' / 'wsj_8k' / 's5' / 'exp' / 'tri4b_ali_test_dev93'

    @property
    def example_id_map_fn(self):
        # provides a function to map example_id to kaldi wsj_8kHz ids
        def _map_example_id(example):
            example_id = example[EXAMPLE_ID].split('_')[0]
            return example_id

        return _map_example_id


class WsjBssKaldiDatabase(HybridASRKaldiDatabaseTemplate):
    rate_in = 8000
    rate_out = 8000

    """Used after separation of the signals for ASR."""
    def __init__(
            self,
            egs_path: Path,
            ali_path: Path=None,
            lfr=False,
            datasets_train=None,
            datasets_eval=None,
            datasets_test=None
    ):
        super().__init__(egs_path=egs_path)
        self._ali_path = Path(ali_path) if ali_path is not None else egs_path
        assert lfr is False, lfr

        if datasets_train is None:
            self._datasets_train = 'train_si284'
        else:
            self._datasets_train = datasets_train

        if datasets_eval is None:
            self._datasets_eval = 'cv_dev93'
        else:
            self._datasets_eval = datasets_eval

        if datasets_test is None:
            self._datasets_test = 'test_eval92'
        else:
            self._datasets_test = datasets_test

    @property
    def datasets_train(self):
        return self._datasets_train

    @property
    def datasets_eval(self):
        return self._datasets_eval

    @property
    def datasets_test(self):
        return self._datasets_test

    @property
    def lang_path(self):
        return kaldi_root / 'egs' / 'wsj_8k' / 's5' / 'data' / 'lang'

    @property
    def hclg_path_ffr(self):
        """Path to HCLG directory created by Kaldi."""
        return kaldi_root / 'egs' / 'wsj_8k' / 's5' / 'exp' \
            / 'tri4b' / 'graph_tgpr'

    @property
    def hclg_path_lfr(self):
        """Path to HCLG directory created by Kaldi."""
        return kaldi_root / 'egs' / 'wsj_8k' / 's5' / 'exp' \
            / 'tri4b_ali_si284_lfr' / 'graph_tgpr'

    @property
    def ali_path_train_ffr(self):
        """Path containing the kaldi alignments for train data."""
        return self._ali_path / 'kaldi_align' / 'train_si284'

    @property
    def ali_path_eval_ffr(self):
        """Path containing the kaldi alignments for dev data."""
        return self._ali_path / 'kaldi_align' / 'cv_dev93'

    def get_lengths(self, datasets, length_transform_fn=None):
        if length_transform_fn is None:

            def length_transform_fn(duration):
                return self.rate_in * duration

        return super().get_lengths(datasets, length_transform_fn)


def get_rng(dataset, example_id):
    string = f"{dataset}_{example_id}"
    seed = (
        int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 2 ** 32
    )
    return np.random.RandomState(seed=seed)


def _example_id_to_rng(example_id):
    hash_value = md5(example_id.encode())
    hash_value = int(hash_value.hexdigest(), 16)
    hash_value = hash_value % 2 ** 32 - 1
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
    if offset < 0:
        x = x[..., -offset:]
    else:
        x = pad_axis(x, (offset, 0), axis=-1)

    if x.shape[-1] < target_length:
        x = pad_axis(x, (0, target_length - x.shape[-1]), axis=-1)
    else:
        x = x[..., :target_length]

    return x


def scenario_map_fn(
        example,
        *,
        snr_range: tuple,

        sync_speech_source=True,
        add_speech_reverberation_direct=True,
        add_speech_reverberation_tail=True,
):
    """
    This will care for convolution with RIR and also generate noise.
    The random noise generator is fixed based on example ID. It will
    therefore generate the same SNR and same noise sequence the next time
    you use this DB.

    Args:
        example: Example dictionary.
        snr_range: Lukas used (20, 30) here.
            This will make your reviewer angry.
        sync_speech_source: pad and/or cut the source signal to match the
            length of the observations. Considers the offset.
        add_speech_reverberation_direct:
            Calculate the speech_reverberation_direct signal.
        add_speech_reverberation_tail:
            Calculate the speech_reverberation_tail signal.

    Returns:

    >>> import functools
    >>> import paderbox as pb
    >>> dataset = 'cv_dev93'
    >>> db = pb.database.wsj_bss.WsjBss()
    >>> ds = db.get_iterator_by_names(dataset)
    >>> ds = ds.map(pb.database.iterator.AudioReader(
    ...     audio_keys=['speech_source', 'rir'],
    ...     read_fn=db.read_fn,
    ... ))
    >>> ds = ds.map(functools.partial(
    ...     pb.database.wsj_bss.scenario_map_fn,
    ...     snr_range=(20, 30),  # Too high, reviewer won't like this
    ...     add_speech_reverberation_direct=True,
    ...     add_speech_reverberation_tail=True,
    ...     sync_speech_source=True,
    ... ))
    >>> example = ds[0]
    >>> pb.notebook.pprint(example)
    {'audio_path': {'rir': ['/net/fastdb/wsj_bss/cv_dev93/0/h_0.wav',
       '/net/fastdb/wsj_bss/cv_dev93/0/h_1.wav'],
      'speech_source': ['/net/fastdb/wsj_8k/13-16.1/wsj1/si_dt_20/4k0/4k0c0301.wav',
       '/net/fastdb/wsj_8k/13-16.1/wsj1/si_dt_20/4k6/4k6c030t.wav']},
     'dataset': 'cv_dev93',
     'example_id': '4k0c0301_4k6c030t_0',
     'gender': ['male', 'male'],
     'kaldi_transcription': ['SAATCHI OFFICIALS SAID THE MANAGEMENT RE:STRUCTURING MIGHT ACCELERATE ITS EFFORTS TO PERSUADE CLIENTS TO USE THE FIRM AS A ONE STOP SHOP FOR BUSINESS SERVICES',
      "THEY HAVE SPENT SEVEN YEARS AND MORE THAN THREE HUNDRED MILLION DOLLARS IN U. S. AID BUILDING THE AREA'S BIGGEST INSURGENT FORCE"],
     'log_weights': [1.2027951449295022, -1.2027951449295022],
     'num_samples': {'observation': 103650, 'speech_source': [103650, 56335]},
     'num_speakers': 2,
     'offset': [0, 17423],
     'room_dimensions': [[8.169], [5.905], [3.073]],
     'sensor_position': [[3.899, 3.8, 3.759, 3.817, 3.916, 3.957],
      [3.199, 3.189, 3.098, 3.017, 3.027, 3.118],
      [1.413, 1.418, 1.423, 1.423, 1.417, 1.413]],
     'sound_decay_time': 0.354,
     'source_id': ['4k0c0301', '4k6c030t'],
     'source_position': [[2.443, 2.71], [3.104, 2.135], [1.557, 1.557]],
     'speaker_id': ['4k0', '4k6'],
     'audio_data': {'speech_source': ndarray(shape=(2, 103650), dtype=float64),
      'rir': ndarray(shape=(2, 6, 8192), dtype=float64),
      'speech_image': ndarray(shape=(2, 6, 103650), dtype=float64),
      'speech_reverberation_direct': ndarray(shape=(2, 6, 103650), dtype=float64),
      'speech_reverberation_tail': ndarray(shape=(2, 6, 103650), dtype=float64),
      'noise_image': ndarray(shape=(6, 103650), dtype=float64),
      'observation': ndarray(shape=(6, 103650), dtype=float64)},
     'snr': 29.749852569493584}
    >>> speech_image = example['audio_data']['speech_reverberation_direct'] + example['audio_data']['speech_reverberation_tail']
    >>> np.testing.assert_allclose(speech_image, example['audio_data']['speech_image'], atol=1e-10)
    >>> 10 * np.log10(np.mean(example['audio_data']['speech_image']**2, axis=(-2, -1)))
    array([ 1.20279517, -1.20279514])
    """
    h = example[AUDIO_DATA][RIR]  # Shape (K, D, T)

    # Estimate start sample first, to make it independent of channel_mode
    rir_start_sample = np.array([get_rir_start_sample(h_k) for h_k in h])

    _, D, rir_length = h.shape

    # TODO: SAMPLE_RATE not defined
    # rir_stop_sample = rir_start_sample + int(SAMPLE_RATE * 0.05)
    # Use 50 milliseconds as early rir part, excluding the propagation delay
    #    (i.e. "rir_start_sample")
    rir_stop_sample = rir_start_sample + int(8000 * 0.05)

    log_weights = example[LOG_WEIGHTS]

    # The two sources have to be cut to same length
    K = example[NUM_SPEAKERS]
    T = example[NUM_SAMPLES][OBSERVATION]
    s = example[AUDIO_DATA][SPEECH_SOURCE]

    def get_convolved_signals(h):
        x = [convolve(s_, h_, truncate=False) for s_, h_ in zip(s, h)]

        for x_, T_ in zip(x, example[NUM_SAMPLES][SPEECH_SOURCE]):
            assert x_.shape == (D, T_ + rir_length - 1), (x_.shape, D, T_ + rir_length - 1)

        # This is Jahn's heuristic to be able to still use WSJ alignments.
        offset = [
            offset_ - rir_start_sample_
            for offset_, rir_start_sample_ in zip(example['offset'], rir_start_sample)
        ]

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

    x *= scale
    example[AUDIO_DATA][SPEECH_IMAGE] = x

    if add_speech_reverberation_direct:
        h_early = h.copy()
        # Replace this with advanced indexing
        for i, h_k in enumerate(h_early):
            h_early[..., rir_stop_sample[i]:] = 0
        x_early = get_convolved_signals(h_early)
        x_early *= scale
        example[AUDIO_DATA][SPEECH_REVERBERATION_DIRECT] = x_early

    if add_speech_reverberation_tail:
        h_tail = h.copy()
        for i, h_k in enumerate(h_tail):
            h_tail[..., :rir_stop_sample[i]] = 0
        x_tail = get_convolved_signals(h_tail)
        x_tail *= scale
        example[AUDIO_DATA][SPEECH_REVERBERATION_TAIL] = x_tail

    if sync_speech_source:
        example[AUDIO_DATA][SPEECH_SOURCE] = np.array([
            extract_piece(x_, offset_, T)
            for x_, offset_ in zip(
                example[AUDIO_DATA][SPEECH_SOURCE],
                example['offset'],
            )
        ])

    clean_mix = np.sum(x, axis=0)

    rng = _example_id_to_rng(example[EXAMPLE_ID])
    snr = rng.uniform(*snr_range)
    example["snr"] = snr

    rng = _example_id_to_rng(example[EXAMPLE_ID])
    ng = NoiseGeneratorWhite()

    # Should the SNR be defined of "reverberated vs noise" or
    # "early reverberated vs noise"?
    n = ng.get_noise_for_signal(clean_mix, snr=snr, rng_state=rng)
    example[AUDIO_DATA][NOISE_IMAGE] = n
    example[AUDIO_DATA][OBSERVATION] = clean_mix + n
    return example


def get_valid_mird_rirs(rng=np.random):
    from paderbox.math import directional
    import scipy.io
    from paderbox.io.data_dir import mird as mird_path

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
        angular_distance = np.abs(directional.rad_to_deg(directional.minus(
            directional.deg_to_rad(float(angle_degree[1])),
            directional.deg_to_rad(float(angle_degree[0])),
        )))
        if angular_distance > 37.5:
            angular_distance_ok = True

    rirs = np.stack([
        scipy.io.loadmat(str(
            mird_path /
            f'Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_{t60}s)_{spacing}_{distance[k]}m_{angle_degree[k]}.mat'
        ))['impulse_response'].T
        for k in range(K)
    ])

    return scipy.signal.resample_poly(rirs, up=1, down=6, axis=-1)
