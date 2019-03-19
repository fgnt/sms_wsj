import subprocess
from pathlib import Path

import numpy as np

from paderbox.database import JsonDatabase
from paderbox.database.database import HybridASRKaldiDatabaseTemplate
from paderbox.io.data_dir import kaldi_root
from paderbox.io.data_dir import database_jsons
from paderbox.io import audioread
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


class WsjBss(JsonDatabase):
    def __init__(
            self,
            json_path: [str, Path]=JSON_PATH,
            datasets_train=None,
            datasets_eval=None,
            datasets_test=None,
    ):
        super().__init__(json_path=json_path)

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
        mode=None,
        channel_mode='all',
        truncate_rir=False,
        snr_range: tuple,
        rir_type='image_method'
):
    """
    This will care for convolution with RIR and also generate noise.
    The random noise generator is fixed based on example ID. It will
    therefore generate the same SNR and same noise sequence the next time
    you use this DB.

    Args:
        example: Example dictionary.
        mode: 'train', 'eval', or 'predict', is deprecated. You can use an
            external code to do this. It is here for legacy code.
        channel_mode: Is deprecated. You can use an
            external code to do this. It is here for legacy code.
        truncate_rir:
        snr_range: Lukas used (20, 30) here.
            This will make your reviewer angry.
        rir_type:
            image_method: Will be the provided pre-generated RIRs.
            mird: Will be random selections from the MIRD database.

    Returns:

    """
    if rir_type == 'image_method':
        h = example[AUDIO_DATA][RIR]  # Shape (K, D, T)
    elif rir_type == 'mird':
        # TODO: Meta information is wrong, since we do not load MIRD meta info
        # TODO: It is also a shame, that we load the wrong RIRs first.
        h = get_valid_mird_rirs(rng=get_rng('', example[EXAMPLE_ID]))
    else:
        raise ValueError(rir_type)

    # Estimate start sample first, to make it independent of channel_mode
    rir_start_sample = get_rir_start_sample(h)

    if isinstance(channel_mode, dict):
        channel_mode = channel_mode[mode]

    if channel_mode == "deterministic":
        channels = [0]
        h = h[:, channels, :]
    elif channel_mode == "random":
        channels = np.random.randint(0, h.shape[1], size=(1,))
        h = h[:, channels, :]
    elif channel_mode == "all":
        pass
    elif channel_mode == "deterministic_2":
        channels = [0, 1]
        h = h[:, channels, :]
    elif channel_mode == "random_2":
        channels = np.random.randint(0, h.shape[1], size=(2,))
        h = h[:, channels, :]
    else:
        raise ValueError(channel_mode[mode])
    _, D, rir_length = h.shape

    if truncate_rir:
        # TODO: SAMPLE_RATE not defined
        # rir_stop_sample = rir_start_sample + int(SAMPLE_RATE * 0.05)
        rir_stop_sample = rir_start_sample + int(8000 * 0.05)

        h[..., rir_stop_sample:] = 0
    # print(f'h {h.shape}')

    log_weights = example[LOG_WEIGHTS]

    # The two sources have to be cut to same length
    K = example[NUM_SPEAKERS]
    T = example[NUM_SAMPLES][OBSERVATION]
    s = example[AUDIO_DATA][SPEECH_SOURCE]

    x = [convolve(s_, h_, truncate=False) for s_, h_ in zip(s, h)]

    for x_, T_ in zip(x, example[NUM_SAMPLES][SPEECH_SOURCE]):
        assert x_.shape == (D, T_ + rir_length - 1)
    # print(f'x_ {x_.shape}')

    # This is Jahn's heuristic to be able to still use WSJ alignments.
    offset = [offset_ - rir_start_sample for offset_ in example['offset']]

    x = [extract_piece(x_, offset_, T) for x_, offset_ in zip(x, offset)]
    x = np.stack(x, axis=0)
    assert x.shape == (K, D, T), x.shape

    x /= np.maximum(np.std(x, axis=-1, keepdims=True), np.finfo(x.dtype).tiny)
    x *= 10 ** (np.asarray(log_weights)[:, None, None] / 20)

    example[AUDIO_DATA][SPEECH_IMAGE] = x
    clean_mix = np.sum(x, axis=0)

    rng = _example_id_to_rng(example[EXAMPLE_ID])
    snr = rng.uniform(*snr_range)
    example["snr"] = snr

    # TODO: Maybe does not yield same noise depending on `channel_mode`
    rng = _example_id_to_rng(example[EXAMPLE_ID])
    ng = NoiseGeneratorWhite()
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
