import subprocess
from pathlib import Path

import numpy as np

from nt.database import JsonDatabase
from nt.database import HybridASRKaldiDatabaseTemplate
from nt.io.data_dir import kaldi_root
from nt.io.data_dir import database_jsons
from nt.io import audioread

JSON_PATH = database_jsons / "wsj_bss.json"
# JSON_PATH = Path(
#     '/net/vol/ldrude/projects/2017/project_dc_storage/wsj_bss.json'
# )


class WsjBss(JsonDatabase):
    rate_in_audio = 16000
    rate_in_rir = 8000
    rate_out = 8000
    assert rate_in_audio % rate_out == 0
    factor = rate_in_audio // rate_out

    # TODO: Implement `build_select_channels_map_fn`? See KaldiHybridASRProvider
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

    @property
    def read_fn(self):
        def fn(x):
            x, sample_rate = audioread.audioread(x)

            if x.ndim == 1 and sample_rate == self.rate_in_audio:
                x = x.astype(np.float32)
                command = (
                    f'sox -N -V1 -t f32 -r {self.rate_in_audio} -c 1 - '
                    f'-t f32 -r {self.rate_out} -c 1 -'
                )
                process = subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    input=x.tobytes(order="f")
                )
                x_resample = np.fromstring(process.stdout, dtype=np.float32)
                assert x_resample.size > 0, (
                    'The command did not yield any output:\n'
                    f'x.shape: {x.shape}\n'
                    f'x_resampled.shape: {x_resample.shape}\n'
                    f'command: {command}\n'
                    f'stdout: {process.stdout.decode()}\n'
                    f'stderr: {process.stderr.decode()}\n'
                )
                return x_resample
            if x.ndim == 2 and sample_rate == self.rate_in_rir == self.rate_out:
                return x
            else:
                raise RuntimeError(
                    f'Unexpected file found: {x}\n'
                    f'x.shape: {x.shape}\n'
                    f'sample_rate: {sample_rate}\n'
                    f'self.rate_in: {self.rate_in_audio}\n'
                    f'self.rate_in: {self.rate_in_rir}\n'
                    f'self.rate_out: {self.rate_out}\n'
                )

        return fn


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
