"""
Example calls:
python -m paderbox.database.wsj.write_wav --dst-dir /destination/dir --json-path /path/to/sms_wsj.json --write-all

mpiexec -np 20 python -m paderbox.database.wsj.write_wav --dst-dir /destination/dir --json-path /path/to/sms_wsj.json --write-all

"""
from paderbox.database import JsonDatabase
import click
import logging
from pathlib import Path
from paderbox.database import keys as K
from paderbox.database.helper import click_convert_to_path
from paderbox.database.wsj_bss.database import scenario_map_fn
from paderbox.database.iterator import AudioReader
from tqdm import tqdm
from paderbox.utils.mpi import COMM, RANK, SIZE, MASTER, IS_MASTER, bcast, barrier
import soundfile
import time
from functools import partial
import numpy as np

type_mapper = {K.SPEECH_SOURCE: 'clean',
               K.SPEECH_REVERBERATION_DIRECT: 'early',
               K.SPEECH_REVERBERATION_TAIL: 'tail',
               K.NOISE_IMAGE: 'noise',
               K.OBSERVATION: 'observation'}
snr_range = (20, 30)

def normalize_audio(d: dict):
    """
    Convert all arrays in the dict to np.float32.
    To have a low quantisation error, tha maximum of all audios is calculated
    and used as normalisation.

    >>> np.float64(np.finfo(np.float32).max)
    32767.0
    >>> normalize_audio({'a': [2., 3.], 'b': [4., 5.]})
    {'a': array([13106, 19660], dtype=int16), 'b': array([26213, 32767], dtype=int16)}

    """

    dtype = np.float32
    float32_max = np.float64(np.finfo(dtype).max)

    correction = float32_max / np.max([np.max(np.abs(v)) for v in d.values()])

    return {
        k: np.array(correction * np.array(v), dtype)
        for k, v in d.items()
    }


def write_wavs(dst_dir, json_path, write_all=False):


    db = JsonDatabase(json_path)
    if write_all:
        if IS_MASTER:
            [(dst_dir / data_type).mkdir(exist_ok=False)
             for data_type in type_mapper.values()]
        map_fn = partial(
            scenario_map_fn, snr_range=snr_range,
            sync_speech_source=True,
            add_speech_reverberation_direct=True,
            add_speech_reverberation_tail=True
        )
    else:
        if IS_MASTER:
            (dst_dir / 'observation').mkdir(exist_ok=False)
        map_fn = partial(
            scenario_map_fn, snr_range=snr_range,
            sync_speech_source=True,
            add_speech_reverberation_direct=False,
            add_speech_reverberation_tail=False
        )
    ds = db.get_dataset(['train_si284', 'cv_dev93', 'test_eval92']).\
        map(AudioReader(audio_keys=[K.RIR, K.SPEECH_SOURCE])).map(map_fn)

    for example in tqdm(ds[RANK::SIZE], disable=not IS_MASTER):
        audio_dict = example[K.AUDIO_DATA]
        example_id = example[K.EXAMPLE_ID]
        del audio_dict[K.SPEECH_IMAGE]
        del audio_dict[K.RIR]
        if not write_all:
            del audio_dict[K.SPEECH_SOURCE]
            del audio_dict[K.NOISE_IMAGE]
        audio_dict = normalize_audio(audio_dict)
        for key, value in audio_dict.items():
            path = dst_dir / type_mapper[key]
            if key in [K.OBSERVATION, K.NOISE_IMAGE]:
                value = value[None]
                appendix = ['']
            elif key in [K.SPEECH_SOURCE, K.SPEECH_REVERBERATION_TAIL, K.SPEECH_REVERBERATION_DIRECT]:
                appendix = ['_0', '_1']
            else:
                raise ValueError('Unexpected key in audio dict', key)
            for idx, speaker in enumerate(value):
                signal = value[idx]
                audio_path = str(path / (example_id + appendix[idx] + '.wav'))
                with soundfile.SoundFile(
                        audio_path, subtype='FLOAT', mode='w', samplerate=8000,
                        channels=1 if signal.ndim == 1 else signal.shape[0]
                ) as f:
                    f.write(signal.T)
    if IS_MASTER:
        created_files = list(dst_dir.rglob("*.wav"))
        logging.info(f"Written {len(created_files)} wav files.")
        if write_all:
            assert len(created_files) == (3 * 2 + 2) * len(ds), len(created_files)
        else:
            assert len(created_files) == len(ds), len(created_files)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.INFO
    )

    @click.command()
    @click.option(
        '-d', '--dst-dir',
        help="Directory which will store the converted WSJ wav files",
        type=click.Path(writable=True)
    )
    @click.option(
        '--json-path', '-j',
        help=f'Path to sms_wsj.json',
        type=click.Path(dir_okay=False),
        callback=click_convert_to_path,
    )
    @click.option('--write-all',
                  is_flag=True,
                  help='Flag indicating whether to write everything to'
                       ' dst_dir or just the observation')
    def main(dst_dir, json_path, write_all):
        logging.info(f"Start - {time.ctime()}")

        dst_dir = Path(dst_dir).expanduser().resolve()
        assert dst_dir.is_dir(), dst_dir
        json_path = Path(json_path).expanduser().resolve()

        write_wavs(dst_dir, json_path, write_all=write_all)
        logging.info(f"Done - {time.ctime()}")

    main()