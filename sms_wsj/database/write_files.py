"""
Example calls:
python -m paderbox.database.wsj.write_wav --dst-dir /destination/dir --json-path /path/to/sms_wsj.json --write-all

mpiexec -np 20 python -m paderbox.database.wsj.write_wav --dst-dir /destination/dir --json-path /path/to/sms_wsj.json --write-all

"""
import logging
import time
from functools import partial
from pathlib import Path

import click
import numpy as np
import soundfile
from paderbox.database import JsonDatabase
from paderbox.database import keys as K
from paderbox.database.helper import click_convert_to_path
from paderbox.database.helper import  dump_database_as_json
from paderbox.database.iterator import AudioReader
from paderbox.database.wsj_bss.database import scenario_map_fn
from paderbox.utils.mpi import RANK, SIZE, IS_MASTER
from tqdm import tqdm

type_mapper = {K.SPEECH_SOURCE: 'clean',
               K.SPEECH_REVERBERATION_DIRECT: 'early',
               K.SPEECH_REVERBERATION_TAIL: 'tail',
               K.NOISE_IMAGE: 'noise',
               K.OBSERVATION: 'observation'}
appendix_mapper = {K.SPEECH_SOURCE: ['_0', '_1'],
                   K.SPEECH_REVERBERATION_DIRECT: ['_0', '_1'],
                   K.SPEECH_REVERBERATION_TAIL: ['_0', '_1'],
                   K.NOISE_IMAGE: [''],
                   K.OBSERVATION: ['']}
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


def write_wavs(dst_dir, ds, write_all=False):
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
    ds = ds.map(AudioReader(audio_keys=[K.RIR, K.SPEECH_SOURCE])).map(map_fn)

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
            for idx, signal in enumerate(value):
                filename = example_id + appendix_mapper[key][idx] + '.wav'
                audio_path = str(path / filename)
                with soundfile.SoundFile(
                        audio_path, subtype='FLOAT', mode='w', samplerate=8000,
                        channels=1 if signal.ndim == 1 else signal.shape[0]
                ) as f:
                    f.write(signal.T)
    if IS_MASTER:
        created_files = list(dst_dir.rglob("*.wav"))
        logging.info(f"Written {len(created_files)} wav files.")
        if write_all:
            assert len(created_files) == (3 * 2 + 2) * len(ds), len(
                created_files)
        else:
            assert len(created_files) == len(ds), len(created_files)


def create_json(dst_dir, ds, write_all):
    json_dict = dict()

    for ex in ds:
        ex_id = ex[K.EXAMPLE_ID]
        if write_all:
            ex[K.AUDIO_PATH].update({
                key: dst_dir / data_type / (ex_id + appendix + '.wav')
                for key, data_type in type_mapper
                for appendix in appendix_mapper[key]
            })
        else:
            ex[K.AUDIO_PATH].update({
                'observation': dst_dir / 'observation' / (ex_id + '.wav')
            })
        del ex[ex_id]
        json_dict[ex_id] = ex
    return json_dict


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
    @click.option(
        '--write-all',
        is_flag=True,
        help='Flag indicating whether to write everything to dst_dir or '
             'just the observation'
    )
    @click.option(
        '--overwrite-json',
        is_flag=True,
        help='Flag indication whether to overwrite the old json with a new '
             'one with updated paths'
    )
    def main(dst_dir, json_path, write_all, overwrite_json):
        logging.info(f"Start - {time.ctime()}")

        dst_dir = Path(dst_dir).expanduser().resolve()
        assert dst_dir.is_dir(), dst_dir
        json_path = Path(json_path).expanduser().resolve()
        assert json_path.is_file(), json_path

        db = JsonDatabase(json_path)
        ds = db.get_dataset(['train_si284', 'cv_dev93', 'test_eval92'])
        write_wavs(dst_dir, json_path, write_all=write_all)

        if overwrite_json and IS_MASTER:
            print(f'Creating a new json and saving it to {json_path}')
            new_json = create_json(dst_dir, ds, write_all)
            new_json[K.DATASETS] = db.database_dict[K.DATASETS]
            dump_database_as_json(json_path, new_json)

        logging.info(f"Done - {time.ctime()}")


    main()
