"""
Example calls:
python -m paderbox.database.wsj.write_wav --dst-dir /destination/dir --json-path /path/to/sms_wsj.json --write-all

mpiexec -np 20 python -m paderbox.database.wsj.write_wav --dst-dir /destination/dir --json-path /path/to/sms_wsj.json --write-all

"""
import logging
import time
from functools import partial
from pathlib import Path

import json
import click
import numpy as np
import soundfile
from lazy_dataset.database import JsonDatabase
from paderbox.database.helper import click_convert_to_path
from sms_wsj.database.utils import scenario_map_fn
import dlp_mpi

type_mapper = {'speech_reverberation_early': 'early',
               'speech_reverberation_tail': 'tail',
               'noise_image': 'noise',
               'observation': 'observation'}


def audio_read(example):
        """
        :param example: example dict
        :return: example dict with audio_data added
        """
        audio_keys = ['rir', 'speech_source']
        keys = list(example['audio_path'].keys())
        for audio_key in audio_keys:
            assert audio_key in keys, (
                f'Trying to read {audio_key} but only {keys} are available'
            )
            audio_data = list()
            for wav_file in example['audio_path'][audio_key]:

                with soundfile.SoundFile(wav_file, mode='r') as f:
                    audio_data.append(f.read())
            example['audio_data'][audio_key] = np.array(audio_data)
        return example


def write_wavs(dst_dir, db, write_all=False):
    if write_all:
        if dlp_mpi.IS_MASTER:
            [(dst_dir / data_type).mkdir(exist_ok=False)
             for data_type in type_mapper.values()]
        map_fn = partial(
            scenario_map_fn,
            sync_speech_source=True,
            add_speech_reverberation_direct=True,
            add_speech_reverberation_tail=True
        )
    else:
        if dlp_mpi.IS_MASTER:
            (dst_dir / 'observation').mkdir(exist_ok=False)
        map_fn = partial(
            scenario_map_fn,
            sync_speech_source=True,
            add_speech_reverberation_direct=False,
            add_speech_reverberation_tail=False
        )
    ds = db.get_dataset(['train_si284', 'cv_dev93', 'test_eval92']).map(
        audio_read).map(map_fn)

    for example in dlp_mpi.split_managed(ds):
        audio_dict = example['audio_data']
        example_id = example['example_id']
        del audio_dict['speech_image']
        del audio_dict['rir']
        del audio_dict['speech_source']
        if not write_all:
            del audio_dict['speech_reverberation_early']
            del audio_dict['speech_reverberation_tale']
            del audio_dict['noise_image']
        assert all([np.max(np.abs(v)) <= 1 for v in audio_dict.values()])
        for key, value in audio_dict.items():
            path = dst_dir / type_mapper[key]
            if key in ['observation', 'noise_image']:
                value = value[None]
            for idx, signal in enumerate(value):
                appendix = f'_{idx}' if len(value) > 1 else ''
                filename = example_id + appendix + '.wav'
                audio_path = str(path / filename)
                with soundfile.SoundFile(
                        audio_path, subtype='FLOAT', mode='w', samplerate=8000,
                        channels=1 if signal.ndim == 1 else signal.shape[0]
                ) as f: f.write(signal.T)

    dlp_mpi.barrier()
    if dlp_mpi.IS_MASTER:
        created_files = list(dst_dir.rglob("*.wav"))
        logging.info(f"Written {len(created_files)} wav files.")
        if write_all:
            assert len(created_files) == (2 * 2 + 2) * len(ds), len(
                created_files)
        else:
            assert len(created_files) == len(ds), len(created_files)


def create_json(dst_dir, db, write_all):
    json_dict = dict(datasets=dict())
    database_dict = db.database_dict['datasets']
    for dataset_name, dataset in database_dict.items():
        dataset_dict = dict()
        for ex_id, ex in dataset.items():
            if write_all:
                for key, data_type in type_mapper.items():
                    if key in ['observation', 'noise_image']:
                        ex['audio_path'][key] = [
                            dst_dir / data_type / (ex_id + '.wav'),
                         ]
                    else:
                        ex['audio_path'][key] = [
                            dst_dir / data_type / (ex_id + '_0.wav'),
                            dst_dir / data_type / (ex_id + '_1.wav')
                        ]
            else:
                ex['audio_path'].update({
                    'observation': dst_dir / 'observation' / (ex_id + '.wav')
                })
            dataset_dict[ex_id] = ex
            json_dict['datasets'][dataset_name] = dataset_dict
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
        if not any([(dst_dir / data_type).exists() for data_type in type_mapper.keys()]):
            write_wavs(dst_dir, db, write_all=write_all)
        else:
            num_wav_files = len(list(dst_dir.rglob("*.wav")))
            if write_all and  num_wav_files == (2 * 2 + 2) * 32000:
                print('Wav files seem to exist. They are not overwritten.')
            elif not write_all and num_wav_files == 32000:
                print('Wav files seem to exist. They are not overwritten.')
            else:
                raise ValueError(
                    'Not all wav files exist. However, the directory structure'
                    ' already exists.')

        if dlp_mpi.IS_MASTER and overwrite_json:
            print(f'Creating a new json and saving it to {json_path}')
            updated_json = create_json(dst_dir, db, write_all)
            json.dump(
                updated_json,
                json_path,
                create_path=True,
                indent=4,
                ensure_ascii=False,
            )

        logging.info(f"Done - {time.ctime()}")

    main()
