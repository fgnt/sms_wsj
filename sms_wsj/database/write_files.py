"""
Example calls:
python -m sms_wsj.database.wsj.write_wav with dst_dir=/destination/dir json-path=/path/to/sms_wsj.json write_all=True --new_json_path=/path/to/new_sms_wsj.json write_all=True

mpiexec -np 20 python -m sms_wsj.database.wsj.write_wav with dst_dir=/destination/dir json-path=/path/to/sms_wsj.json write_all=True --new_json_path=/path/to/new_sms_wsj.json write_all=True

"""

from functools import partial
from pathlib import Path

import json
import sacred
import numpy as np
import soundfile
from lazy_dataset.database import JsonDatabase
from sms_wsj.database.utils import scenario_map_fn, _example_id_to_rng
import dlp_mpi


ex = sacred.Experiment('Write WSJ BSS files')

type_mapper = {
    'speech_reverberation_early': 'early',
    'speech_reverberation_tail': 'tail',
    'noise_image': 'noise',
    'observation': 'observation'
}


def check_files(dst_dir):
    return [
        p for p in dst_dir.rglob("*.wav") if any(
        [
            (
                    p.match(str(dst_dir / f'{data_type}/**/*.wav')) or
                    p.match(str(dst_dir / f'{data_type}/*.wav'))
            ) for data_type in type_mapper.values()
        ]
    )
    ]


def audio_read(example):
    """
    :param example: example dict
    :return: example dict with audio_data added
    """
    audio_keys = ['rir', 'speech_source']
    keys = list(example['audio_path'].keys())
    example['audio_data'] = dict()
    for audio_key in audio_keys:
        assert audio_key in keys, (
            f'Trying to read {audio_key} but only {keys} are available'
        )
        audio_data = list()
        for wav_file in example['audio_path'][audio_key]:

            with soundfile.SoundFile(wav_file, mode='r') as f:
                audio_data.append(f.read().T)
        example['audio_data'][audio_key] = np.array(audio_data)
    return example


def write_wavs(dst_dir, db, write_all=False, snr_range=(20, 30)):
    if write_all:
        if dlp_mpi.IS_MASTER:
            [(dst_dir / data_type).mkdir(exist_ok=False)
             for data_type in type_mapper.values()]
        map_fn = partial(
            scenario_map_fn,
            snr_range = snr_range,
            sync_speech_source=True,
            add_speech_reverberation_early=True,
            add_speech_reverberation_tail=True
        )
    else:
        if dlp_mpi.IS_MASTER:
            (dst_dir / 'observation').mkdir(exist_ok=False)
        map_fn = partial(
            scenario_map_fn,
            snr_range=snr_range,
            sync_speech_source=True,
            add_speech_reverberation_early=False,
            add_speech_reverberation_tail=False
        )
    for dataset in ['train_si284', 'cv_dev93', 'test_eval92']:
        if dlp_mpi.IS_MASTER:
            [(dst_dir / data_type / dataset).mkdir(exist_ok=False)
             for data_type in type_mapper.values()]
        ds = db.get_dataset(dataset).map(audio_read).map(map_fn)
        for example in dlp_mpi.split_managed(ds):
            audio_dict = example['audio_data']
            example_id = example['example_id']
            if not write_all:
                del audio_dict['speech_reverberation_early']
                del audio_dict['speech_reverberation_tail']
                del audio_dict['noise_image']
            assert all([np.max(np.abs(v)) <= 1 for v in audio_dict.values()]), (
                example_id, [np.max(np.abs(v)) for v in audio_dict.values()])
            for key, value in audio_dict.items():
                if key not in type_mapper:
                    continue
                path = dst_dir / type_mapper[key] / dataset
                if key in ['observation', 'noise_image']:
                    value = value[None]
                for idx, signal in enumerate(value):
                    appendix = f'_{idx}' if len(value) > 1 else ''
                    filename = example_id + appendix + '.wav'
                    audio_path = str(path / filename)
                    with soundfile.SoundFile(
                            audio_path, subtype='FLOAT', mode='w',
                            samplerate=8000,
                            channels=1 if signal.ndim == 1 else signal.shape[0]
                    ) as f:
                        f.write(signal.T)

        dlp_mpi.barrier()

    if dlp_mpi.IS_MASTER:
        created_files = check_files(dst_dir)
        print(f"Written {len(created_files)} wav files.")
    if write_all:
        expect = (2 * 2 + 2) * 32000
        assert len(created_files) == expect, (
            len(created_files), expect
        )
    else:
        assert len(created_files) == 32000, len(created_files)


def create_json(dst_dir, db, write_all, snr_range=(20, 30)):
    json_dict = dict(datasets=dict())
    database_dict = db.data['datasets']
    for dataset_name, dataset in database_dict.items():
        dataset_dict = dict()
        for ex_id, ex in dataset.items():
            if write_all:
                for key, data_type in type_mapper.items():
                    if key in ['observation', 'noise_image']:
                        ex['audio_path'][key] = str(dst_dir / data_type / dataset_name / (
                                ex_id + '.wav')),
                    else:
                        ex['audio_path'][key] = [
                            str(dst_dir / data_type / dataset_name / (ex_id + '_0.wav')),
                            str(dst_dir / data_type / dataset_name / (ex_id + '_1.wav'))
                        ]
            else:
                ex['audio_path'].update({
                    'observation': str(dst_dir / 'observation' / dataset_name / (ex_id + '.wav'))
                })
            rng = _example_id_to_rng(ex_id)
            snr = rng.uniform(*snr_range)
            del ex['dataset']
            ex["snr"] = snr
            dataset_dict[ex_id] = ex
            json_dict['datasets'][dataset_name] = dataset_dict
    return json_dict


@ex.config
def config():
    dst_dir = None
    json_path = None
    write_all = True
    new_json_path = None
    snr_range = (20, 30)
    assert dst_dir is not None, 'You have to specify a destination dir'
    assert json_path is not None, 'You have to specify a path to sms_wsj.json'

@ex.automain
def main(dst_dir, json_path, write_all, new_json_path, snr_range):
    json_path = Path(json_path).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()
    if dlp_mpi.IS_MASTER:
        assert json_path.exists(), json_path
        dst_dir.mkdir(exist_ok=True, parents=True)
        if not any([(dst_dir / data_type).exists()
                    for data_type in type_mapper.keys()]):
            write_files = True
        else:
            write_files = False
            num_wav_files = len(check_files(dst_dir))
            if write_all and num_wav_files == (2 * 2 + 2) * 32000:
                print('Wav files seem to exist. They are not overwritten.')
            elif not write_all and num_wav_files == 32000 and (
                    dst_dir / 'observation').exists():
                print('Wav files seem to exist. They are not overwritten.')
            else:
                raise ValueError(
                    'Not all wav files exist. However, the directory structure'
                    ' already exists.')
    else:
        write_files = None
    write_files = dlp_mpi.COMM.bcast(write_files, root=dlp_mpi.MASTER)
    db = JsonDatabase(json_path)
    if write_files:
        write_wavs(dst_dir, db, write_all=write_all, snr_range=snr_range)

    if dlp_mpi.IS_MASTER and new_json_path:
        print(f'Creating a new json and saving it to {new_json_path}')
        new_json_path = Path(new_json_path).expanduser().resolve()
        updated_json = create_json(dst_dir, db, write_all, snr_range=snr_range)
        new_json_path.parent.mkdir(exist_ok=True, parents=True)
        with new_json_path.open('w') as f:
            json.dump(updated_json, f, indent=4, ensure_ascii=False)
        print(f'{json_path} written')
