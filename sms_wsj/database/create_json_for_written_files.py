"""
This script writes a new json which includes the files
written to disk with sms_wsj.database.write_files.py
"""

from sms_wsj.database.write_files import check_files, KEY_MAPPER
from sms_wsj.database.utils import _example_id_to_rng
import json
import sacred
from pathlib import Path
from lazy_dataset.database import JsonDatabase

ex = sacred.Experiment('Write SMS-WSJ json after wav files are written')


def create_json(db_dir, original_json_path, write_all, snr_range=(20, 30)):
    db = JsonDatabase(original_json_path)
    json_dict = dict(datasets=dict())
    database_dict = db.data['datasets']

    if write_all:
        key_mapper = KEY_MAPPER
    else:
        key_mapper = {'observation': 'observation'}

    for dataset_name, dataset in database_dict.items():
        dataset_dict = dict()
        for ex_id, ex in dataset.items():
            for key, data_type in key_mapper.items():
                current_path = db_dir / data_type / dataset_name
                if key in ['observation', 'noise_image']:
                    ex['audio_path'][key] = str(current_path / f'{ex_id}.wav')
                else:
                    ex['audio_path'][key] = [
                        str(current_path / f'{ex_id}_{k}.wav')
                        for k in range(len(ex['speaker_id']))
                    ]
            rng = _example_id_to_rng(ex_id)
            snr = rng.uniform(*snr_range)
            if 'dataset' in ex:
                del ex['dataset']
            ex["snr"] = snr
            dataset_dict[ex_id] = ex
            json_dict['datasets'][dataset_name] = dataset_dict
    return json_dict



@ex.config
def config():
    db_dir = None
    original_json_path = None

    # If `False`, expects only observation to exist,
    # else expect all intermediate signals.
    write_all = True

    # Default behavior is to overwrite an existing `sms_wsj.json`. You may
    # specify a different path here to change where the JSON is written to.
    json_path = None

    snr_range = (20, 30)

    assert db_dir is not None, 'You have to specify a database dir'
    assert original_json_path is not None, 'You have to specify a path to' \
                                           ' the original sms_wsj.json'

    debug = False


@ex.automain
def main(db_dir, original_json_path, write_all, json_path, snr_range):
    original_json_path = Path(original_json_path).expanduser().resolve()
    db_dir = Path(db_dir).expanduser().resolve()
    if json_path is not None:
        json_path = Path(json_path).expanduser().resolve()
    else:
        json_path = original_json_path
    print(f'Creating a new json and saving it to {json_path}')
    num_wav_files = len(check_files(db_dir))
    message = f'Not all wav files seem to exists, you have {num_wav_files},' \
        f' please check your db directory: {db_dir}'
    if write_all:
        assert num_wav_files == (2 * 2 + 2) * 35875, message
    else:
        assert num_wav_files == 35875, message
    updated_json = create_json(db_dir, original_json_path, write_all, snr_range=snr_range)
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with json_path.open('w') as f:
        json.dump(updated_json, f, indent=4, ensure_ascii=False)
    print(f'{json_path} written.')
