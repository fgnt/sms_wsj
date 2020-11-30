"""
This script writes a new json which includes the files
written to disk with sms_wsj.database.write_files.py

Additionally, the script allows to update the paths
in case of a change in the database location by using
the old sms_wsj.json as intermediate json.
However, this script does not change the speaker
and utterance combination, log weights, etc. which are
specified in the intermediate json.

"""

from sms_wsj.database.write_files import check_files, KEY_MAPPER
from sms_wsj.database.utils import _example_id_to_rng
import json
import sacred
from pathlib import Path
from lazy_dataset.database import JsonDatabase

ex = sacred.Experiment('Write SMS-WSJ json after wav files are written')


def create_json(db_dir, intermediate_json_path, write_all, snr_range=(20, 30)):
    db = JsonDatabase(intermediate_json_path)
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

            if 'original_source' not in ex['audio_path']:
                # legacy code
                ex['audio_path']['original_source'] = ex['audio_path']['speech_source']

            ex['audio_path']['original_source'] = [
                # .../sms_wsj/cache/wsj_8k_zeromean/13-11.1/wsj1/si_tr_s/4ax/4axc0218.wav
                str(db_dir.joinpath(*Path(rir).parts[-6:]))
                for rir in ex['audio_path']['original_source']
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
    intermed_json_path = None

    # If `False`, expects only observation to exist,
    # else expect all intermediate signals.
    write_all = True

    # Default behavior is to overwrite an existing `sms_wsj.json`. You may
    # specify a different path here to change where the JSON is written to.
    json_path = None

    snr_range = (20, 30)

    assert db_dir is not None, 'You have to specify a database dir'
    assert intermed_json_path is not None, 'You have to specify a path' \
                                               ' to the original sms_wsj.json'

    debug = False


@ex.automain
def main(db_dir, intermed_json_path , write_all, json_path, snr_range):
    intermed_json_path = Path(intermed_json_path).expanduser().resolve()
    db_dir = Path(db_dir).expanduser().resolve()
    if json_path is not None:
        json_path = Path(json_path).expanduser().resolve()
    else:
        json_path = intermed_json_path
    print(f'Creating a new json and saving it to {json_path}')
    num_wav_files = len(check_files(db_dir))
    message = f'Not all wav files seem to exists, you have {num_wav_files},' \
        f' please check your db directory: {db_dir}'
    if write_all:
        assert num_wav_files in [(2 * speakers + 2) * 35875 for speakers in [2, 3, 4]], message
    else:
        assert num_wav_files == 35875, message
    updated_json = create_json(db_dir, intermed_json_path , write_all,
                               snr_range=snr_range)
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with json_path.open('w') as f:
        json.dump(updated_json, f, indent=4, ensure_ascii=False)
    print(f'{json_path} written.')
