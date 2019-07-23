import os
from pathlib import Path

import sacred
from paderbox.database import JsonDatabase
from paderbox.utils.process_caller import run_process
from sms_wsj.kaldi.utils import create_data_dir, create_kaldi_dir
from sms_wsj.kaldi.utils import get_alignments

kaldi_root = Path(os.environ['KALDI_ROOT'])
assert kaldi_root.exists(), (
    f'The environmental variable KALDI_ROOT has to be set to a working kaldi'
    f' root, at the moment it points to f{kaldi_root}'
)
assert (kaldi_root / 'src').exists(), (
    f'The environmental variable KALDI_ROOT has to be set to a working kaldi'
    f' root, at the moment it points to f{kaldi_root}'
)
assert (kaldi_root / 'src' / 'base' / '.depend.mk').exists(), (
    'The kaldi your KALDI_ROOT points to is not installed, please refer to'
    ' kaldi for further information on how to install it'
)
ex = sacred.Experiment('Kaldi ASR baseline training')


@ex.config
def config():
    egs_path = None
    json_path = None
    num_jobs = os.cpu_count()
    stage = 0
    # ToDo: change to kaldi_root/egs/ if no egs_path is defined?
    assert egs_path is not None, \
        'The directory where all asr training related data is stored has' \
        ' to be defined, use "with storage_dir=/path/to/storage/dir"'
    assert json_path is not None, \
        'The path to the json describing the SMS-WSJ database has to be' \
        ' defined, use "with json_path=/path/to/json/sms_wsj.json"' \
        ' (for creating the json use ...)'


@ex.automain
def run(_config, egs_path, json_path, stage):
    sms_db = JsonDatabase(json_path)
    sms_kaldi_dir = Path(egs_path).resolve().expanduser() / 'sms_wsj' / 's5'
    if stage <= 0:
        create_kaldi_dir(sms_kaldi_dir)
    if stage <= 1:
        create_data_dir(sms_kaldi_dir, sms_db, data_type='wsj_8k')
    if stage <= 2:
        print('Start training tri3 model on wsj_8k')
        run_process([
            f'{sms_kaldi_dir}/get_tri3_model.bash',
            '--dest_dir', f'{sms_kaldi_dir}',
            '--num_jobs', str(_config["num_jobs"])],
            cwd=str(sms_kaldi_dir),
            stdout=None, stderr=None
        )
    if stage <= 3:
        create_data_dir(sms_kaldi_dir, sms_db, data_type='sms_early')
    if stage <= 4:
        get_alignments(sms_kaldi_dir, 'sms_early',
                       num_jobs=_config['num_jobs'])
    if stage <= 5:
        create_data_dir(sms_kaldi_dir, sms_db, data_type='observation')
