import sacred
import os

from sms_wsj.kaldi.utils import create_data_dir

kaldi_root = os.environ['KALDI_ROOT']

ex = sacred.Experiment('Kaldi ASR baseline training')

@ex.config
def config():
    storage_dir = None
    json_path = None
    num_jobs = os.cpu_count()
    assert storage_dir is not None, 'The directory where all asr training related data is stored has to be defined, use "with storage_dir=/path/to/storage/dir"'
    assert json_path is not None, 'The path to the json describing the SMS-WSJ database has to be defined, use "with json_path=/path/to/json/sms_wsj.json" (for creating the json use ...)'


@ex.automain
def run(_config, storage_dir, json_path):
    create_data_dir(storage_dir, json_path, db='wsj_8k')