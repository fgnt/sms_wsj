import os
from pathlib import Path
import shutil
from sms_wsj.kaldi.io import dump_keyed_lines
from sms_wsj import git_root

DB2AudioKeyMapper = dict(
    wsj_8k='source_signal',
    sms_early='speech_reverberation_direct',
    sms='observation'
)

kaldi_root = Path(os.environ['KALDI_ROOT'])

SAMPLE_RATE = 8000

NEEDED_FILES = ['cmd.sh', 'path.sh']
NEEDED_DIRS = ['data/lang', 'data/local', 'conf', 'local']
NEEDED_SCRIPTS = ['get_tri3_model.bash']


def create_kaldi_dir(egs_path):
    """

    :param storage_dir:
    :return:
    """
    egs_path.mkdir(exist_ok=True)
    (egs_path / 'data').mkdir(exist_ok=True, parents=True)

    org_dir = (egs_path / '..' / '..' / 'wsj' / 's5').resolve()
    for file in NEEDED_FILES:
        os.symlink(org_dir / file, egs_path/ file)
    for dirs in NEEDED_DIRS:
        os.symlink(org_dir / dirs, egs_path / dirs)
    for link in ['steps', 'utils']:
        linkto = os.readlink(org_dir / link)
        os.symlink(linkto, egs_path / link)
    for script_name in NEEDED_SCRIPTS:
        shutil.copyfile(git_root / 'scripts' / script_name,
                        egs_path /  script_name)


def create_data_dir(
        kaldi_dir, db, dataset_names=None, db_name='wsj_8k'
):
    """

    :param kaldi_dir:
    :param db:
    :param dataset_name:
    :param db_name:
    :return:
    """
    if dataset_names is None:
        dataset_names = (db.datasets_train, db.datasets_eval)
    data_dir = kaldi_dir / 'data' / db_name
    audio_key = DB2AudioKeyMapper[db_name]
    dataset = db.get_dataset(dataset_names)
    example_id_to_wav = dict()
    example_id_to_speaker = dict()
    example_id_to_trans = dict()
    example_id_to_duration = dict()
    speaker_to_gender = dict()
    dataset_to_example_id = dict()
    for example in dataset:
        example_id = example['example_id']
        dataset_name = example['dataset_name']
        example_id_to_wav[example_id] = example['audio_path'][audio_key]
        try:
            example_id_to_trans[example_id] = example_id['kaldi_transcription']
        except KeyError as e:
            raise e
        speaker_id = example['speaker_id']
        example_id_to_speaker[example_id] = speaker_id
        speaker_to_gender[speaker_id] = example['gender']
        example_id_to_duration[example_id] = example['num_samples']
        dataset_to_example_id[dataset_name] = example_id

    assert len(example_id_to_speaker) > 0, dataset
    for dataset_name in dataset_names:
        path = data_dir / dataset_name
        for name, dictionary in (
                ("utt2spk", example_id_to_speaker),
                ("text", example_id_to_trans),
                ("utt2duration", example_id_to_duration),
                ("wav.scp", example_id_to_wav),
                ("speaker2gender", speaker_to_gender)
        ):
            dictionary = {key: value for key, value in dictionary.items()
                          if key in dataset_to_example_id[dataset]}

            assert len(dictionary) > 0, (dataset, name)
            dump_keyed_lines(dictionary, path / name)
