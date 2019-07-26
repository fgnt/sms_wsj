import os
import shutil
import stat
from collections import defaultdict
from pathlib import Path

from paderbox.kaldi.io import dump_keyed_lines
from paderbox.utils.process_caller import run_process
from sms_wsj import git_root

DB2AudioKeyMapper = dict(
    wsj_8k='speech_source',
    sms_early='speech_reverberation_direct',
    sms='observation'
)

kaldi_root = Path(os.environ['KALDI_ROOT'])

SAMPLE_RATE = 8000

REQUIRED_FILES = []
REQUIRED_DIRS = ['data/lang', 'data/local',
                 'local', 'steps', 'utils']
REQUIRED_SCRIPTS = ['get_tri3_model.bash', 'path.sh', 'cmd.sh']
DIRS_WITH_CHANGEABLE_FILES = ['conf', 'data/lang_test_tgpr']



def create_kaldi_dir(egs_path):
    """

    :param egs_path:
    :return:
    """
    print(f'Create {egs_path} directory')
    (egs_path / 'data').mkdir(exist_ok=False, parents=True)

    org_dir = (egs_path / '..' / '..' / 'wsj' / 's5').resolve()
    for file in REQUIRED_FILES:
        os.symlink(org_dir / file, egs_path/ file)
    for dirs in REQUIRED_DIRS:
        os.symlink(org_dir / dirs, egs_path / dirs)
    for dirs in DIRS_WITH_CHANGEABLE_FILES:
        shutil.copytree(org_dir / dirs, egs_path/ dirs)
    for script_name in REQUIRED_SCRIPTS:
        shutil.copyfile(git_root / 'scripts' / script_name,
                        egs_path / script_name)
        # make script executable
        file = egs_path / script_name
        st = os.stat(file)
        os.chmod(file, st.st_mode | stat.S_IEXEC)

    if SAMPLE_RATE != 16000:
        for file in ['mfcc.conf', 'mfcc_hires.conf']:
            with (egs_path / 'conf' / file).open('a') as fd:
                fd.writelines(f"--sample-frequency={SAMPLE_RATE}\n")


def create_data_dir(
        kaldi_dir, db, dataset_names=None,
        data_type ='wsj_8k', target_speaker=0, ref_channel=0
):
    """

    :param kaldi_dir:
    :param db:
    :param dataset_names:
    :param data_type:
    :param target_speaker:
    :param ref_channel:
    :return:
    """
    print(f'Create data dir for {data_type} data')
    data_dir = kaldi_dir / 'data' / data_type
    data_dir.mkdir(exist_ok=False, parents=False)
    audio_key = DB2AudioKeyMapper[data_type]

    example_id_to_wav = dict()
    example_id_to_speaker = dict()
    example_id_to_trans = dict()
    example_id_to_duration = dict()
    speaker_to_gender = defaultdict(lambda: defaultdict(list))
    dataset_to_example_id = defaultdict(list)

    if dataset_names is None:
        dataset_names = ('train_si284', 'cv_dev93')
    dataset = db.get_dataset(dataset_names)
    for example in dataset:
        example_id = example['example_id']
        dataset_name = example['dataset']
        wav = example['audio_path'][audio_key][target_speaker]
        wav_command = f'sox {wav} -t wav  -b 16 - remix {ref_channel + 1} |'
        example_id_to_wav[example_id] = wav_command
        try:
            speaker = example['kaldi_transcription'][target_speaker]
            example_id_to_trans[example_id] = speaker
        except KeyError as e:
            raise e
        speaker_id = example['speaker_id'][target_speaker]
        example_id_to_speaker[example_id] = speaker_id
        gender = example['gender'][target_speaker]
        speaker_to_gender[dataset_name][speaker_id] = gender
        num_samples = example['num_samples']['observation']
        example_id_to_duration[example_id] = f"{num_samples/ SAMPLE_RATE:.2f}"
        dataset_to_example_id[dataset_name].append(example_id)

    assert len(example_id_to_speaker) > 0, dataset
    for dataset_name in dataset_names:
        path = data_dir / dataset_name
        path.mkdir(exist_ok=True, parents=False)
        for name, dictionary in (
                ("utt2spk", example_id_to_speaker),
                ("text", example_id_to_trans),
                ("utt2dur", example_id_to_duration),
                ("wav.scp", example_id_to_wav)
        ):
            dictionary = {key: value for key, value in dictionary.items()
                          if key in dataset_to_example_id[dataset_name]}

            assert len(dictionary) > 0, (dataset_name, name)
            dump_keyed_lines(dictionary, path / name)
        dictionary = speaker_to_gender[dataset_name]
        assert len(dictionary) > 0, (dataset_name, name)
        dump_keyed_lines(dictionary, path / 'spk2gender')


def calculate_mfccs(base_dir, dataset, num_jobs=20, config='mfcc.conf',
                    recalc=False, kaldi_cmd='run.pl'):
    """
    :param base_dir: kaldi egs directory with steps and utils dir
    :param dataset: name of folder in data
    :param num_jobs: number of parallel jobs
    :param config: mfcc config
    :param recalc: recalc feats if already calculated
    :param kaldi_cmd:
    :return:
    """
    base_dir = base_dir.expanduser().resolve()

    if isinstance(dataset, str):
        dataset = base_dir / 'data' / dataset
    assert dataset.exists(), dataset
    if not (dataset / 'feats.scp').exists() or recalc:
        run_process([
            'steps/make_mfcc.sh', '--nj', str(num_jobs),
            '--mfcc-config', f'{base_dir}/conf/{config}',
            '--cmd', f'{kaldi_cmd}', f'{dataset}',
            f'{dataset}/make_mfcc', f'{dataset}/mfcc'],
            cwd=str(base_dir), stdout=None, stderr=None
        )

    if not (dataset / 'cmvn.scp').exists() or recalc:
        run_process([
            f'steps/compute_cmvn_stats.sh',
            f'{dataset}', f'{dataset}/make_mfcc', f'{dataset}/mfcc'],
            cwd=str(base_dir), stdout=None, stderr=None
        )
    run_process([
        f'utils/fix_data_dir.sh', f'{dataset}'],
        cwd=str(base_dir), stdout=None, stderr=None
    )


def get_alignments(egs_dir, num_jobs, kaldi_cmd='run.pl',
                   data_type='sms_early', dataset_names=None):
    if dataset_names is None:
        dataset_names = ('train_si284', 'cv_dev93')

    for dataset in dataset_names:
        if not (egs_dir / 'data' / data_type / dataset / 'fetas.scp').exists():
            calculate_mfccs(egs_dir, egs_dir / data_type / dataset_names,
                            num_jobs=num_jobs, kaldi_cmd=kaldi_cmd)
        run_process([
            f'{egs_dir}/steps/align_fmllr.sh',
            '--cmd', kaldi_cmd,
            '--nj', str(num_jobs),
            f'{egs_dir}/data/{data_type}/{dataset}',
            f'{egs_dir}/data/lang',
            f'{egs_dir}/exp/{data_type}/tri4b',
            f'{egs_dir}/exp/{data_type}/tri4b_ali_{dataset}'
        ],
            cwd=str(egs_dir),
            stdout=None, stderr=None
        )
