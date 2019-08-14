import os
import shutil
import stat
from collections import defaultdict
from pathlib import Path

from paderbox.kaldi.io import dump_keyed_lines
from paderbox.database import JsonDatabase
from paderbox.utils.process_caller import run_process
from sms_wsj import git_root

DB2AudioKeyMapper = dict(
    wsj_8k='speech_source',
    sms_early='speech_reverberation_direct',
    sms='observation',
    noise='noise_image',
    sms_late='speech_reverberation_tail'
)

kaldi_root = Path(os.environ['KALDI_ROOT'])

SAMPLE_RATE = 8000

REQUIRED_FILES = []
REQUIRED_DIRS = ['data/lang', 'data/local',
                 'local', 'steps', 'utils']
DIRS_WITH_CHANGEABLE_FILES = ['conf', 'data/lang_test_tgpr',
                              'data/lang_test_tg']



def create_kaldi_dir(egs_path, org_dir=None):
    """

    :param egs_path:
    :return:
    """
    print(f'Create {egs_path} directory')
    (egs_path / 'data').mkdir(exist_ok=False, parents=True)
    if org_dir is None:
        org_dir = (egs_path / '..' / '..' / 'wsj' / 's5').resolve()
    for file in REQUIRED_FILES:
        os.symlink(org_dir / file, egs_path/ file)
    for dirs in REQUIRED_DIRS:
        os.symlink(org_dir / dirs, egs_path / dirs)
    for dirs in DIRS_WITH_CHANGEABLE_FILES:
        shutil.copytree(org_dir / dirs, egs_path/ dirs)
    for script in (git_root / 'scripts').glob('*'):
        if script.name in ['path.sh', 'cmd.sh']:
            new_script_path = egs_path / script.name
        else:
            (egs_path / 'local_sms').mkdir(exist_ok=True)
            new_script_path = egs_path / 'local_sms' / script.name

        shutil.copyfile(script, new_script_path)
        # make script executable
        st = os.stat(new_script_path)
        os.chmod(new_script_path, st.st_mode | stat.S_IEXEC)

    if SAMPLE_RATE != 16000:
        for file in ['mfcc.conf', 'mfcc_hires.conf']:
            with (egs_path / 'conf' / file).open('a') as fd:
                fd.writelines(f"--sample-frequency={SAMPLE_RATE}\n")


def create_data_dir(
        kaldi_dir, db=None, json_path=None, dataset_names=None,
        data_type='wsj_8k', target_speaker=0, ref_channels=0
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
    print(f'Create data dir for {data_type}/{dataset_names} data')

    if data_type == 'sms_single_speaker':
        audio_key = [DB2AudioKeyMapper[data]
                     for data in ['sms_early', 'sms_late', 'noise']]
    elif data_type == 'sms_image':
        audio_key = [DB2AudioKeyMapper[data]
                     for data in ['sms_early', 'sms_late']]
    else:
        audio_key = DB2AudioKeyMapper[data_type]
    assert not (db is None and json_path is None), (db, json_path)
    if db is None:
        assert json_path is not None, json_path
        db = JsonDatabase(json_path)

    data_dir = kaldi_dir / 'data' / data_type
    data_dir.mkdir(exist_ok=True, parents=False)

    if not isinstance(ref_channels, (list, tuple)):
        ref_channels = [ref_channels]
    example_id_to_wav = dict()
    example_id_to_speaker = dict()
    example_id_to_trans = dict()
    example_id_to_duration = dict()
    speaker_to_gender = defaultdict(lambda: defaultdict(list))
    dataset_to_example_id = defaultdict(list)

    if dataset_names is None:
        dataset_names = ('train_si284', 'cv_dev93', 'test_eval92')
    elif isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert not any([
        (data_dir / dataset_name).exists() for dataset_name in dataset_names])
    dataset = db.get_dataset(dataset_names)
    for example in dataset:
        for ref_ch in ref_channels:
            example_id = example['example_id']
            dataset_name = example['dataset']
            if isinstance(audio_key, (list, tuple)):
                mix_command = 'sox -m -v 1 ' + ' -v 1 '.join(
                    [str(example['audio_path'][audio][target_speaker])
                     for audio in audio_key]
                )
                wav_command = f'{mix_command} -t wav - | sox -t wav -' \
                    f' -t wav -b 16 - remix {ref_ch + 1} |'
            else:
                wav = example['audio_path'][audio_key][target_speaker]
                wav_command = f'sox {wav} -t wav  -b 16 - remix {ref_ch + 1} |'
            example_id += f'_c{ref_ch}' if len(ref_channels) > 1 else ''
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
        path.mkdir(exist_ok=False, parents=False)
        for name, dictionary in (
                ("utt2spk", example_id_to_speaker),
                ("text", example_id_to_trans),
                ("utt2dur", example_id_to_duration),
                ("wav.scp", example_id_to_wav)
        ):
            dictionary = {key: value for key, value in dictionary.items()
                          if key in dataset_to_example_id[dataset_name]}

            assert len(dictionary) > 0, (dataset_name, name)
            if name == 'utt2dur':
                dump_keyed_lines(dictionary, path / 'reco2dur')
            dump_keyed_lines(dictionary, path / name)
        dictionary = speaker_to_gender[dataset_name]
        assert len(dictionary) > 0, (dataset_name, name)
        dump_keyed_lines(dictionary, path / 'spk2gender')
        run_process([
            f'utils/fix_data_dir.sh', f'{path}'],
            cwd=str(kaldi_dir), stdout=None, stderr=None
        )


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


def calculate_ivectors(ivector_dir, dest_dir, org_dir, train_affix,
                       dataset_dir, extractor_dir, dataype='sms', num_jobs=8,
                       kaldi_cmd='run.pl'):
    '''

    :param ivector_dir: ivector directory may be a string, bool or Path
    :param base_dir: kaldi egs directory with steps and utils dir
    :param train_affix: dataset specifier (dataset name without train or dev)
    :param dataset_dir: kaldi dataset dir
    :param extractor_dir: directory of the ivector extractor (may be None)
    :param num_jobs: number of parallel jobs
    :return:
    '''
    dest_dir = dest_dir.expanduser().resolve()

    if isinstance(ivector_dir, str):
        ivector_dir = dest_dir / 'exp' / dataype / (
                'nnet3_' + train_affix) / ivector_dir
    elif ivector_dir is True:
        ivector_dir = dest_dir / 'exp' / dataype / (
                'nnet3_' + train_affix) / (
            f'ivectors_{dataset_dir.parent.name}_{dataset_dir.name}')
    elif isinstance(ivector_dir, Path):
        ivector_dir = ivector_dir
    else:
        raise ValueError(f'ivector_dir {ivector_dir} has to be either'
                         f' a Path, a string or bolean')
    if not ivector_dir.exists():
        if extractor_dir is None:
            extractor_dir = org_dir / f'exp/{dataype}/' \
                f'nnet3_{train_affix}/extractor'
        else:
            if isinstance(extractor_dir, str):
                extractor_dir = org_dir / f'exp/{dataype}/{extractor_dir}'
        assert extractor_dir.exists(), extractor_dir
        print(f'Directory {ivector_dir} not found, estimating ivectors')
        run_process([
            'steps/online/nnet2/extract_ivectors_online.sh',
            '--cmd', f'{kaldi_cmd}', '--nj', f'{num_jobs}', f'{dataset_dir}',
            f'{extractor_dir}', str(ivector_dir)],
            cwd=str(dest_dir),
            stdout=None, stderr=None
        )
    return ivector_dir


def get_alignments(egs_dir, num_jobs, kaldi_cmd='run.pl',
                   gmm_data_type=None, data_type='sms_early',
                   dataset_names=None):
    if dataset_names is None:
        dataset_names = ('train_si284', 'cv_dev93')
    if gmm_data_type is None:
        gmm_data_type = data_type

    for dataset in dataset_names:
        dataset_dir = egs_dir / 'data' / data_type / dataset
        if not (dataset_dir / 'feats.scp').exists():
            calculate_mfccs(egs_dir, dataset_dir, num_jobs=num_jobs,
                            kaldi_cmd=kaldi_cmd)
        run_process([
            f'{egs_dir}/steps/align_fmllr.sh',
            '--cmd', kaldi_cmd,
            '--nj', str(num_jobs),
            f'{dataset_dir}',
            f'{egs_dir}/data/lang',
            f'{egs_dir}/exp/{gmm_data_type}/tri4b',
            f'{egs_dir}/exp/{data_type}/tri4b_ali_{dataset}'
        ],
            cwd=str(egs_dir),
            stdout=None, stderr=None
        )
