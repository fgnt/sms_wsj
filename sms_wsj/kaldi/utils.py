import os
import shutil
import stat
import subprocess
from collections import defaultdict
from functools import partial
from pathlib import Path

from lazy_dataset.database import JsonDatabase
from sms_wsj import git_root

DB2AudioKeyMapper = dict(
    wsj_8k='original_source',
    sms_early='speech_reverberation_early',
    sms='observation',
    noise='noise_image',
    sms_late='speech_reverberation_tail'
)

kaldi_root = Path(os.environ['KALDI_ROOT'])

REQUIRED_FILES = []
REQUIRED_DIRS = ['data/lang', 'data/local',
                 'local', 'steps', 'utils']
DIRS_WITH_CHANGEABLE_FILES = ['conf', 'data/lang_test_tgpr',
                              'data/lang_test_tg']


def create_kaldi_dir(egs_path, org_dir=None, exist_ok=False, sample_rate=8000):
    """

    Args:
        egs_path:
        org_dir:
            An egs folder (e.g. $KALDI_ROOT/egs/wsj/s5). This folder is used as
            reference to create the new eps folder.
            e.g.
             - make symlinks to the 'local', 'steps', 'utils', 'data/lang' and
               'data/local' folder
             - copy 'conf', 'data/lang_test_tgpr' and 'data/lang_test_tg' to
               the new folder



    Returns:

    """
    print(f'Create {egs_path} directory')
    (egs_path / 'data').mkdir(exist_ok=exist_ok, parents=True)
    if org_dir is None:
        org_dir = (egs_path / '..' / '..' / 'wsj' / 's5').resolve()
    for file in REQUIRED_FILES:
        os.symlink(org_dir / file, egs_path / file)
    for dirs in REQUIRED_DIRS:
        os.symlink(org_dir / dirs, egs_path / dirs)
    for dirs in DIRS_WITH_CHANGEABLE_FILES:
        shutil.copytree(org_dir / dirs, egs_path / dirs)
    for script in (git_root / 'scripts').glob('*'):
        if script.name in ['path.sh', 'cmd.sh']:
            new_script_path = egs_path / script.name
        else:
            (egs_path / 'local_sms').mkdir(exist_ok=True)
            new_script_path = egs_path / 'local_sms' / script.name

        shutil.copyfile(script, new_script_path)
        if script.name == 'path.sh':
            with new_script_path.open('r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f'export KALDI_ROOT={kaldi_root}' + '\n' + content)
        # make script executable
        st = os.stat(new_script_path)
        os.chmod(new_script_path, st.st_mode | stat.S_IEXEC)

    if sample_rate != 16000:
        for file in ['mfcc.conf', 'mfcc_hires.conf']:
            with (egs_path / 'conf' / file).open('a') as fd:
                fd.writelines(f"--sample-frequency={sample_rate}\n")


def _get_wav_command_for_json(example, ref_ch, spk, audio_key):
    if isinstance(audio_key, (list, tuple)):
        mix_command = 'sox -m -v 1 ' + ' -v 1 '.join(
            [str(example['audio_path'][audio][spk])
             if isinstance(example['audio_path'][audio], (list, tuple))
             else str(example['audio_path'][audio]) for audio in audio_key]
        )
        wav_command = f'{mix_command} -t wav - | sox -t wav -' \
            f' -t wav -b 16 - remix {ref_ch + 1} |'
    else:
        if isinstance(example['audio_path'][audio_key], (list, tuple)):
            wav = example['audio_path'][audio_key][spk]
        else:
            wav = example['audio_path'][audio_key]
        wav_command = f'sox {wav} -t wav  -b 16 - remix {ref_ch + 1} |'
    return wav_command


def _get_wav_command_for_audio_dir(
        example, ref_ch, spk, audio_dir, id_to_file_name_fn):
    dataset_name = example['dataset']
    ex_id = example['example_id']
    try:
        audio_path = audio_dir / dataset_name / id_to_file_name_fn(ex_id, spk)
        assert audio_path.exists(), audio_path
    except AssertionError:
        audio_path = audio_dir / id_to_file_name_fn(ex_id, spk)
        assert audio_path.exists(), audio_path

    wav_command = f'sox {audio_path} -t wav  -b 16 - remix {ref_ch + 1} |'
    return wav_command


def create_data_dir(
        kaldi_dir, db=None, json_path=None, dataset_names=None,
        data_type='wsj_8k', target_speaker=0, ref_channels=0,
        sample_rate=8000):
    """
    Wrapper calling _create_data_dir for data_dirs from json or db object
    """
    if data_type == 'sms_single_speaker':
        audio_key = [DB2AudioKeyMapper[data]
                     for data in ['sms_early', 'sms_late', 'noise']]
    elif data_type == 'sms_image':
        audio_key = [DB2AudioKeyMapper[data]
                     for data in ['sms_early', 'sms_late']]
    else:
        audio_key = DB2AudioKeyMapper[data_type]
    get_wav_command_fn = partial(
        _get_wav_command_for_json, audio_key=audio_key
    )
    _create_data_dir(
        get_wav_command_fn, kaldi_dir=kaldi_dir, db=db, json_path=json_path,
        dataset_names=dataset_names, data_type=data_type,
        target_speaker=target_speaker, ref_channels=ref_channels,
        sample_rate=sample_rate
    )


def create_data_dir_from_audio_dir(
        audio_dir, kaldi_dir, id_to_file_name='{id}_{spk}.wav', db=None,
        json_path=None, dataset_names=None, data_type='wsj_8k',
        target_speaker=0, ref_channels=0, sample_rate=8000,
):
    """
    Wrapper calling _create_data_dir for data_dirs from audio_dir
    """
    if isinstance(id_to_file_name, str):

        if '{}' in id_to_file_name:
            id_to_file_name_fn = lambda _id, spk: id_to_file_name.format(_id, spk)
        else:
            id_to_file_name_fn = lambda _id, spk: id_to_file_name.format(
                id=_id, spk=spk)
    else:
        id_to_file_name_fn = id_to_file_name
    assert callable(id_to_file_name_fn), id_to_file_name_fn
    if isinstance(target_speaker, (list, tuple)) and len(target_speaker) > 1:
        assert id_to_file_name_fn('id1', 'spk1') != id_to_file_name_fn(
            'id1', 'spk2'), (id_to_file_name_fn('id1', 'spk1'),
                             id_to_file_name_fn('id1', 'spk2'))
        assert id_to_file_name_fn('id1', 'spk1') != id_to_file_name_fn(
            'id2', 'spk1'), (id_to_file_name_fn('id1', 'spk1'),
                             id_to_file_name_fn('id2', 'spk1'))

    get_wav_command_fn = partial(
        _get_wav_command_for_audio_dir, audio_dir=audio_dir,
        id_to_file_name_fn=id_to_file_name_fn
    )
    _create_data_dir(
        get_wav_command_fn, kaldi_dir=kaldi_dir, db=db, json_path=json_path,
        dataset_names=dataset_names, data_type=data_type,
        target_speaker=target_speaker, ref_channels=ref_channels,
        sample_rate=sample_rate
    )


def _create_data_dir(
        get_wav_command_fn, kaldi_dir, db=None, json_path=None,
        dataset_names=None, data_type='wsj_8k', target_speaker=0,
        ref_channels=0, sample_rate=8000,
):
    """

    Args:
        get_wav_command_fn:
        kaldi_dir:
        db:
        json_path:
        dataset_names:
        data_type:
        target_speaker:
        ref_channels:

    Returns:

    """

    assert not (db is None and json_path is None), (db, json_path)
    if db is None:
        db = JsonDatabase(json_path)

    kaldi_dir = Path(kaldi_dir).expanduser().resolve()

    data_dir = kaldi_dir / 'data' / data_type
    data_dir.mkdir(exist_ok=True, parents=True)

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
    if not isinstance(target_speaker, (list, tuple)):
        target_speaker = [target_speaker]
    assert not any([
        (data_dir / dataset_name).exists() for dataset_name in dataset_names
    ]), (
        'One of the following directories already exists: '
        f'{[data_dir / ds_name for ds_name in dataset_names]}\n'
        'Delete them if you want to restart this stage'
    )

    print(
        'Create data dir for '
        f'{", ".join([f"{data_type}/{ds_name}" for ds_name in dataset_names])} '
        'data'
    )

    dataset = db.get_dataset(dataset_names)
    for example in dataset:
        for ref_ch in ref_channels:
            org_example_id = example['example_id']
            dataset_name = example['dataset']
            for t_spk in target_speaker:
                speaker_id = example['speaker_id'][t_spk]
                example_id = speaker_id + '_' + org_example_id
                example_id += f'_c{ref_ch}' if len(ref_channels) > 1 else ''
                example_id_to_wav[example_id] = get_wav_command_fn(
                    example, ref_ch=ref_ch, spk=t_spk)
                try:
                    transcription = example['kaldi_transcription'][t_spk]
                except KeyError:
                    transcription = example['transcription'][t_spk]
                example_id_to_trans[example_id] = transcription

                example_id_to_speaker[example_id] = speaker_id
                gender = example['gender'][t_spk]
                speaker_to_gender[dataset_name][speaker_id] = gender
                if isinstance(example['num_samples'], dict):
                    num_samples = example['num_samples']['observation']
                else:
                    num_samples = example['num_samples']
                example_id_to_duration[
                    example_id] = f"{num_samples / sample_rate:.2f}"
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


def calculate_ivectors(ivector_dir, dest_dir, dataset_dir, extractor_dir=None,
                       org_dir=None, model_data_type='sms',
                       data_type='sms', num_jobs=8, kaldi_cmd='run.pl'):
    """

    Args:
        ivector_dir: ivector directory may be a string, bool or Path
        dest_dir: kaldi egs directory with steps and utils dir
        dataset_dir: kaldi data dir
        extractor_dir: directory of the ivector extractor (may be None)
        org_dir: kaldi egs directory used if extractor_dir is only a string
        model_data_type: dataset specifier for the extractor data type
        data_type: dataset specifier for the input data
        num_jobs: number of parallel jobs
        kaldi_cmd:

    Returns:

    """

    dest_dir = dest_dir.expanduser().resolve()

    if isinstance(ivector_dir, str):
        ivector_dir = dest_dir / 'exp' / model_data_type / 'nnet3' / \
                      ivector_dir
    elif ivector_dir is True:
        ivector_dir = dest_dir / 'exp' / model_data_type / 'nnet3' / (
            f'ivectors_{data_type}_{dataset_dir.name}')
    elif isinstance(ivector_dir, Path):
        ivector_dir = ivector_dir
    else:
        raise ValueError(f'ivector_dir {ivector_dir} has to be either'
                         f' a Path, a string or bolean')
    if not ivector_dir.exists():
        if extractor_dir is None:
            extractor_dir = org_dir / f'exp/{model_data_type}/' \
                f'nnet3/extractor'
        else:
            if isinstance(extractor_dir, str):
                extractor_dir = org_dir / f'exp/{model_data_type}/' \
                    f'{extractor_dir}'
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
            cwd=str(egs_dir)
        )


def run_process(cmd, cwd=None, stderr=None, stdout=None):
    if isinstance(cmd, str):
        shell = True
    else:
        shell = False
    subprocess.run(
        cmd, universal_newlines=True, shell=shell, stdout=stdout,
        stderr=stderr, check=True, env=None, cwd=cwd
    )


def dump_keyed_lines(data_dict: dict, file: Path):
    """
        Used to write Kaldi files

    """
    file = Path(file)
    file = Path(file).expanduser().resolve()
    if file.name in ['utt2dur', 'spk2gender']:
        kaldi_type = file.name
    else:
        kaldi_type = None
    items = data_dict.items()
    # text_file = Path(text_file)
    data = []
    for k, text in items:
        if isinstance(text, list):
            text = ' '.join(map(str, text))
        if kaldi_type == 'utt2dur':
            text_number = float(text)
            assert 0. < text_number < 1000., (
                f'Strange duration: {k}: {text_number} s'
            )
        elif kaldi_type == 'spk2gender':
            text = dict(male='m', female='f', m='m', f='f')[text]
        else:
            pass
        data.append(f'{k} {text}')

    file.write_text('\n'.join(data) + '\n')


def pc2_environ(kaldi_dir):
    CCS_NODEFILE = Path(os.environ['CCS_NODEFILE'])
    if (kaldi_dir / '.queue').exists():
        print('Deleting already existing .queue directory')
        shutil.rmtree(kaldi_dir / '.queue')
    (kaldi_dir / '.queue').mkdir()
    (kaldi_dir / '.queue' / 'machines').write_text(CCS_NODEFILE.read_text())
