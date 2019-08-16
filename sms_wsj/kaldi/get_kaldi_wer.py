"""
Example call on local machine:
Automatically takes all available cores:
$ python -m sms_wsj.kaldi.get_kaldi_wer -F /EXP/DIR with kaldi_data_dir=/KALDI/DATA/DIR model_egs_dir=/MODEL/EGS/DIR

Uses the json_path to create a kaldi data dir
$ python -m sms_wsj.kaldi.get_kaldi_wer -F /EXP/DIR with json_path=/JSON/PATH model_egs_dir=/MODEL/EGS/DIR

Evaluates audio data in audio_dir, expects audio_dir/dataset to exist and audio files of format {exampled_id}_0.wav the format can be changed using the variable id_to_file_name
$ python -m sms_wsj.kaldi.get_kaldi_wer -F /EXP/DIR with audio_dir=/AUDIO/DIR json_path=/JSON/PATH model_egs_dir=/MODEL/EGS/DIR

The follwoing command is used to directly decode a dataset. Expects kaldi_data_dir/dataset to exist.
$ python -m sms_wsj.kaldi.get_kaldi_wer -F /EXP/DIR decode with kaldi_data_dir=/KALDI/DATA/DIR model_egs_dir=/MODEL/EGS/DIR dataset=test_eval92


Example call on pc2 (HPC system in paderborn):
$ ccsalloc --group=hpc-prf-nt1 --res=rset=64:vmem=2G:mem=2G:ncpus=1 -t 6h python -m sms_wsj.kaldi.get_kaldi_wer -F ~/sacred/chime5/arrayBSS/54/kaldi/inear with inear audio_dir=../../audio/dev

"""

import os

from collections import defaultdict
from pathlib import Path

from sms_wsj.kaldi.utils import dump_keyed_lines, create_data_dir
from lazy_dataset.database import JsonDatabase
from shutil import copyfile, copytree

import sacred

from sms_wsj.kaldi.utils import run_process
from sms_wsj.kaldi.utils import create_kaldi_dir, SAMPLE_RATE
from sms_wsj.kaldi.utils import calculate_mfccs, calculate_ivectors

ex = sacred.Experiment('Kaldi array')
kaldi_root = Path(os.environ['KALDI_ROOT'])

REQUIRED_FILES = ['cmd.sh', 'path.sh']
REQUIRED_DIRS = ['data/lang', 'data/local', 'data/srilm', 'conf', 'local']


@ex.capture
def copy_ref_dir(out_dir, ref_dir, audio_dir, allow_missing_files=False):
    audio_dir = audio_dir.expanduser().resolve()

    # ToDo: improve exception msg when len(used_ids) == 0.
    #       Example cause: wrong audio_dir

    target_speaker = 0
    required_files = ['utt2spk', 'text']
    with (ref_dir / 'text').open() as file:
        text = file.readlines()
    ref_ids = [line.split(' ', maxsplit=1)[0].strip() for line in text]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True)
    for files in required_files:
        copyfile(str(ref_dir / files), str(out_dir / files))
    ids = {
        wav_file: wav_file.name.split('.wav')[0].split(f'_{target_speaker}')[0]
        for wav_file in audio_dir.glob('*')
    }
    assert len(ids) > 0, ids

    used_ids = {
        kaldi_id: wav_file
        for wav_file, kaldi_ids in ids.items()
        for kaldi_id in kaldi_ids
        if kaldi_id in ref_ids
    }
    assert len(used_ids) > 0, used_ids

    if len(used_ids) < len(ids):
        print(f'Not all files in {audio_dir} were used, '
              f'{len(ids) - len(used_ids)} ids are not used in kaldi')
    elif len(used_ids) < len(ref_ids):
        if not allow_missing_files:
            raise ValueError(
                f'{len(ref_ids) - len(used_ids)} files are missing in {audio_dir}.'
                f' We found only {len(used_ids)} files but expect {len(ref_ids)}'
                f' files')
        print(
            f'{len(ref_ids) - len(used_ids)} files are missing in {audio_dir}.'
            f' We found only {len(used_ids)} files but expect {len(ref_ids)}'
            f' files. Still continuing to decode the remaining files')
        ref_ids = [_id for _id in used_ids.keys()]
        ref_ids.sort()
        for files in ['utt2spk', 'text']:
            with (out_dir / files).open() as fd:
                lines = fd.readlines()
                lines = [line for line in lines
                         if line.split(' ')[0] in used_ids]
            (out_dir / files).unlink()
            with (out_dir / files).open('w') as fd:
                fd.writelines(lines)

    wavs = [' '.join([kaldi_id, str(used_ids[kaldi_id])]) + '\n'
            for kaldi_id in ref_ids]
    with (out_dir / 'wav.scp').open('w') as file:
        file.writelines(wavs)


@ex.command
def create_data_dir_from_audio_dir(
        audio_dir: Path, dataset_names, base_dir=None, json_path=None, db=None,
        data_type='sms_enh', id_to_file_name='{}_0.wav', target_speaker=0
):
    """
    """
    if base_dir is None:
        assert len(ex.current_run.observers) == 1, (
            'FileObserver` missing. Add a `FileObserver` with `-F foo/bar/`.'
        )
        base_dir = Path(ex.current_run.observers[0].basedir)

    data_dir = base_dir / 'data'
    assert isinstance(data_type, str), data_type
    if db is None:
        db = JsonDatabase(json_path)

    if isinstance(id_to_file_name, str):
        id_to_file_name_fn = lambda _id: id_to_file_name.format(_id)
    else:
        id_to_file_name_fn = id_to_file_name
    assert callable(id_to_file_name_fn), id_to_file_name_fn

    data_dir.mkdir(exist_ok=True, parents=True)

    example_id_to_wav = dict()
    example_id_to_speaker = dict()
    example_id_to_trans = dict()
    example_id_to_duration = dict()
    speaker_to_gender = defaultdict(lambda: defaultdict(list))
    dataset_to_example_id = defaultdict(list)

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert not any([
        (data_dir / dataset_name).exists() for dataset_name in
        dataset_names]), data_dir
    assert all([(audio_dir / dataset_name) for dataset_name in
        dataset_names]), audio_dir
    dataset = db.get_dataset(dataset_names)
    for example in dataset:
        example_id = example['example_id']
        dataset_name = example['dataset']
        audio_path = audio_dir / dataset_name / id_to_file_name_fn(example_id)
        assert audio_path.exists(), audio_path
        example_id_to_wav[example_id] = audio_path
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
        example_id_to_duration[
            example_id] = f"{num_samples / SAMPLE_RATE:.2f}"
        dataset_to_example_id[dataset_name].append(example_id)

    assert len(example_id_to_speaker) > 0, dataset
    for dataset_name in dataset_names:
        dset_dir = data_dir / data_type / dataset_name
        dset_dir.mkdir(parents=True)
        for name, dictionary in (
                ("utt2spk", example_id_to_speaker),
                ("text", example_id_to_trans),
                ("utt2dur", example_id_to_duration),
                ("wav.scp", example_id_to_wav)
        ):
            dictionary = {key: value for key, value in dictionary.items()
                          if key in dataset_to_example_id[dataset_name]}

            assert len(dictionary) > 0, (dataset_name, name)
            dump_keyed_lines(dictionary, dset_dir / name)
        dictionary = speaker_to_gender[dataset_name]
        assert len(dictionary) > 0, (dataset_name, name)
        dump_keyed_lines(dictionary, dset_dir / 'spk2gender')


@ex.command
def decode(model_egs_dir, dataset_dir, base_dir=None, model_data_type='sms',
           model_dir='chain/tdnn1a_sp', ivector_dir=True, extractor_dir=None,
           data_type='sms_enh', hires=True, num_jobs=8, kaldi_cmd='run.pl'):
    '''

    :param model_dir: name of model or Path to model_dir
    :param dest_dir: kaldi egs dir for the decoding
    :param org_dir: kaldi egs dir from which information for decoding are gathered
    :param audio_dir: directory of audio files to decode (may be None)
    :param ref_dir: reference kaldi dataset directory or name for decode dataset
    :param ivector_dir: directory or name for the ivectors (may be None or False)
    :param extractor_dir: directory of the ivector extractor (maybe None)
    :param hires: flag for using high resolution mfcc features (True / False)
    :param enh: name of the enhancement method, used for creating dataset name
    :param num_jobs: number of parallel jobs
    :return:
    '''
    if base_dir is None:
        assert len(ex.current_run.observers) == 1, (
            'FileObserver` missing. Add a `FileObserver` with `-F foo/bar/`.'
        )
        base_dir = Path(ex.current_run.observers[0].basedir)
        base_dir = base_dir.expanduser().resolve()
        dataset_dir = base_dir / dataset_dir
        assert dataset_dir.exists(), dataset_dir
        copytree(dataset_dir, base_dir / 'data' / dataset_dir.name,
                 symlinks=True)
        dataset_dir = base_dir / 'data' / dataset_dir.name
    else:
        base_dir = base_dir.expanduser().resolve()
    model_egs_dir = Path(model_egs_dir).expanduser().resolve()
    if isinstance(model_dir, str):
        model_dir = model_egs_dir / 'exp' / model_data_type / model_dir
    else:
        model_dir = Path(model_dir)
    assert model_dir.exists(), f'{model_dir} does not exist'


    os.environ['PATH'] = f'{base_dir}/utils:{os.environ["PATH"]}'
    decode_dir = base_dir / 'exp' / model_data_type / model_dir.name
    if not decode_dir.exists():
        decode_dir.mkdir(parents=True)
        [os.symlink(file, decode_dir / file.name)
         for file in (model_dir).glob('*') if file.is_file()]
        assert (decode_dir / 'final.mdl').exists(), (
            f'final.mdl not in decode_dir: {decode_dir}, '
            f'maybe using worn model_egs_dir: {model_egs_dir}?'
        )
    decode_name = f'decode_{data_type}_{dataset_dir.name}'
    (decode_dir / decode_name).mkdir(exist_ok=False)
    if not base_dir == model_egs_dir and not (base_dir / 'steps').exists():
        create_kaldi_dir(base_dir, model_egs_dir, exist_ok=True)
        if kaldi_cmd == 'ssh.pl':
            CCS_NODEFILE = Path(os.environ['CCS_NODEFILE'])
            (base_dir / '.queue').mkdir()
            (base_dir / '.queue' / 'machines').write_text(
                CCS_NODEFILE.read_text())
        elif kaldi_cmd == 'run.pl':
            pass
        else:
            raise ValueError(kaldi_cmd)
    run_process([
        f'{base_dir}/utils/fix_data_dir.sh', str(dataset_dir)],
        cwd=str(base_dir), stdout=None, stderr=None
    )
    config = 'mfcc_hires.conf' if hires else 'mfcc.conf'
    calculate_mfccs(base_dir, dataset_dir, num_jobs=num_jobs,
                    config=config, recalc=True, kaldi_cmd=kaldi_cmd)
    ivector_dir = calculate_ivectors(
        ivector_dir, base_dir, model_egs_dir, dataset_dir,
        extractor_dir, model_data_type, data_type, num_jobs, kaldi_cmd
    )
    run_process([
        'steps/nnet3/decode.sh', '--acwt', '1.0',
        '--post-decode-acwt', '10.0',
        '--extra-left-context', '0', '--extra-right-context', '0',
        '--extra-left-context-initial', '0', '--extra-right-context-final',
        '0', '--frames-per-chunk', '140', '--nj', str(num_jobs),
        '--cmd', f'{kaldi_cmd}', '--online-ivector-dir',
        str(ivector_dir), f'{model_dir.parent}/tree_a_sp/graph_tgpr',
        str(dataset_dir), str(decode_dir / decode_name)],
        cwd=str(base_dir),
        stdout=None, stderr=None
    )
    print((decode_dir / decode_name / 'scoring_kaldi' / 'best_wer'
           ).read_text())


def on_pc2():
    if 'CCS_NODEFILE' in os.environ:
        return True
    else:
        return False


@ex.config
def default():
    """
    audio_dir: Dir to decode. The path can be absolute or relative to the sacred
        base folder. If None take the ref_dev_dir as audio_dir. Note when
        ref_dev_dir is a relative path, it is relative to org_dir and not sacred
        base folder.

    model_egs_dir: egs directory of the trained model with data and exp dir


    """
    model_egs_dir = None

    # Only one of these two variables has to be defined
    audio_dir = None
    kaldi_data_dir = None

    json_path = None # only necessary if using audio dir

    model_data_type = 'sms_single_speaker'
    dataset_names = ['test_eval92', 'cv_dev93']

    # This is only used when decode is called directly
    if isinstance(dataset_names, str):
        dataset_dir = f'{kaldi_data_dir}/{dataset_names}'
    else:
        dataset_dir = None
    if kaldi_data_dir is None and audio_dir is None and dataset_dir is None:
        data_type = ['wsj_8k', 'sms_early', 'sms_image',
                     'sms_single_speaker', 'sms']
    else:
        data_type = 'sms_enh'

    # only used with audio_dir
    id_to_file_name = '{}_0.wav'
    target_speaker = 0


    # am specific values which usually do not have to be changed
    ivector_dir = True
    extractor_dir = 'nnet3/extractor'
    model_dir = 'chain/tdnn1a_sp'
    hires = True

    if 'CCS_NODEFILE' in os.environ:
        num_jobs = len(list(
            Path(os.environ['CCS_NODEFILE']).read_text().strip().splitlines()
        ))
    else:
        num_jobs = os.cpu_count()

    if on_pc2():
        kaldi_cmd = 'ssh.pl'
    else:
        kaldi_cmd = 'run.pl'

@ex.named_config
def sms_single_speaker():
    model_egs_dir = '/scratch/hpc-prf-nt1/jensheit/python_packages/sms_wsj/data/sms_single_speaker/s5'


def check_config_element(element):
    if element is not None and not isinstance(element, bool):
        element_path = element
        if Path(element_path).exists():
            element_path = Path(element_path)
    elif isinstance(element, bool):
        element_path = element
    else:
        element_path = None
    return element_path


@ex.automain
def run(_config, _run, audio_dir, kaldi_data_dir, json_path):
    assert Path(kaldi_root).exists(), kaldi_root

    assert len(ex.current_run.observers) == 1, (
        'FileObserver` missing. Add a `FileObserver` with `-F foo/bar/`.'
    )
    base_dir = Path(ex.current_run.observers[0].basedir)
    base_dir = base_dir.expanduser().resolve()
    assert bool(audio_dir) ^ bool(kaldi_data_dir), 'Either '
    if audio_dir is not None:
        audio_dir = base_dir / audio_dir
        assert audio_dir.exists(), audio_dir
        json_path = base_dir / json_path
        assert json_path.exists(), json_path
        db = JsonDatabase(json_path)
    elif kaldi_data_dir is not None:
        kaldi_data_dir = base_dir / kaldi_data_dir
        assert kaldi_data_dir.exists(), kaldi_data_dir
        assert json_path is None, json_path
    elif json_path is not None:
        json_path = base_dir / json_path
        assert json_path.exists(), json_path
        db = JsonDatabase(json_path)
    else:
        raise ValueError('Either json_path, audio_dir or kaldi_data_dir has'
                         'to be defined.')
    if _config['model_egs_dir'] is None:
        model_egs_dir = kaldi_root / 'egs' / 'sms_wsj' / 's5'
    else:
        model_egs_dir = Path(_config['model_egs_dir'])
    assert model_egs_dir.exists(), model_egs_dir

    dataset_names = _config['dataset_names']
    if not isinstance(dataset_names, (tuple, list)):
        dataset_names = [dataset_names]
    data_type = _config['data_type']
    if not isinstance(data_type, (tuple, list)):
        data_type = [data_type]

    for d_type in data_type:
        for dset in dataset_names:
            dataset_dir = base_dir / 'data' / d_type / dset
            if audio_dir is not None:
                assert len(data_type) == 1, data_type
                create_data_dir_from_audio_dir(
                    audio_dir, base_dir=base_dir, db=db, dataset_names=dset
                )
            elif kaldi_data_dir is None:
                create_data_dir(base_dir, db=db, data_type=d_type,
                                dataset_names=dataset_names)
            else:
                assert len(data_type) == 1, (
                    'when using a predefined kaldi_data_dir not more then one '
                    'data_type should be defined. Better use the decode'
                    'command directly'
                )
                copytree(kaldi_data_dir / dset, dataset_dir, symlinks=True)

            decode(
                base_dir=base_dir,
                model_egs_dir=model_egs_dir,
                dataset_dir=dataset_dir,
                model_data_type=_config['model_data_type'],
                model_dir=check_config_element(_config['model_dir']),
                ivector_dir=check_config_element(_config['ivector_dir']),
                extractor_dir=check_config_element(_config['extractor_dir']),
                hires=_config['hires'],
                num_jobs=_config['num_jobs']
            )
