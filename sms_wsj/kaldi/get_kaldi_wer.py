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
$ ccsalloc --group=hpc-prf-nt1 --res=rset=64:mem=2G:ncpus=1 -t 2h ompi -- python -m sms_wsj.kaldi.get_kaldi_wer -F /EXP/DIR decode with kaldi_data_dir=/KALDI/DATA/DIR model_egs_dir=/MODEL/EGS/DIR dataset=test_eval92

"""

import os
from pathlib import Path
from shutil import copytree

import sacred
from lazy_dataset.database import JsonDatabase
from sms_wsj.kaldi.utils import calculate_mfccs, calculate_ivectors
from sms_wsj.kaldi.utils import create_data_dir_from_audio_dir
from sms_wsj.kaldi.utils import create_kaldi_dir, create_data_dir
from sms_wsj.kaldi.utils import run_process

ex = sacred.Experiment('Kaldi array')
kaldi_root = Path(os.environ['KALDI_ROOT'])


@ex.command
def create_dir(
        audio_dir: Path, dataset_names=None, base_dir=None, json_path=None, db=None,
        data_type='sms_enh', id_to_file_name='{id}_{spk}.wav', target_speaker=0,
        sample_rate=8000,
):
    """

    Args:
        audio_dir: path to audio_dir
        dataset_names: datasets to create a data_dir for
        base_dir: directory in which all information is copied or generated
        json_path: path to wsj_bss.json file
        db: JsonDatabase object
        data_type: name of data type to evaluate
        id_to_file_name: template to get the wav file name from the example_id
        target_speaker: index of speaker to decode
        sample_rate:

    Returns:

    """
    if base_dir is None:
        assert len(ex.current_run.observers) == 1, (
            'FileObserver` missing. Add a `FileObserver` with `-F foo/bar/`.'
        )
        base_dir = Path(
            ex.current_run.observers[0].basedir).expanduser().resolve()

    audio_dir = Path(audio_dir).expanduser().resolve()
    create_data_dir_from_audio_dir(
        audio_dir, base_dir, id_to_file_name=id_to_file_name, db=db,
        json_path=json_path, dataset_names=dataset_names, data_type=data_type,
        target_speaker=target_speaker, sample_rate=sample_rate
    )


@ex.command
def decode(model_egs_dir, dataset_dir, base_dir=None, model_data_type='sms',
           model_dir='chain/tdnn1a_sp', ivector_dir=True, extractor_dir=None,
           data_type='sms_enh', hires=True, num_jobs=8, kaldi_cmd='run.pl'):
    """

    Args:
        model_egs_dir: path to the egs dir the model was trained in
        dataset_dir: kaldi egs dir for the decoding
                e.g.: "<egs_folder>/data/cv_dev_93"
        base_dir: directory in which all information is copied or generated
        model_data_type: data type on which the model was trained
        model_dir: name of model or Path to model_dir
        ivector_dir: directory or name for the ivectors (may be None or False)
        extractor_dir: directory of the ivector extractor (maybe None)
        data_type: name of data type to evaluate
        hires: flag for using high resolution mfcc features (True / False)
        num_jobs: number of parallel jobs
        kaldi_cmd: kaldi cmd for example run.pl, ssh.pl queue.pl

    Returns:

    """

    if base_dir is None:
        assert len(ex.current_run.observers) == 1, (
            'FileObserver` missing. Add a `FileObserver` with `-F foo/bar/`.'
        )
        base_dir = Path(ex.current_run.observers[0].basedir)
        base_dir = base_dir.expanduser().resolve()
        dataset_dir = dataset_dir.expanduser().resolve()
        assert dataset_dir.exists(), dataset_dir
        copytree(dataset_dir, base_dir / 'data' / dataset_dir.name,
                 symlinks=True)
        dataset_dir = base_dir / 'data' / dataset_dir.name
        run_process([
            f'utils/fix_data_dir.sh', f'{dataset_dir}'],
            cwd=str(base_dir), stdout=None, stderr=None)
    else:
        base_dir = base_dir.expanduser().resolve()
    model_egs_dir = Path(model_egs_dir).expanduser().resolve()
    if isinstance(model_dir, str):
        model_dir = model_egs_dir / 'exp' / model_data_type / model_dir

    assert model_dir.exists(), f'{model_dir} does not exist'

    os.environ['PATH'] = f'{base_dir}/utils:{os.environ["PATH"]}'
    decode_dir = base_dir / 'exp' / model_data_type / model_dir.name
    if not decode_dir.exists():
        decode_dir.mkdir(parents=True)
        [os.symlink(str(file), str(decode_dir / file.name))
         for file in model_dir.glob('*') if file.is_file()]
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
    config = 'mfcc_hires.conf' if hires else 'mfcc.conf'
    calculate_mfccs(base_dir, dataset_dir, num_jobs=num_jobs,
                    config=config, recalc=True, kaldi_cmd=kaldi_cmd)
    ivector_dir = calculate_ivectors(
        ivector_dir, base_dir, dataset_dir, extractor_dir, model_egs_dir,
        model_data_type, data_type, num_jobs, kaldi_cmd
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


@ex.config
def default():
    """
    If audio_dir and json_path are defined, the wavs in audio_dir will be
    decoded. If necessary a mapping from example_id to the wav names can be
    specified using id_to_file_name.
    If kaldi_data_dir it will be used as data_dir for decoding. In this case
    audio_dir hast to be None
    If neither audio_dir nor kaldi_data_dir is defined, but json_path is not
    None. kaldi data_dirs for all data_type and dataset_names are created and
    decoded.

    model_egs_dir: egs directory of the trained model with data and exp dir
    num_jobs: if not specified takes the the number of cores as default

    """
    model_egs_dir = None

    # Only one of these two variables has to be defined
    audio_dir = None
    kaldi_data_dir = None

    json_path = None

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
    id_to_file_name = '{id}_{spk}.wav'
    # id_to_file_name = '{}_{}.wav' is another possible default, but only
    # if the first {} represents the example id and the second the speaker id
    target_speaker = [0, 1]

    ref_channels = 0

    if ref_channels > 0 and isinstance(data_type, (list, tuple)):
        assert 'wsj_8k' not in data_type, data_type
    else:
        assert not data_type == 'wsj_8k', (ref_channels, data_type)

    # am specific values which usually do not have to be changed
    ivector_dir = True
    extractor_dir = 'nnet3/extractor'
    model_dir = 'chain/tdnn1a_sp'
    hires = True
    kaldi_cmd = 'run.pl'
    sample_rate = 8000

    # only used for the paderborn parallel computing center
    if 'CCS_NODEFILE' in os.environ:
        num_jobs = len(list(
            Path(os.environ['CCS_NODEFILE']).read_text().strip().splitlines()
        ))
    else:
        # WSJ dev has only 8 speaker and Kaldi fails, when num_jobs is higher.
        num_jobs = min(8, os.cpu_count())


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
    if audio_dir is not None:
        audio_dir = Path(audio_dir).expanduser().resolve()
        assert audio_dir.exists(), audio_dir
        json_path = Path(json_path).expanduser().resolve()
        assert json_path.exists(), json_path
        db = JsonDatabase(json_path)
    elif kaldi_data_dir is not None:
        kaldi_data_dir = Path(kaldi_data_dir).expanduser().resolve()
        assert kaldi_data_dir.exists(), kaldi_data_dir
        assert json_path is None, json_path
    elif json_path is not None:
        json_path = Path(json_path).expanduser().resolve()
        assert json_path.exists(), json_path
        db = JsonDatabase(json_path)
    else:
        raise ValueError('Either json_path, audio_dir or kaldi_data_dir has'
                         'to be defined.')
    if _config['model_egs_dir'] is None:
        model_egs_dir = kaldi_root / 'egs' / 'sms_wsj' / 's5'
    else:
        model_egs_dir = Path(_config['model_egs_dir']).expanduser().resolve()
    assert model_egs_dir.exists(), model_egs_dir

    dataset_names = _config['dataset_names']
    if not isinstance(dataset_names, (tuple, list)):
        dataset_names = [dataset_names]
    data_type = _config['data_type']
    if not isinstance(data_type, (tuple, list)):
        data_type = [data_type]

    kaldi_cmd = _config['kaldi_cmd']
    if not base_dir == model_egs_dir and not (base_dir / 'steps').exists():
        create_kaldi_dir(base_dir, model_egs_dir, exist_ok=True,
                         sample_rate=_config['sample_rate'])
        if kaldi_cmd == 'ssh.pl':
            CCS_NODEFILE = Path(os.environ['CCS_NODEFILE'])
            (base_dir / '.queue').mkdir()
            (base_dir / '.queue' / 'machines').write_text(
                CCS_NODEFILE.read_text())
        elif kaldi_cmd == 'run.pl':
            pass
        else:
            raise ValueError(kaldi_cmd)

    for d_type in data_type:
        for dset in dataset_names:
            dataset_dir = base_dir / 'data' / d_type / dset
            if audio_dir is not None:
                assert len(data_type) == 1, data_type
                create_dir(
                    audio_dir, base_dir=base_dir, db=db, dataset_names=dset
                )
            elif kaldi_data_dir is None:
                create_data_dir(
                    base_dir, db=db, data_type=d_type, dataset_names=dset,
                    ref_channels=_config['ref_channels'],
                    target_speaker=_config['target_speaker'],
                    sample_rate=_config['sample_rate'],
                )
            else:
                assert len(data_type) == 1, (
                    'when using a predefined kaldi_data_dir not more then one '
                    'data_type should be defined. Better use the decode'
                    'command directly'
                )
                copytree(kaldi_data_dir / dset, dataset_dir, symlinks=True)
                run_process([
                    f'utils/fix_data_dir.sh', f'{dataset_dir}'],
                    cwd=str(base_dir), stdout=None, stderr=None)

            decode(
                base_dir=base_dir,
                model_egs_dir=model_egs_dir,
                dataset_dir=dataset_dir,
                model_dir=check_config_element(_config['model_dir']),
                ivector_dir=check_config_element(_config['ivector_dir']),
                extractor_dir=check_config_element(_config['extractor_dir']),
                data_type=d_type
            )
