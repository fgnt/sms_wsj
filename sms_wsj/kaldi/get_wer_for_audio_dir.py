"""
Example call on local machine:
Automatically takes all available cores:
$ python -m sms_wsj.get_wer_for_audio_dir -F ~/sacred/chime5/arrayBSS/54/kaldi/inear with inear audio_dir=../../audio/dev

Example call on pc2 (HPC system in paderborn):
$ ccsalloc --group=hpc-prf-nt1 --res=rset=64:vmem=2G:mem=2G:ncpus=1 -t 6h python -m sms_wsj.get_wer_for_audio_dir -F ~/sacred/chime5/arrayBSS/54/kaldi/inear with inear audio_dir=../../audio/dev

"""

import os

from collections import defaultdict
from pathlib import Path

from sms_wsj.kaldi.utils import dump_keyed_lines
from lazy_dataset.database import JsonDatabase
from shutil import copyfile

import sacred

from sms_wsj.kaldi.utils import run_process
from paderbox.utils.pc2 import write_ccsinfo_files
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


def create_data_dir(data_dir, audio_dir, json_path, dataset_name):
    """
    """

    assert json_path is not None, json_path
    db = JsonDatabase(json_path)
    target_speaker = 0
    data_dir.mkdir(exist_ok=True, parents=True)

    example_id_to_wav = dict()
    example_id_to_speaker = dict()
    example_id_to_trans = dict()
    example_id_to_duration = dict()
    speaker_to_gender = defaultdict(lambda: defaultdict(list))
    dataset_to_example_id = defaultdict(list)

    assert isinstance(dataset_name, str), dataset_name
    dataset_names = [dataset_name]
    assert not any([
        (data_dir / dataset_name).exists() for dataset_name in
        dataset_names])
    dataset = db.get_dataset(dataset_names)
    for example in dataset:
        example_id = example['example_id']
        dataset_name = example['dataset']
        audio_path = audio_dir / (example_id + f'_beamformed_{target_speaker+1}.wav')
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
    for name, dictionary in (
            ("utt2spk", example_id_to_speaker),
            ("text", example_id_to_trans),
            ("utt2dur", example_id_to_duration),
            ("wav.scp", example_id_to_wav)
    ):
        dictionary = {key: value for key, value in dictionary.items()
                      if key in dataset_to_example_id[dataset_name]}

        assert len(dictionary) > 0, (dataset_name, name)
        dump_keyed_lines(dictionary, data_dir / name)
    dictionary = speaker_to_gender[dataset_name]
    assert len(dictionary) > 0, (dataset_name, name)
    dump_keyed_lines(dictionary, data_dir / 'spk2gender')



def get_data_dir(
        base_dir: Path, audio_dir: Path, json_path: Path, enh='bss_beam',
        dataset='test_eval92', hires=True, num_jobs=8
):
    base_dir = base_dir.expanduser().resolve()
    if isinstance(enh, Path):
        data_dir = enh
    elif 'hires' in enh and hires:
        data_dir = base_dir / 'data' / 'sms_enh' / f'{dataset}_{enh}'
    elif hires:
        data_dir = base_dir / 'data' / 'sms_enh' / f'{dataset}_{enh}_hires'
    else:
        data_dir = base_dir / 'data' / 'sms_enh' / f'{dataset}_{enh}'
    config = 'mfcc_hires.conf' if hires else 'mfcc.conf'
    if not data_dir.exists():
        print(f'Directory {data_dir} not found creating data directory')

        assert audio_dir.exists(), audio_dir
        create_data_dir(data_dir, audio_dir, json_path, dataset)
        run_process([
            f'{base_dir}/utils/fix_data_dir.sh', str(data_dir)],
            cwd=str(base_dir), stdout=None, stderr=None
        )
        calculate_mfccs(base_dir, data_dir, num_jobs=num_jobs,
                        config=config, recalc=True)
    return data_dir


@ex.capture
def decode(model_dir, dest_dir, org_dir, audio_dir: Path, json_path: Path,
           dataset='test_eval92',  model_data_type='sms',
           model_type='tdnn1a_sp', ivector_dir=False, extractor_dir=None,
           hires=True, enh='bss_beam', num_jobs=8, kaldi_cmd='run.pl'):
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
    dest_dir = dest_dir.expanduser().resolve()
    if isinstance(model_dir, str):
        model_dir = org_dir / 'exp' / model_data_type / model_dir
    assert model_dir.exists(), f'{model_dir} does not exist'
    if not dest_dir == org_dir:
        create_kaldi_dir(dest_dir, org_dir)
        # if 'CCS_NODEFILE' in os.environ:
        if kaldi_cmd == 'ssh.pl':
            CCS_NODEFILE = Path(os.environ['CCS_NODEFILE'])
            (dest_dir / '.queue').mkdir()
            (dest_dir / '.queue' / 'machines').write_text(
                CCS_NODEFILE.read_text())
        elif kaldi_cmd == 'run.pl':
            pass
        else:
            raise ValueError(kaldi_cmd)
    os.environ['PATH'] = f'{dest_dir}/utils:{os.environ["PATH"]}'
    train_affix = model_dir.name.split('_', maxsplit=1)[-1]
    dataset_dir = get_data_dir(dest_dir, audio_dir, json_path, enh,
                               dataset, hires, num_jobs)
    decode_dir = dest_dir / f'exp/{model_data_type}/{model_dir.name}/{model_type}'
    if not decode_dir.exists():
        decode_dir.mkdir(parents=True)
        [os.symlink(file, decode_dir / file.name)
         for file in (model_dir / model_type).glob('*') if file.is_file()]
        assert (decode_dir / 'final.mdl').exists(), (
            f'final.mdl not in decode_dir: {decode_dir},'
            f' maybe using worn org_dir: {org_dir}?'
        )
    (decode_dir / f'decode_{enh}').mkdir(exist_ok=False)
    ivector_dir = calculate_ivectors(
        ivector_dir, dest_dir, org_dir, train_affix, dataset_dir,
        extractor_dir, model_data_type, num_jobs)
    run_process([
        'steps/nnet3/decode.sh', '--acwt', '1.0',
        '--post-decode-acwt', '10.0',
        '--extra-left-context', '0', '--extra-right-context', '0',
        '--extra-left-context-initial', '0', '--extra-right-context-final',
        '0',
        '--frames-per-chunk', '140', '--nj', str(num_jobs), '--cmd',
        f'{kaldi_cmd}', '--online-ivector-dir',
        str(ivector_dir), f'{model_dir}/tree_a_sp/graph_tgpr',
        str(dataset_dir), str(decode_dir / f'decode_{enh}')],
        cwd=str(dest_dir),
        stdout=None, stderr=None
    )
    print((
                  decode_dir / f'decode_{enh}' / 'scoring_kaldi' / 'best_wer'
          ).read_text())


def on_pc2():
    # len(Path(os.environ["CCS_NODEFILE"]).read_text().strip().split("\n"))

    # Better
    # if 'PC2SYSNAME' in os.environ:
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

    org_dir/model_dir: Folder of the trained model

    enh: Name of this experiment for the kaldi folders

    """
    org_dir = True
    model_dir = 'chain'
    audio_dir = None
    ivector_dir = True
    dataset = 'cv_dev93'
    model_data_type = 'sms'
    json_path = None
    enh = 'bss_beam'
    model_type = 'tdnn1a_sp'
    extractor_dir = None
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
def sms_image():
    model_data_type = 'sms_image'
    extractor_dir = 'nnet3/extractor'
    org_dir='/scratch/hpc-prf-nt1/jensheit/python_packages/sms_wsj/data/sms_image/s5'
    dataset = 'test_eval92'




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
def run(_config, _run, audio_dir):
    assert Path(kaldi_root).exists(), kaldi_root

    assert len(ex.current_run.observers) == 1, (
        'FileObserver` missing. Add a `FileObserver` with `-F foo/bar/`.'
    )
    base_dir = Path(ex.current_run.observers[0].basedir)
    if audio_dir is not None:
        audio_dir = base_dir / audio_dir
        assert audio_dir.exists(), audio_dir
    if isinstance(_config['org_dir'], bool):
        org_dir = kaldi_root / 'egs' / 'wsj' / 's5'
    else:
        org_dir = Path(_config['org_dir'])
        assert org_dir.exists(), org_dir

    sacred_dir = base_dir / str(_run._id)

    try:
        decode(model_dir=check_config_element(_config['model_dir']),
               dest_dir=base_dir,
               org_dir=org_dir,
               audio_dir=audio_dir,
               model_type=_config['model_type'],
               model_data_type=_config['model_data_type'],
               ivector_dir=check_config_element(_config['ivector_dir']),
               extractor_dir=check_config_element(_config['extractor_dir']),
               json_path=check_config_element(_config['json_path']),
               dataset=_config['dataset'],
               hires=_config['hires'],
               enh=check_config_element(_config['enh']),
               num_jobs=_config['num_jobs']
               )
    finally:
        write_ccsinfo_files(sacred_dir, reqid_file=None,
                            info_file='CCS_INFO_END', consider_mpi=False)
