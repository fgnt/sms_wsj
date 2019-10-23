"""
Example calls:
python -m sms_wsj.database.wsj.write_wav --database_dir-dir /destination/dir --json-path /path/to/sms_wsj.json


"""
import json
import os
import re
import tempfile
from pathlib import Path

import sacred
import sh
import soundfile as sf

ex = sacred.Experiment('Create wsj json')
kaldi_root = Path(os.environ['KALDI_ROOT'])
kaldi_wsj_egs_dir = kaldi_root / 'egs' / 'wsj' / 's5'
kaldi_wsj_data_dir = kaldi_wsj_egs_dir / 'data' / 'local' / 'data'
kaldi_wsj_tools = kaldi_wsj_egs_dir / 'data' / 'local' / 'data'

def create_official_datasets(
        official_sets, official_names, wsj_root, as_wav, genders, transcript
):

    _examples = dict()

    for idx, set_list in enumerate(official_sets):
        set_name = official_names[idx]
        _examples[set_name] = dict()
        for ods in set_list:
            set_path = wsj_root / ods
            if set_path.match('*.ndx'):
                _example = read_ndx(
                    set_path, wsj_root, as_wav, genders, transcript
                )
            else:
                if as_wav:
                    wav_files = list(set_path.glob('*/*.wav'))
                else:
                    wav_files = list(set_path.glob('*/*.wv1'))
                _example = process_example_paths(
                    wav_files, genders, transcript
                )
            _examples[set_name].update(_example)

    return _examples


def read_nsamples(audio_path):

    if audio_path.suffix == '.wv1':
        f = open(audio_path, 'rb')
        header = f.read(1024).decode("utf-8")  # nist header is a multiple of
        # 1024 bytes
        nsamples = int(re.search("sample_count -i (.+?)\n", header).group(1))
    else:
        info = sf.info(str(audio_path), verbose=True)
        nsamples = info.frames
    return nsamples


def read_ndx(ndx_file: Path, wsj_root, as_wav,
             genders, transcript):
    assert ndx_file.match('*.ndx')

    with open(ndx_file) as fid:
        if ndx_file.match('*/si_et_20.ndx') or \
                ndx_file.match('*/si_et_05.ndx'):
            lines = [line.rstrip() + ".wv1" for line in fid
                     if not line.startswith(";")]
        else:
            lines = [line.rstrip() for line in fid
                     if line.lower().rstrip().endswith(".wv1")
                     ]

    fixed_paths = list()

    for line in lines:
        disk, wav_path = line.split(':')
        disk = '{}-{}.{}'.format(*disk.split('_'))

        # wrong disk-ids for test_eval93 and test_eval93_5k
        disk = disk.replace('13-32.1', '13-33.1')
        wav_path = wav_path.lstrip(' /')  # remove leading whitespace and
        # slash
        audio_path = wsj_root / disk / wav_path
        if as_wav:
            audio_path = audio_path.with_suffix('.wav')
        if "11-2.1/wsj0/si_tr_s/401" in str(audio_path):
            continue  # skip 401 subdirectory in train sets
        fixed_paths.append(audio_path)

    _examples = process_example_paths(fixed_paths, genders, transcript)

    return _examples


def process_example_paths(example_paths, genders, transcript):
    """
    Creates an entry in keys.EXAMPLE for every example in `example_paths`

    :param example_paths: List of Paths to example .wv files
    :type: List
    :param genders: Mapping from speaker id to gender
    :type: dict
    :param transcript: Mapping from raw example id to dirty, clean and kaldi
        transcription
    :type: dict

    :return _examples: Partial entries in keys.EXAMPLE for examples in
        `set_name`
    :type: dict
    """
    _examples = dict()

    for path in example_paths:

        wav_file = path.parts[-1]
        example_id = wav_file.split('.')[0]

        speaker_id = example_id[0:3]
        nsamples = read_nsamples(path)
        gender = genders[speaker_id]

        example = {
            'example_id': example_id,
            'audio_path': {
                'observation': str(path)
            },
            'num_samples': {
                'observation': nsamples
            },
            'speaker_id': speaker_id,
            'gender': gender,
            'transcription': transcript['clean word'][example_id],
            'kaldi_transcription': transcript['kaldi'][example_id]
        }

        _examples[example_id] = example

    return _examples


def get_transcriptions(root: Path, wsj_root: Path):
    word = dict()

    dot_files = list(root.rglob('*.dot'))
    ptx_files = list(root.rglob('*.ptx'))
    ptx_files = [ptx_file for ptx_file in ptx_files if Path(
        str(ptx_file).replace('.ptx', '.dot')) not in dot_files]

    for file_path in dot_files + ptx_files:
        with open(file_path) as fid:
            matches = re.findall("^(.+)\s+\((\S+)\)$", fid.read(), flags=re.M)
        word.update({utt_id: trans for trans, utt_id in matches})

    kaldi = dict()
    files = list(kaldi_wsj_data_dir.glob('*.txt'))
    for file in files:
        with open(file) as fid:
            matches = re.findall("^(\S+) (.+)$", fid.read(), flags=re.M)
        kaldi.update({utt_id: trans for utt_id, trans in matches})

    data_dict = dict()
    data_dict["word"] = word
    data_dict["clean word"] = normalize_transcription(word, wsj_root)
    data_dict["kaldi"] = kaldi
    return data_dict


def normalize_transcription(transcriptions, wsj_root: Path):
    """ Passes the dirty transcription dict to a Kaldi Perl script for cleanup.

    We use the original Perl file, to make sure, that the cleanup is done
    exactly as it is done by Kaldi.

    :param transcriptions: Dirty transcription dictionary
    :param wsj_root: Path to WSJ database

    :return result: Clean transcription dictionary
    """
    assert len(transcriptions) > 0, 'No transcriptions to clean up.'
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory).absolute()
        with open(temporary_directory / 'dirty.txt', 'w') as f:
            for key, value in transcriptions.items():
                f.write('{} {}\n'.format(key, value))
        result = sh.perl(
            sh.cat(str(temporary_directory / 'dirty.txt')),
            kaldi_wsj_tools / 'normalize_transcript.pl',
            '<NOISE>'
        )
    result = [line.split(maxsplit=1) for line in result.strip().split('\n')]
    result = {k: v for k, v in result}
    return result


def get_gender_mapping(wsj_root: Path):

    spkrinfo = list(wsj_root.glob('*/wsj?/doc/**/*spkrinfo.txt')) + \
               list(kaldi_wsj_data_dir.glob('**/*spkrinfo.txt'))

    _spkr_gender_mapping = dict()

    for path in spkrinfo:
        with open(path, 'r') as fid:
            for line in fid:
                if not (line.startswith(';') or line.startswith('---')):
                    line = line.split()
                    _spkr_gender_mapping[line[0].lower()] = 'male' \
                        if line[1] == 'M' else 'female'

    return _spkr_gender_mapping


@ex.config
def config():
    database_dir = None
    json_path = None
    wsj_json = None
    as_wav = True
    assert database_dir is not None, 'You have to specify the database dir'
    assert json_path is not None, 'You have to specify a path for the new json'


@ex.automain
def create_database(database_dir, json_path, as_wav):
    database_dir = Path(database_dir).expanduser().resolve()
    json_path = Path(json_path).expanduser().resolve()
    if json_path.exists():
        raise FileExistsError(json_path)
    assert database_dir.exists(), database_dir

    database_dir = Path(database_dir)

    train_sets = [
        ["11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx"],
        ["13-34.1/wsj1/doc/indices/si_tr_s.ndx",
         "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx"]
    ]
    train_set_names = [
        "train_si84",  # 7138 examples
        "train_si284"  # 37416 examples
    ]

    test_sets = [
        ["11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx"],
        ["11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx"],
        ["13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx"],
        ["13-32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx"]
    ]

    test_set_names = [
        "test_eval92",  # 333 examples
        "test_eval92_5k",  # 330 examples
        "test_eval93",  # 213 examples
        "test_eval93_5k"  # 215 examples
    ]

    dev_sets = [
        ["13-34.1/wsj1/doc/indices/h1_p0.ndx"],
        ["13-34.1/wsj1/doc/indices/h2_p0.ndx"],
    ]
    dev_set_names = [
        "cv_dev93",  # 503 examples
        "cv_dev93_5k",  # 513 examples
    ]

    transcriptions = get_transcriptions(database_dir, database_dir)
    gender_mapping = get_gender_mapping(database_dir)

    examples = dict()

    examples_tr = create_official_datasets(
        train_sets,
        train_set_names,
        database_dir,
        as_wav,
        gender_mapping,
        transcriptions
    )
    examples.update(examples_tr)

    examples_dt = create_official_datasets(
        dev_sets,
        dev_set_names,
        database_dir,
        as_wav, gender_mapping,
        transcriptions
    )
    examples.update(examples_dt)

    examples_et = create_official_datasets(
        test_sets,
        test_set_names,
        database_dir,
        as_wav,
        gender_mapping,
        transcriptions
    )
    examples.update(examples_et)

    database = {
        'datasets': examples,
    }
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with json_path.open('w') as f:
        json.dump(database, f, indent=4, ensure_ascii=False)
    print(f'{json_path} written')
