from pathlib import Path
import json
import argparse
import tempfile
import sh
import re
import soundfile as sf
import time

from nt.io.data_dir import wsj
from nt.database import keys
from nt.io.audioread import read_nist_wsj


def make_unique_examples(unique_examples, existing_examples, set_name):
    example_ids = list(unique_examples.keys())
    for ex_id in example_ids:
        if ex_id in existing_examples:
            ex_id_unique = "{}_{}".format(set_name, ex_id)
            entry = unique_examples.pop(ex_id)
            entry['example_id'] = ex_id_unique
            unique_examples[ex_id_unique] = entry
    return unique_examples


def write_json(database_path: Path, json_path: Path):
    """
    Creates database structure and dumps it as JSON. 
    Database creation and set naming is taking from kaldi:
    KALDI_ROOT/egs/wsj/s5/local/wsj_data_prep.sh

    :param database_path: Path to WSJ database
    :param json_path: Path where JSON should be dumped

    """
    print("Start: {}".format(time.ctime()))
    database_path = Path(database_path).absolute()
    json_path = Path(json_path).absolute()
    database = create_database(database_path)
    Path.mkdir(json_path.parent, parents=True, exist_ok=True)
    with open(json_path, 'w') as fid:
        json.dump(database, fid, sort_keys=True, indent=4, ensure_ascii=False)
    print("Done: {}".format(time.ctime()))


def create_database(wsj_path: Path):
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
        "test_eval93_5k"  # 213 examples, has actually 215?!
    ]

    dev_sets = [
        ["13-34.1/wsj1/doc/indices/h1_p0.ndx"],
        ["13-34.1/wsj1/doc/indices/h2_p0.ndx"],
        ["13-16.1/wsj1/si_dt_20/"],
        ["13-16.1/wsj1/si_dt_05/"]
    ]
    dev_set_names = [
        "test_dev_93",  # 503 examples
        "test_dev_93_5k",  # 513 examples
        "dev_dt_20",  # 503 examples
        "dev_dt_05"  # 913 examples
    ]

    transcriptions = get_transcriptions(wsj_path, wsj_path)
    gender_mapping = get_gender_mapping(wsj_path)

    datasets = dict()
    examples = dict()

    datasets_tr, examples_tr = \
        create_official_datasets(train_sets,
                                 train_set_names,
                                 wsj_path, gender_mapping,
                                 transcriptions
                                 )

    datasets.update(datasets_tr)
    examples.update(examples_tr)

    datasets_dt, examples_dt = \
        create_official_datasets(dev_sets,
                                 dev_set_names,
                                 wsj_path, gender_mapping,
                                 transcriptions
                                 )

    datasets.update(datasets_dt)
    examples.update(examples_dt)

    datasets_et, examples_et = \
        create_official_datasets(test_sets,
                                 test_set_names,
                                 wsj_path, gender_mapping,
                                 transcriptions
                                 )

    datasets.update(datasets_et)
    examples.update(examples_et)

    database = {
        keys.DATASETS: datasets,
        keys.EXAMPLES: examples
    }

    return database


def create_official_datasets(official_sets, official_names, wsj_root,
                             genders,
                             transcript):

    _examples = dict()
    _datasets = dict()

    for idx, set_list in enumerate(official_sets):
        example_list = list()
        set_name = official_names[idx]
        for ods in set_list:
            set_path = wsj_root / ods
            if set_path.match('*.ndx'):
                _example = read_ndx(set_path, wsj_root,
                                    genders,
                                    transcript)
            else:
                wav_files = list(set_path.glob('*/*.wv1'))
                _example = process_example_paths(wav_files, genders,
                                                 transcript)
            _example = make_unique_examples(_example, _examples, set_name)
            example_list += list(_example.keys())
            _examples.update(_example)
        _datasets[set_name] = sorted(example_list)

    return _datasets, _examples


def read_ndx(ndx_file: Path, wsj_root, genders, transcript):
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
        disk = disk.replace('13-32.1', '13-33.1')  # wrong disk-ids for
        # test_eval93 and test_eval93_5k
        wav_path = wav_path.lstrip(' /')  # remove leading whitespace and
        # slash
        audio_path = str(wsj_root / disk / wav_path)
        if "11-2.1/wsj0/si_tr_s/401" in audio_path:
            continue  # skip 401 subdirectory in train sets
        fixed_paths.append(Path(audio_path))

    _examples = process_example_paths(fixed_paths, genders,
                                      transcript)

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
        info = read_nist_wsj(path, audioread_function=sf.info, verbose=True)
        nsamples = info.frames
        gender = genders[speaker_id]

        example = {
            keys.EXAMPLE_ID: example_id,
            keys.AUDIO_PATH: {
                keys.OBSERVATION: str(path)
            },
            keys.NUM_SAMPLES: {
                keys.OBSERVATION: nsamples
            },
            keys.SPEAKER_ID: speaker_id,
            keys.GENDER: gender,
            keys.TRANSCRIPTION: transcript['clean word'][example_id],
            keys.KALDI_TRANSCRIPTION: transcript['kaldi'][example_id]
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
            matches = re.findall("^(.+)\s+\((\S+)\)$", fid.read(),
                                 flags=re.M)
        word.update({utt_id: trans for trans, utt_id in matches})

    kaldi = dict()
    kaldi_wsj_data_dir = wsj_root / "kaldi_data"
    files = list(kaldi_wsj_data_dir.glob('*.txt'))
    for file in files:
        file_path = kaldi_wsj_data_dir / file
        with open(file_path) as fid:
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
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory).absolute()
        with open(temporary_directory / 'dirty.txt', 'w') as f:
            for key, value in transcriptions.items():
                f.write('{} {}\n'.format(key, value))
        result = sh.perl(
            sh.cat(str(temporary_directory / 'dirty.txt')),
            wsj_root / 'kaldi_tools' / 'normalize_transcript.pl',
            '<NOISE>'
        )
    result = [line.split(maxsplit=1) for line in result.strip().split('\n')]
    result = {k: v for k, v in result}
    return result


def get_gender_mapping(wsj_root: Path):

    spkrinfo = list(wsj_root.glob('*/wsj?/doc/**/*spkrinfo.txt')) + \
               list(wsj_root.glob('kaldi_data/**/*spkrinfo.txt'))

    _spkr_gender_mapping = dict()

    for path in spkrinfo:
        with open(path, 'r') as fid:
            for line in fid:
                if not (line.startswith(';') or line.startswith('---')):
                    line = line.split()
                    _spkr_gender_mapping[line[0].lower()] = keys.MALE \
                        if line[1] == 'M' else keys.FEMALE

    return _spkr_gender_mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--database_path', type=str, default=wsj)
    parser.add_argument('-j', '--json_path', type=str, default='wsj.json')
    args = parser.parse_args()
    write_json(database_path=args.database_path, json_path=args.json_path)
