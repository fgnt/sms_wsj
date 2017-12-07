import os
import glob
import json
import argparse
import tempfile
import sh
import re

from nt.io.data_dir import wsj
from nt.database import keys
from nt.io.audioread import read_nist_wsj, getparams


def write_json(database_path, json_path):
    database_path = os.path.abspath(database_path)
    json_path = os.path.abspath(json_path)
    database = create_database(database_path)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as fid:
        json.dump(database, fid, sort_keys=True, indent=4, ensure_ascii=False)


def create_database(wsj_path):
    official_train_sets = [
        ["13-34.1/wsj1/doc/indices/si_tr_s.ndx",
         "11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx"],
        ["11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx"]
    ]
    official_train_set_names = [
        "official_si_tr_284",
        "official_si_tr_84"
    ]

    official_test_sets = [
        ["11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx"],
        ["11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx"],
        # ["13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx"],
        # ["13-32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx"]
    ]

    official_test_set_names = [
        "official_si_et_20",
        "official_si_et_05",
        # "official_si_et_h1/wsj64k",
        # "official_si_et_h2/wsj5k"
    ]

    official_dev_sets = [
        ["13-34.1/wsj1/doc/indices/h1_p0.ndx"],
        ["13-34.1/wsj1/doc/indices/h2_p0.ndx"],
        ["13-16.1/wsj1/si_dt_20/"],
        ["13-16.1/wsj1/si_dt_05/"]
    ]
    official_dev_set_names = [
        "official_si_dt_20",
        "official_si_dt_05",
        "official_Dev-set_Hub_1",
        "official_Dev-set_Hub_2"
    ]

    transcriptions = get_transcriptions(wsj_path, wsj_path)
    gender_mapping = get_gender_mapping(wsj_path)

    datasets = dict()
    examples = dict()

    datasets_tr, examples_tr = \
        create_official_datasets(official_train_sets,
                                 official_train_set_names,
                                 wsj_path, gender_mapping,
                                 transcriptions
                                 )

    datasets.update(datasets_tr)
    examples.update(examples_tr)

    datasets_dt, examples_dt = \
        create_official_datasets(official_dev_sets,
                                 official_dev_set_names,
                                 wsj_path, gender_mapping,
                                 transcriptions
                                 )

    datasets.update(datasets_dt)
    examples.update(examples_dt)

    datasets_et, examples_et = \
        create_official_datasets(official_test_sets,
                                 official_test_set_names,
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


def create_official_datasets(official_sets, official_names, wsj_root, genders,
                             transcript):

    _examples = dict()
    _datasets = dict()

    for idx, set_list in enumerate(official_sets):
        example_list = list()
        set_name = official_names[idx]
        for ods in set_list:
            set_path = os.path.join(wsj_root, ods)
            if set_path.endswith('.ndx'):
                _example = read_ndx(set_path, wsj_root, set_name, genders,
                                    transcript)
            else:
                wav_files = glob.glob(os.path.join(set_path, '*/*.wv?'))
                _example = process_example_paths(wav_files, set_name, genders,
                                                 transcript)
            example_list += list(_example.keys())
            _examples.update(_example)
        _datasets[set_name] = sorted(example_list)

    return _datasets, _examples


def read_ndx(ndx_file, wsj_root, set_name, genders, transcript):
    assert ndx_file.endswith('.ndx')

    with open(ndx_file) as fid:
        if ndx_file.endswith('si_et_20.ndx') or \
                ndx_file.endswith('si_et_05.ndx'):
            lines = [line.rstrip() + ".wv1" for line in fid
                     if not line.startswith(";")]
        else:
            lines = [line.rstrip() for line in fid
                     if line.lower().rstrip().endswith(".wv1") or
                     line.lower().rstrip().endswith(".wv2")]

    fixed_paths = list()

    for line in lines:
        disk, wav_path = line.split(':')
        disk = '{}-{}.{}'.format(*disk.split('_'))
        wav_path = wav_path.lstrip(' /')  # remove leading whitespace and
        # slash
        audio_path = os.path.join(wsj_root, disk, wav_path)
        if "11-2.1/wsj0/si_tr_s/401" in audio_path:
            continue
        fixed_paths.append(audio_path)

    _examples = process_example_paths(fixed_paths, set_name, genders,
                                      transcript)

    return _examples


def process_example_paths(example_paths, set_name, genders, transcript):
    """
    Creates an entry in keys.EXAMPLE for every example in `example_paths`

    :param example_paths: List of paths to example .wv files
    :type: List
    :param set_name: Dataset name which accounts the examples
    :type: String
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
    set_name = '_'.join(set_name.split('_')[1:])

    for path in example_paths:

        wav_file = os.path.split(path)[-1]
        raw_example_id = wav_file.split('.')[0]

        if '_tr_' or '_dt_' in set_name:
            # ensure unique example ids in train sets because
            # 'official_si_tr_284' contains all examples from
            # 'official_si_tr_84'
            # dev sets have overlapping examples
            example_id = '{}_{}'.format(set_name, raw_example_id)
        else:
            example_id = raw_example_id

        channel = wav_file[-1]
        speaker_id = raw_example_id[0:3]
        params = read_nist_wsj(path, audioread_function=getparams)
        nsamples = params[3]
        gender = genders[speaker_id]

        example = {
            keys.EXAMPLE_ID: example_id,
            keys.AUDIO_PATH: {
                keys.OBSERVATION: {
                    "{c}{no}".format(c=keys.CHANNEL,
                                     no=channel): path
                }
            },
            keys.NUM_SAMPLES: {
                keys.OBSERVATION: nsamples
            },
            keys.SPEAKER_ID: speaker_id,
            keys.GENDER: gender,
            keys.TRANSCRIPTION: transcript['clean word'][raw_example_id],
            keys.KALDI_TRANSCRIPTION: transcript['kaldi'][raw_example_id]
        }

        if example_id in _examples:
            # add second channel <example_id>.wv2 (or .wv1 if .wv2 was processed
            # first)
            _examples[example_id][keys.AUDIO_PATH][keys.OBSERVATION].update(
                example[keys.AUDIO_PATH][keys.OBSERVATION]
            )
        else:
            _examples[example_id] = example

    return _examples


def get_transcriptions(root, wsj_root):
    word = dict()
    for subdir, _, files in os.walk(root):
        dot_files = [file for file in files if file.endswith(".dot")]
        ptx_files = [file for file in files if file.endswith(".ptx") and
                     file.replace(".ptx", ".dot") not in dot_files]

        for file in dot_files + ptx_files:
            file_path = os.path.join(root, subdir, file)
            with open(file_path) as fid:
                matches = re.findall("^(.+)\s+\((\S+)\)$", fid.read(),
                                     flags=re.M)
            word.update({utt_id: trans for trans, utt_id in matches})

    kaldi = dict()
    kaldi_wsj_data_dir = os.path.join(wsj_root, "kaldi_data")
    files = [os.path.join(kaldi_wsj_data_dir, file)
             for file in os.listdir(kaldi_wsj_data_dir)
             if os.path.isfile(os.path.join(kaldi_wsj_data_dir, file)) and
             file.endswith(".txt")]
    for file in files:
        file_path = os.path.join(kaldi_wsj_data_dir, file)
        with open(file_path) as fid:
            matches = re.findall("^(\S+) (.+)$", fid.read(), flags=re.M)
        kaldi.update({utt_id: trans for utt_id, trans in matches})

    data_dict = dict()
    data_dict["word"] = word
    data_dict["clean word"] = normalize_transcription(word)
    data_dict["kaldi"] = kaldi
    return data_dict


def normalize_transcription(transcriptions):
    """ Passes the dirty transcription dict to a Kaldi Perl script for cleanup.

    We use the original Perl file, to make sure, that the cleanup is done
    exactly as it is done by Kaldi.

    :param transcriptions: Dirty transcription dictionary

    :return result: Clean transcription dictionary
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = os.path.abspath(temporary_directory)
        with open(os.path.join(temporary_directory, 'dirty.txt'), 'w') as f:
            for key, value in transcriptions.items():
                f.write('{} {}\n'.format(key, value))
        result = sh.perl(
            sh.cat(os.path.join(temporary_directory, 'dirty.txt')),
            wsj / 'kaldi_tools' / 'normalize_transcript.pl',
            '<NOISE>'
        )
    result = [line.split(maxsplit=1) for line in result.strip().split('\n')]
    result = {k: v for k, v in result}
    return result


def get_gender_mapping(wsj_root):

    spkrinfo = glob.glob(os.path.join(wsj_root, '*/wsj?/doc/**/*spkrinfo.txt'),
                         recursive=True) + \
               glob.glob(os.path.join(wsj_root, 'kaldi_data/**/*spkrinfo.txt'),
                         recursive=True)

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
