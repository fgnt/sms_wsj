import re
from os import listdir, path, walk
import tempfile
import sh

from pathlib import Path
from collections import defaultdict
import concurrent.futures

from nt.database.helper import dump_database_as_json
from nt.io.data_dir import wsj
from nt.io.audioread import read_nist_wsj


def main():
    main_path = str(wsj)

    scenarios, flists = get_data_sets(main_path)
    scenarios["train"].update(get_official_train_sets(main_path))
    scenarios["test"].update(get_official_test_sets(main_path))
    scenarios["dev"].update(get_official_dev_sets(main_path))

    annotations = create_annotations(scenarios, debug=1)

    transcriptions = get_transcriptions(main_path, main_path)

    data = {
        'test': {'annotations': annotations['test'], 'flists': {"wave": scenarios["test"]}},
        'train': {'annotations': annotations['train'], 'flists': {"wave": scenarios["train"]}},
        'dev': {'annotations': annotations['dev'], 'flists': {"wave": scenarios["dev"]}},
        'orth': transcriptions,
        'flists': flists
    }

    # Creating the wsj.json file
    dump_database_as_json('wsj.json', data)


def update_dict(target_dict, dict_to_add):
    for key, value in dict_to_add.items():
        if key not in target_dict:
            target_dict.update({key: value})
        elif isinstance(target_dict[key], dict) and isinstance(value, dict):
            update_dict(target_dict[key], value)
        elif target_dict[key] == value:
            pass
        else:
            print("Conflicting key: {}".format(key))
            print("Conflicting values: {}!={}".format(target_dict[key], value))


def get_data_sets(root):
    data_sets = dict()
    flists = list()

    for dir_name in listdir(root):
        dir_file = path.join(root, dir_name, dir_name + ".dir")
        if not path.isfile(dir_file):
            continue

        with open(dir_file) as fid:
            lines = [line.rstrip() for line in fid
                     if line.lower().rstrip().endswith(".wv1") or
                     line.lower().rstrip().endswith(".wv2")]
        for line in lines:
            if dir_name == '13_16_1':
                line = line.upper()
            file_path = path.join(root, dir_name, line.replace("./", ""))
            if "11_2_1/wsj0/si_tr_s/401" in file_path:
                continue
            path_elements = line.split("/")
            utt_id, file_type = path_elements[-1].split(".")
            utt_id = utt_id.lower()
            file_type = file_type.lower()
            if file_type == "wv1":
                channel = "ch1"
            elif file_type == "wv2":
                channel = "ch2"
            else:
                raise Exception

            set_name = path_elements[2].lower()
            set_type = set_name.split("_")[1]
            if set_type == "tr":
                set_type = "train"
            elif set_type == "dt":
                set_type = "dev"
            else:
                set_type = "test"

            if path_elements[3].lower() != utt_id[:3]:
                set_name += "/" + path_elements[3].lower()

            update_dict(data_sets,
                        {set_type: {set_name: {
                            utt_id: {"observed": {channel: file_path}}}}})

            if (set_type + "/flists/wave/" + set_name) not in flists:
                flists.append(set_type + "/flists/wave/" + set_name)

    return data_sets, flists


def read_ndx(wsj_root, ndx_file):
    assert ndx_file.endswith('.ndx')

    data_dict = dict()
    with open(ndx_file) as fid:
        if ndx_file.endswith('si_et_20.ndx') or \
                ndx_file.endswith('si_et_05.ndx'):
            lines = [line.rstrip() + ".wv1" for line in fid
                     if not line.startswith(";")]
        else:
            lines = [line.rstrip() for line in fid
                     if line.lower().rstrip().endswith(".wv1") or
                     line.lower().rstrip().endswith(".wv2")]

    for line in lines:
        if line.startswith("13_16_1"):
            line = line.upper()
        file_path = path.join(
            wsj_root, line.replace(": ", "").replace(":", "/"))
        if "11_2_1/wsj0/si_tr_s/401" in file_path:
            continue
        utt_id, file_type = path.basename(file_path).split(".")
        utt_id = utt_id.lower()
        file_type = file_type.lower()
        if file_type == "wv1":
            channel = "ch1"
        elif file_type == "wv2":
            channel = "ch2"
        else:
            raise Exception

        update_dict(data_dict,
                    {utt_id: {"observed": {channel: file_path}}})
    return data_dict


def get_official_train_sets(wsj_root):
    official_train_sets = [
        "13_34_1/wsj1/doc/indices/si_tr_s.ndx",
        "11_13_1/wsj0/doc/indices/train/tr_s_wv1.ndx"
    ]
    official_train_set_names = [
        "official_si_284",
        "official_si_84"
    ]

    data_dict = dict()
    for idx in range(0, len(official_train_sets)):
        ndx_file = path.join(wsj_root, official_train_sets[idx])
        set_name = official_train_set_names[idx]
        data_dict[set_name] = read_ndx(wsj_root, ndx_file)

    update_dict(data_dict["official_si_284"], data_dict["official_si_84"])

    return data_dict


def get_official_test_sets(wsj_root):
    official_test_sets = [
        "11_13_1/wsj0/doc/indices/test/nvp/si_et_20.ndx",
        "11_13_1/wsj0/doc/indices/test/nvp/si_et_05.ndx",
        "13_32_1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx",
        "13_32_1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx"
    ]

    official_test_set_names = [
        "official_si_et_20",
        "official_si_et_05",
        "official_si_et_h1/wsj64k",
        "official_si_et_h2/wsj5k"
    ]

    data_dict = dict()

    for idx in range(0, len(official_test_sets)):
        ndx_file = path.join(wsj_root, official_test_sets[idx])
        set_name = official_test_set_names[idx]
        data_dict[set_name] = read_ndx(wsj_root, ndx_file)

    return data_dict


def get_official_dev_sets(wsj_root):
    official_dev_sets = [
        "13_34_1/wsj1/doc/indices/h1_p0.ndx",
        "13_34_1/wsj1/doc/indices/h2_p0.ndx",
        "13_16_1/WSJ1/SI_DT_20/",
        "13_16_1/WSJ1/SI_DT_05/"
    ]
    official_dev_set_names = [
        "official_si_dt_20",
        "official_si_dt_05",
        "official_Dev-set_Hub_1",
        "official_Dev-set_Hub_2"
    ]

    data_dict = dict()

    for idx in range(0, len(official_dev_sets)):
        set_path = path.join(wsj_root, official_dev_sets[idx])
        set_name = official_dev_set_names[idx]
        if set_path.endswith('.ndx'):
            ndx_file = path.join(wsj_root, official_dev_sets[idx])
            set_name = official_dev_set_names[idx]
            data_dict[set_name] = read_ndx(wsj_root, ndx_file)
        else:
            for subdir, dirs, files in walk(set_path):
                for file in files:
                    if not file.endswith(".WV1"):
                        continue
                    utt_id = file.split(".")[0].lower()
                    file_path = path.join(wsj_root, subdir, file)

                    update_dict(data_dict,
                                {set_name: {
                                    utt_id: {"observed": {"ch1": file_path}}}})

    return data_dict


def get_transcriptions(root, wsj_root):
    word = dict()
    for subdir, _, files in walk(root):
        dot_files = [file for file in files if file.endswith(".dot")]
        ptx_files = [file for file in files if file.endswith(".ptx") and
                     file.replace(".ptx", ".dot") not in dot_files]

        for file in dot_files + ptx_files:
            file_path = path.join(root, subdir, file)
            with open(file_path) as fid:
                matches = re.findall("^(.+)\s+\((\S+)\)$", fid.read(),
                                     flags=re.M)
            word.update({utt_id: trans for trans, utt_id in matches})

    kaldi = dict()
    kaldi_wsj_data_dir = path.join(wsj_root, "kaldi_data")
    files = [path.join(kaldi_wsj_data_dir, file)
             for file in listdir(kaldi_wsj_data_dir)
             if path.isfile(path.join(kaldi_wsj_data_dir, file)) and
             file.endswith(".txt")]
    for file in files:
        file_path = path.join(kaldi_wsj_data_dir, file)
        with open(file_path) as fid:
            matches = re.findall("^(\S+) (.+)$", fid.read(), flags=re.M)
        kaldi.update({utt_id: trans for utt_id, trans in matches})

    data_dict = dict()
    data_dict["word"] = word
    print('Start')
    data_dict["clean word"] = normalize_transcription(word)
    print('Stop')
    data_dict["kaldi"] = kaldi
    return data_dict


def normalize_transcription(transcriptions):
    """ Passes the dirty transcription dict to a Kaldi Perl script for cleanup.

    We use the original Perl file, to make sure, that the cleanup is done
    exactly as it is done by Kaldi.

    Args:
        transcriptions: Dirty transcription dictionary.

    Returns: Clean transcription dictionary.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        with (temporary_directory / 'dirty.txt').open('w') as f:
            for key, value in transcriptions.items():
                f.write('{} {}\n'.format(key, value))
        result = sh.perl(
            sh.cat(temporary_directory / 'dirty.txt'),
            wsj / 'kaldi_tools' / 'normalize_transcript.pl',
            '<NOISE>'
        )
    result = [line.split(maxsplit=1) for line in result.strip().split('\n')]
    result = {k: v for k, v in result}
    return result


def create_annot_for_flist(ds, scenarios, debug=0):

    i = 1

    nsamples = dict()
    stage = ds[0]
    dataset = ds[1]

    if dataset.startswith('official'):
        if debug >= 1:
            print('Creating annotations for {}\n'.format(dataset))

        file_path = scenarios[stage][dataset]

        for utt in file_path:
            nsamples_list = list()
            for feature_channel in file_path[utt].keys():
                for channel in file_path[utt][feature_channel].keys():
                    p = file_path[utt][feature_channel][channel]
                    try:
                        audio = read_nist_wsj(p)
                        nsamples_list.append(len(audio))
                    except OSError:  # Could not open file
                        nsamples_list.append(0)
            nsamples[utt] = max(nsamples_list)

            if debug >= 2:
                print('{}: {} / {}'.format(dataset, i, len(file_path)))
                i += 1

        if debug >= 1:
            print('\n {} finished \n'.format(dataset))

        return stage, dataset, nsamples

    else:
        return None


def create_annotations(scenarios, debug=0):

    annotations = {'train': defaultdict(lambda: defaultdict(dict)),
                   'test': defaultdict(lambda: defaultdict(dict)),
                   'dev': defaultdict(lambda: defaultdict(dict))}

    if debug >= 1:
        print('Creating annotations...\n')

    datasets = [(stage, ds) for stage in list(scenarios.keys()) for ds in list(scenarios[stage].keys())]
    print(datasets)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

        future_tasks = [executor.submit(create_annot_for_flist, ds, scenarios, debug) for ds in datasets]

        for future in concurrent.futures.as_completed(future_tasks):
            result = future.result()
            if result is not None:
                stage = result[0]
                dataset = result[1]
                nsamples = result[2]

                for utt, length in nsamples.items():
                    annotations[stage][dataset][utt]['nsamples'] = length

    return annotations


if __name__ == '__main__':
    main()
