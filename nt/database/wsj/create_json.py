import re
from os import listdir, path, walk

from nt.database.helper import dump_database_as_json
from nt.io.data_dir import wsj


def main():
    main_path = str(wsj)

    scenarios, flists = get_data_sets(main_path)
    scenarios["train"].update(get_official_train_sets(main_path))
    scenarios["test"].update(get_official_test_sets(main_path))
    scenarios["dev"].update(get_official_dev_sets(main_path))

    transcriptions = get_transcriptions(main_path, main_path)

    data = {'test': {'flists': {"wave": scenarios["test"]}},
            'train': {'flists': {"wave": scenarios["train"]}},
            'dev': {'flists': {"wave": scenarios["dev"]}},
            'orth': transcriptions,
            'flists': (flists)
            }

    # creating the wsj.json file
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
        dir_file = path.join(root, dir_name, dir_name+".dir")
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
                        {set_type: {set_name: {utt_id: {"observed": {channel: file_path}}}}})

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
    official_train_set_names =[
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
                                {set_name: {utt_id: {"observed": {"ch1": file_path}}}})

    return data_dict


def get_transcriptions(root, wsj_root):
    word = dict()
    clean_word = dict()
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
            clean_word.update({utt_id: normalize_transcription(trans)
                               for trans, utt_id in matches})

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
    data_dict["clean word"] = clean_word
    data_dict["kaldi"] = kaldi
    return data_dict


def normalize_transcription(trans):
    """ converts a word transcription by removing additional Information like
    noises or written punctuation marks

    :type trans: string
    :param trans: contains the string that gets converted into a clean
     transcription
    :return: the clean version of the string as a string
    """
    trans = trans.upper()  # Upcase everything to match the CMU dictionary.
    trans = trans.replace('\\', '')
    trans = trans.replace('\r', '')
    trans = trans.replace('%PERCENT', 'PERCENT')
    trans = trans.replace('.POINT', 'POINT')
    trans = re.sub('\[<\w+\]', '', trans)
    trans = re.sub('\[\w+>\]', '', trans)
    trans = re.sub('\[\w+/\]', '', trans)
    trans = re.sub('\[/\w+\]', '', trans)
    trans = trans.replace(' ~ ', ' ').replace(' . ', ' ').replace(' .', '')
    trans = re.sub('\[\w+\]', 'n', trans)
    trans = trans.replace(' n ', ' ')
    trans = re.sub('^n ', '', trans)
    trans = re.sub(' n$', '', trans)
    trans = trans.replace('--DASH', '-DASH')

    def _repl(matchobj):
        return matchobj.group(1)
    trans = re.sub("<([\w\']+)>", _repl, trans)
    trans = trans.replace('OFFICALS', 'OFFICIALS')
    trans = trans.replace('EXISITING', 'EXISTING')
    trans = trans.replace('GOVERMENT\'S', 'GOVERNMENT\'S')
    trans = trans.replace('GRAMOPHONEPERIOD', 'GRAMAPHONEPERIOD')
    trans = trans.replace(' ~', ' ')
    trans = trans.replace('~ ', '')
    trans = trans.replace('*', '')
    trans = re.sub('^\. ', '', trans)

    return trans


if __name__ == '__main__':
    main()
