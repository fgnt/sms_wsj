import numpy as np

import lazy_dataset
from sms_wsj.database.create_intermediate_json import combine_rirs_and_sources


def filter_duplicates(l):
    """
    >>> filter_duplicates([{'a': 1}, {'b': 1}, {'a': 1}])
    [{'a': 1}, {'b': 1}]
    """
    def make_hashable(o):
        try:
            hash(o)
            return o
        except TypeError:
            return helper[type(o)](o)

    helper = {
        set: lambda o: tuple([make_hashable(e) for e in o]),
        tuple: lambda o: tuple([make_hashable(e) for e in o]),
        list: lambda o: tuple([make_hashable(e) for e in o]),
        dict: lambda o: frozenset(
            [(make_hashable(k), make_hashable(v)) for k, v in
             o.items()]),
    }

    l = list(l)

    return list({
        hashable: entry
        for hashable, entry in zip(make_hashable(l), l)
    }.values())


def split_rirs_and_sources(
        ds
):
    """
    Split a dataset in the rir and source dataset.
    These datasets can be used to recreate a dataset with combine_rirs_and_sources.
    The `dataset_name` argument can be used to either get the same dataset
    or a random new dataset with different utterance pairs.

        db = SmsWsj(...)
        ds = db.get_dataset('train_si284')
        rir_ds, source_ds = split_rirs_and_sources(ds)
        ds_new = lazy_dataset.new(combine_rirs_and_sources(rir_ds, source_ds, 2, f'train_si284_rng{np.random.randint(0, 2**32)}'))

    Note: The new dataset have to use the `scenario_map_fn`, because the
          observation and the intermediate signal have to be calculated on
          demand.

    >>> import os, re
    >>> from paderbox.utils.pretty import pprint
    >>> from pprint import pprint, pformat
    >>> from sms_wsj.database.database import SmsWsj
    >>> def print_ex(ex):
    ...     print(re.sub(r"'[^']+(?=/(?:wsj_8k_zeromean|speech_source|early|tail|rirs|noise|observation)/[^']+.wav')", r"'...", pformat(ex)))
    >>> db = SmsWsj(os.environ.get('NT_DATABASE_JSONS_DIR') + '/sms_wsj.json')
    >>> ds = db.get_dataset('cv_dev93')
    >>> print_ex(ds[0])
    {'audio_path': {'noise_image': '.../noise/cv_dev93/0_4k6c0303_4k4c0319.wav',
                    'observation': '.../observation/cv_dev93/0_4k6c0303_4k4c0319.wav',
                    'original_source': ['.../wsj_8k_zeromean/13-16.1/wsj1/si_dt_20/4k6/4k6c0303.wav',
                                        '.../wsj_8k_zeromean/13-16.1/wsj1/si_dt_20/4k4/4k4c0319.wav'],
                    'rir': ['.../rirs/cv_dev93/0/h_0.wav',
                            '.../rirs/cv_dev93/0/h_1.wav'],
                    'speech_reverberation_early': ['.../early/cv_dev93/0_4k6c0303_4k4c0319_0.wav',
                                                   '.../early/cv_dev93/0_4k6c0303_4k4c0319_1.wav'],
                    'speech_reverberation_tail': ['.../tail/cv_dev93/0_4k6c0303_4k4c0319_0.wav',
                                                  '.../tail/cv_dev93/0_4k6c0303_4k4c0319_1.wav'],
                    'speech_source': ['.../speech_source/cv_dev93/0_4k6c0303_4k4c0319_0.wav',
                                      '.../speech_source/cv_dev93/0_4k6c0303_4k4c0319_1.wav']},
     'dataset': 'cv_dev93',
     'example_id': '0_4k6c0303_4k4c0319',
     'gender': ['male', 'female'],
     'kaldi_transcription': ['IN ADDITION TO DEFORESTATION EXAMPLES ARE',
                             'THE PROFIT HAS BEEN PLOWED BACK INTO THE BANK WHICH '
                             'HAS PURSUED ITS MISSION TO REBUILD A DECAYING '
                             'NEIGHBORHOOD WITH A SINGULAR FOCUS'],
     'log_weights': [0.9885484337248203, -0.9885484337248203],
     'num_samples': {'observation': 93389, 'original_source': [31633, 93389]},
     'num_speakers': 2,
     'offset': [52476, 0],
     'room_dimensions': [[8.169], [5.905], [3.073]],
     'sensor_position': [[4.015, 3.973, 4.03, 4.129, 4.172, 4.115],
                         [3.265, 3.175, 3.093, 3.102, 3.192, 3.274],
                         [1.55, 1.556, 1.563, 1.563, 1.558, 1.551]],
     'snr': 23.287502642941252,
     'sound_decay_time': 0.387,
     'source_id': ['4k6c0303', '4k4c0319'],
     'source_position': [[3.312, 3.0], [1.921, 2.379], [1.557, 1.557]],
     'speaker_id': ['4k6', '4k4']}
    >>> rir_ds, source_ds = split_rirs_and_sources(ds)
    >>> print_ex(rir_ds[0])
    {'audio_path': {'rir': ['.../rirs/cv_dev93/0/h_0.wav',
                            '.../rirs/cv_dev93/0/h_1.wav']},
     'example_id': '0',
     'room_dimensions': [[8.169], [5.905], [3.073]],
     'sensor_position': [[4.015, 3.973, 4.03, 4.129, 4.172, 4.115],
                         [3.265, 3.175, 3.093, 3.102, 3.192, 3.274],
                         [1.55, 1.556, 1.563, 1.563, 1.558, 1.551]],
     'sound_decay_time': 0.387,
     'source_position': [[3.312, 3.0], [1.921, 2.379], [1.557, 1.557]]}
    >>> print_ex(source_ds[0])
    {'audio_path': {'observation': '.../wsj_8k_zeromean/13-16.1/wsj1/si_dt_20/4k6/4k6c0303.wav'},
     'example_id': '4k6c0303',
     'gender': 'male',
     'kaldi_transcription': 'IN ADDITION TO DEFORESTATION EXAMPLES ARE',
     'num_samples': 31633,
     'speaker_id': '4k6'}
    >>> ds_new = lazy_dataset.new(combine_rirs_and_sources(rir_ds, source_ds, 2, 'cv_dev93'))

    >>> print_ex(ds_new[0])
    {'audio_path': {'original_source': ['.../wsj_8k_zeromean/13-16.1/wsj1/si_dt_20/4k6/4k6c0303.wav',
                                        '.../wsj_8k_zeromean/13-16.1/wsj1/si_dt_20/4k4/4k4c0319.wav'],
                    'rir': ['.../rirs/cv_dev93/0/h_0.wav',
                            '.../rirs/cv_dev93/0/h_1.wav']},
     'dataset': 'cv_dev93',
     'example_id': '0_4k6c0303_4k4c0319',
     'gender': ['male', 'female'],
     'kaldi_transcription': ['IN ADDITION TO DEFORESTATION EXAMPLES ARE',
                             'THE PROFIT HAS BEEN PLOWED BACK INTO THE BANK WHICH '
                             'HAS PURSUED ITS MISSION TO REBUILD A DECAYING '
                             'NEIGHBORHOOD WITH A SINGULAR FOCUS'],
     'log_weights': [0.9885484337248203, -0.9885484337248203],
     'num_samples': {'observation': 93389, 'original_source': [31633, 93389]},
     'num_speakers': 2,
     'offset': [52476, 0],
     'room_dimensions': [[8.169], [5.905], [3.073]],
     'sensor_position': [[4.015, 3.973, 4.03, 4.129, 4.172, 4.115],
                         [3.265, 3.175, 3.093, 3.102, 3.192, 3.274],
                         [1.55, 1.556, 1.563, 1.563, 1.558, 1.551]],
     'sound_decay_time': 0.387,
     'source_id': ['4k6c0303', '4k4c0319'],
     'source_position': [[3.312, 3.0], [1.921, 2.379], [1.557, 1.557]],
     'speaker_id': ['4k6', '4k4']}
    >>> ex_new, ex = ds_new[0], ds[0]
    >>> del ex['snr']
    >>> for k in ['speech_reverberation_early', 'speech_source', 'noise_image',
    ...           'observation', 'speech_reverberation_tail']:
    ...     del ex['audio_path'][k]
    >>> set.symmetric_difference(set(ex_new.keys()), set(ex.keys()))
    set()
    >>> assert ex_new == ex
    >>> from paderbox.utils.nested import flatten
    >>> ex_new, ex = flatten(ex_new), flatten(ex)
    >>> set.symmetric_difference(set(ex_new.keys()), set(ex.keys()))
    set()
    >>> for k in sorted(set(ex_new.keys()) | set(ex.keys())):
    ...     assert ex_new[k] == ex[k], (k, ex_new[k], ex[k])

    >>> ds_new = lazy_dataset.new(combine_rirs_and_sources(rir_ds, source_ds, 2, 'cv_dev93_rng1'))
    >>> print_ex(ds_new[0])
    {'audio_path': {'original_source': ['.../wsj_8k_zeromean/13-16.1/wsj1/si_dt_20/4k2/4k2c031a.wav',
                                        '.../wsj_8k_zeromean/13-16.1/wsj1/si_dt_20/4k3/4k3c0316.wav'],
                    'rir': ['.../rirs/cv_dev93/0/h_0.wav',
                            '.../rirs/cv_dev93/0/h_1.wav']},
     'dataset': 'cv_dev93_rng1',
     'example_id': '0_4k2c031a_4k3c0316',
     'gender': ['female', 'female'],
     'kaldi_transcription': ['THE S. E. C. IS MANEUVERING TO E- CURB WHAT FUNDS '
                             'CAN SAY IN NEWSLETTERS JUST AS HOLDERS ARE DEMANDING '
                             'MORE INFORMATION',
                             'IN MISSOURI STATE PARTY LEADERS ARE ACTIVELY '
                             'COURTING DEMOCRATS WHO VOTED FOR MR. REAGAN'],
     'log_weights': [1.7891449809156743, -1.7891449809156748],
     'num_samples': {'observation': 59008, 'original_source': [58326, 59008]},
     'num_speakers': 2,
     'offset': [659, 0],
     'room_dimensions': [[8.169], [5.905], [3.073]],
     'sensor_position': [[4.015, 3.973, 4.03, 4.129, 4.172, 4.115],
                         [3.265, 3.175, 3.093, 3.102, 3.192, 3.274],
                         [1.55, 1.556, 1.563, 1.563, 1.558, 1.551]],
     'sound_decay_time': 0.387,
     'source_id': ['4k2c031a', '4k3c0316'],
     'source_position': [[3.312, 3.0], [1.921, 2.379], [1.557, 1.557]],
     'speaker_id': ['4k2', '4k3']}
    """
    def get_sources(ex):
        num_speakers = len(ex['speaker_id'])

        for spk_idx in range(num_speakers):
            example_id = ex['source_id'][spk_idx]
            yield (
                example_id,
                {
                    'audio_path': {
                        'observation': (ex['audio_path'].get('original_source') or
                                        ex['audio_path']['speech_source'])[
                            spk_idx],
                    },
                    'example_id': example_id,
                    **{
                        k: ex[k][spk_idx]
                        for k in ['gender', 'kaldi_transcription', 'speaker_id']
                    },
                    'num_samples': (ex['num_samples'].get('original_source') or
                                    ex['num_samples']['speech_source'])[spk_idx],
                },
            )

    source_ds = lazy_dataset.new(dict(list(ds.map(get_sources).unbatch())))

    def get_rir_ex(ex):
        example_id = ex['example_id'].split('_')[0]
        return (
            example_id,
            {
                'audio_path': {'rir': ex['audio_path']['rir']},
                'example_id': example_id,
                **{
                    k: ex[k]
                    for k in
                    ['room_dimensions', 'sound_decay_time', 'source_position',
                     'sensor_position']
                },
            }
        )

    rir_ds = lazy_dataset.new(dict(list(ds.map(get_rir_ex))))
    assert len(rir_ds) == len(ds), (len(rir_ds), len(ds))

    return rir_ds, source_ds


class SMSWSJRandomDataset(lazy_dataset.Dataset):
    """
    >>> import os, re
    >>> from paderbox.utils.pretty import pprint
    >>> from paderbox.utils.timer import Timer
    >>> from pprint import pprint, pformat
    >>> from sms_wsj.database.database import SmsWsj
    >>> def print_ex(ex):
    ...     print(re.sub(r"'[^']+(?=/(?:wsj_8k_zeromean|early|tail|rirs|noise|observation)/[^']+.wav')", r"'...", pformat(ex)))
    >>> db = SmsWsj(os.environ.get('NT_DATABASE_JSONS_DIR') + '/sms_wsj.json')
    >>> ds = db.get_dataset('train_si284')
    >>> ds = SMSWSJRandomDataset(ds)
    >>> with Timer() as t:
    ...     ds = ds.copy(freeze=True)
    >>> print(t)
    <class 'paderbox.utils.timer.Timer'>: 16.2 s
    >>> print_ex(ds[0])
    {'audio_path': {'original_source': ['.../wsj_8k_zeromean/13-6.1/wsj1/si_tr_s/498/498c040z.wav',
                                        '.../wsj_8k_zeromean/13-3.1/wsj1/si_tr_s/478/478c040q.wav'],
                    'rir': ['.../rirs/train_si284/0/h_0.wav',
                            '.../rirs/train_si284/0/h_1.wav']},
     'dataset': 'cv_dev93_rng4193246114',
     'example_id': '0_498c040z_478c040q',
     'gender': ['female', 'female'],
     'kaldi_transcription': ['THUS LESLEY ASKS CAN THERE BE ANY DOUBT THAT JESUS '
                             'IS ALSO ON THE SIDE OF THE A. N. C.',
                             "<NOISE> AFTER ALL HE ISN'T THE ONE WHO HAS TO RISK "
                             'GETTING HIT <NOISE> OVER THE HEAD WITH A METAL PIPE'],
     'log_weights': [-0.5361178511234126, 0.5361178511234126],
     'num_samples': {'observation': 52086, 'original_source': [52086, 42571]},
     'num_speakers': 2,
     'offset': [0, 2547],
     'room_dimensions': [[7.875], [5.839], [3.088]],
     'sensor_position': [[3.974, 3.923, 3.823, 3.774, 3.825, 3.925],
                         [2.979, 3.065, 3.063, 2.976, 2.89, 2.891],
                         [1.418, 1.421, 1.426, 1.427, 1.424, 1.42]],
     'sound_decay_time': 0.413,
     'source_id': ['498c040z', '478c040q'],
     'source_position': [[3.81, 5.333], [1.919, 3.777], [1.423, 1.423]],
     'speaker_id': ['498', '478']}


    """
    def __init__(self, dataset, num_speakers=2, rng=np.random):
        self.dataset = dataset
        dataset_name = set(dataset.map(lambda ex: ex['dataset']))
        assert len(dataset_name) == 1, dataset_name
        self.dataset_name = dataset_name.pop()
        self.num_speakers = num_speakers
        self.rng = rng
        self.rir_ds, self.source_ds = split_rirs_and_sources(dataset)

    def get_new_dataset(self):
        return lazy_dataset.new(combine_rirs_and_sources(
            self.rir_ds, self.source_ds, self.num_speakers,
            f'{self.dataset_name}_rng{self.rng.randint(0, 2**32)}'))

    def copy(self, freeze: bool = False) -> 'lazy_dataset.Dataset':
        if freeze:
            return self.get_new_dataset()
        else:
            return SMSWSJRandomDataset(
                self.dataset, self.num_speakers, self.rng,
            )

    def __iter__(self):
        return iter(self.get_new_dataset())

    def __len__(self):
        return len(self.dataset)

    # ToDo: Implement getitem for str.
    #
