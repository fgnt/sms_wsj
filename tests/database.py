from pathlib import Path

import numpy as np

import sms_wsj
from sms_wsj.database.database import SmsWsj, AudioReader


json_path = Path(sms_wsj.__file__) / 'cache' / 'sms_wsj.json'


def test_first_example():
    db = SmsWsj(json_path)
    ds = db.get_dataset('cv_dev93')
    ds = ds.map(AudioReader(AudioReader.all_keys))

    example = ds[0]

    np.testing.assert_allclose(
        example['audio_data']['observation'],
        np.sum(
            example['audio_data']['speech_reverberation_early']
            + example['audio_data']['speech_reverberation_tail'],
            axis=0  # sum above speaker
        ) + example['audio_data']['noise_image'],
        atol=1e-7
    )

    assert len(example['audio_data']) == 8
    assert example['audio_data']['observation'].shape == (6, 93389)
    assert example['audio_data']['noise_image'].shape == (6, 93389)
    assert example['audio_data']['speech_reverberation_early'].shape == (2, 6, 93389)
    assert example['audio_data']['speech_reverberation_tail'].shape == (2, 6, 93389)
    assert example['audio_data']['speech_source'].shape == (2, 93389)
    assert example['audio_data']['speech_image'].shape == (2, 6, 93389)
    assert example['audio_data']['rir'].shape == (2, 6, 8192)
    assert example['audio_data']['original_source'][0].shape == (31633,)
    assert example['audio_data']['original_source'][1].shape == (93389,)

    assert list(example.keys()) == [
        'room_dimensions', 'sound_decay_time', 'source_position',
        'sensor_position', 'example_id', 'num_speakers', 'speaker_id',
        'source_id', 'gender', 'kaldi_transcription', 'log_weights',
        'num_samples', 'offset', 'audio_path', 'snr', 'dataset', 'audio_data'
    ]

    assert example['example_id'] == '0_4k6c0303_4k4c0319'
    assert example['snr'] == 23.287502642941252
    assert example['room_dimensions'] == [[8.169], [5.905], [3.073]]
    assert example['source_position'] == [[3.312, 3.0], [1.921, 2.379], [1.557, 1.557]]
    assert example['sensor_position'] == [
        [4.015, 3.973, 4.03, 4.129, 4.172, 4.115],
        [3.265, 3.175, 3.093, 3.102, 3.192, 3.274],
        [1.55, 1.556, 1.563, 1.563, 1.558, 1.551]]
    assert example['sound_decay_time'] == 0.387
    assert example['offset'] == [52476, 0]
    assert example['log_weights'] == [0.9885484337248203, -0.9885484337248203]
    assert example['num_samples'] == {'observation': 93389,
                                      'original_source': [31633, 93389]}


def test_random_example():
    db = SmsWsj(json_path)
    ds = db.get_dataset('cv_dev93')
    ds = ds.map(AudioReader(AudioReader.all_keys))

    example = ds.random_choice()

    np.testing.assert_allclose(
        example['audio_data']['observation'],
        np.sum(
            example['audio_data']['speech_reverberation_early']
            + example['audio_data']['speech_reverberation_tail'],
            axis=0  # sum above speaker
        ) + example['audio_data']['noise_image'],
        atol=1e-7
    )


def test_order():
    db = SmsWsj(json_path)

    ds = db.get_dataset('cv_dev93')
    for scenario_id, example in enumerate(ds):
        assert scenario_id == int(example['example_id'].split('_')[0])

    ds = db.get_dataset('test_eval92')
    for scenario_id, example in enumerate(ds):
        assert scenario_id == int(example['example_id'].split('_')[0])

    ds = db.get_dataset('train_si284')
    for scenario_id, example in enumerate(ds):
        assert scenario_id == int(example['example_id'].split('_')[0])
