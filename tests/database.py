import numpy as np
from sms_wsj.database.database import SmsWsj, AudioReader


def test_example():
    db = SmsWsj()
    ds = db.get_dataset('cv_dev93')
    ds = ds.map(AudioReader(rir=True))

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

    assert len(example['audio_data']) == 7
    assert example['audio_data']['observation'].shape == (6, 103650)
    assert example['audio_data']['noise_image'].shape == (6, 103650)
    assert example['audio_data']['speech_reverberation_early'].shape == (2, 6, 103650)
    assert example['audio_data']['speech_reverberation_tail'].shape == (2, 6, 103650)
    assert example['audio_data']['speech_source'].shape == (2, 103650)
    assert example['audio_data']['speech_image'].shape == (2, 6, 103650)
    assert example['audio_data']['rir'].shape == (2, 6, 8192)

    np.testing.assert_allclose(
        example['audio_data']['observation'],
        np.sum(
            example['audio_data']['speech_reverberation_early']
            + example['audio_data']['speech_reverberation_tail'],
            axis=0  # sum above speaker
        ) + example['audio_data']['noise_image'],
        atol=1e-7
    )

    assert list(example.keys()) == [
        'room_dimensions', 'sound_decay_time', 'source_position',
        'sensor_position', 'num_speakers', 'speaker_id', 'source_id',
        'gender', 'kaldi_transcription', 'log_weights', 'num_samples',
        'offset', 'audio_path', 'snr', 'example_id', 'dataset', 'audio_data']

    assert example['example_id'] == '4k0c0301_4k6c030t_0'
    assert example['snr'] == 29.749852569493584
    assert example['room_dimensions'] == [[8.169], [5.905], [3.073]]
    assert example['source_position'] == [[2.443, 2.71], [3.104, 2.135], [1.557, 1.557]]
    assert example['sensor_position'] == [[3.899, 3.8, 3.759, 3.817, 3.916, 3.957],
                        [3.199, 3.189, 3.098, 3.017, 3.027, 3.118],
                        [1.413, 1.418, 1.423, 1.423, 1.417, 1.413]]
    assert example['sound_decay_time'] == 0.354
    assert example['offset'] == [0, 17423]
    assert example['log_weights'] == [1.2027951449295022, -1.2027951449295022]
    assert example['num_samples'] == {'observation': 103650,
                                      'speech_source': [103650, 56335]}


def test_order():
    db = SmsWsj()

    ds = db.get_dataset('cv_dev93')
    for scenario_id, example in enumerate(ds):
        assert scenario_id == int(example['example_id'].split('_')[-1])

    ds = db.get_dataset('test_eval92')
    for scenario_id, example in enumerate(ds):
        assert scenario_id == int(example['example_id'].split('_')[-1])

    ds = db.get_dataset('train_si284')
    for scenario_id, example in enumerate(ds):
        assert scenario_id == int(example['example_id'].split('_')[-1])
