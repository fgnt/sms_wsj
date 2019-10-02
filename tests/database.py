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
