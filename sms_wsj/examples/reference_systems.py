"""

# Example call:
cd path/where/to/store/simulation/results

# Define how you want to parallelize.
# Note: The mixture models need significant more computation power.
# ccsalloc is an HPC scheduler, and this command requests 50 workers and each has 2GB memory
run="ccsalloc --res=rset=50:mem=2g:ncpus=1 --tracefile=ompi.%reqid.trace -t 2h ompi -- "
run="mpiexec -np 16 "

# For the experiments in the paper you need to run the following commands:
${run} python -m sms_wsj.examples.reference_systems with observation
${run} python -m sms_wsj.examples.reference_systems with mm_masking
${run} python -m sms_wsj.examples.reference_systems with mm_mvdr_souden
${run} python -m sms_wsj.examples.reference_systems with irm_mvdr
${run} python -m sms_wsj.examples.reference_systems with ibm_mvdr
${run} python -m sms_wsj.examples.reference_systems with image
${run} python -m sms_wsj.examples.reference_systems with image_early

"""

import numpy as np
from pathlib import Path
import sacred

from einops import rearrange

import dlp_mpi
from lazy_dataset import from_list

from nara_wpe.utils import stft as _stft, istft as _istft
from pb_bss.extraction import mask_module
from pb_bss.extraction import (
    apply_beamforming_vector,
    get_power_spectral_density_matrix,
    get_single_source_bf_vector,
)
from pb_bss.evaluation.wrapper import OutputMetrics
from pb_bss.distribution import CACGMMTrainer
from pb_bss import initializer
from pb_bss.permutation_alignment import DHTVPermutationAlignment

from sms_wsj.database import SmsWsj, AudioReader
from sms_wsj.io import dump_audio, dump_json

experiment = sacred.Experiment('Ref systems')


@experiment.config
def config():
    dataset = ['cv_dev93', 'test_eval92']  # or 'test_eval92'
    Observation = None

    stft_size = 512
    stft_shift = 128
    stft_window_length = None
    stft_window = 'hann'
    json_path = None


@experiment.named_config
def observation():
    out = 'observation'
    Observation = 'Observation'
    mask_estimator = 'IBM'
    beamformer = 'ch0'
    postfilter = None


@experiment.named_config
def mm_masking():
    out = 'mm_masking'
    Observation = 'Observation'
    mask_estimator = 'cacgmm'
    beamformer = 'ch0'
    postfilter = 'mask_mul'
    weight_constant_axis = -3  # pi_tk
    # weight_constant_axis = -1  # pi_fk


@experiment.named_config
def mm_mvdr_souden():
    out = 'mm_mvdr_souden'
    Observation = 'Observation'
    mask_estimator = 'cacgmm'
    beamformer = 'mvdr_souden'
    postfilter = None
    weight_constant_axis = -3  # pi_tk
    # weight_constant_axis = -1  # pi_fk

@experiment.named_config
def irm_mvdr():
    out = 'irm_mvdr'
    Observation = 'Observation'
    mask_estimator = 'IRM'
    beamformer = 'mvdr_souden'
    postfilter = None


@experiment.named_config
def ibm_mvdr():
    out = 'ibm_mvdr'
    Observation = 'Observation'
    mask_estimator = 'IBM'
    beamformer = 'mvdr_souden'
    postfilter = None


@experiment.named_config
def image():
    out = 'image'
    Observation = 'speech_image'
    # mask_estimator = 'ICM_0'
    mask_estimator = None
    beamformer = 'ch0'
    postfilter = None


@experiment.named_config
def image_mask():
    out = 'image_mask'
    Observation = 'Observation'
    mask_estimator = 'ICM_0'
    # mask_estimator = None
    beamformer = 'ch0'
    postfilter = 'mask_mul'


@experiment.named_config
def image_early():
    out = 'image_early'
    Observation = 'speech_reverberation_early'
    # mask_estimator = 'ICM_0_early'
    mask_estimator = None
    beamformer = 'ch0'
    postfilter = None


def get_multi_speaker_metrics(
    mask,  # T Ktarget F
    Observation,  # D T F (stft signal)
    speech_source,  # Ksource N (time signal)
    Speech_image=None,  # Ksource D T F (stft signal)
    Noise_image=None,  # D T F (stft signal)
    istft=None,  # callable(signal, num_samples=num_samples)
    bf_algorithm='mvdr_souden',
    postfilter=None,  # [None, 'mask_mul']
) -> OutputMetrics:
    """

    >>> from IPython.lib.pretty import pprint
    >>> from pb_bss.testing import dummy_data
    >>> from paderbox.transform.module_stft import stft, istft
    >>> from pb_bss.extraction import ideal_ratio_mask, phase_sensitive_mask
    >>> from pb_bss.extraction import ideal_complex_mask

    >>> example = dummy_data.reverberation_data()

    >>> Observation = stft(example['audio_data']['observation'])
    >>> Speech_image = stft(example['audio_data']['speech_image'])
    >>> Noise_image = stft(example['audio_data']['noise_image'])
    >>> speech_source = example['audio_data']['speech_source']

    >>> mask = ideal_ratio_mask(np.abs([*Speech_image, Noise_image]).sum(1))
    >>> X_mask = mask[:-1]
    >>> N_mask = mask[-1]
    >>> kwargs = {}
    >>> kwargs['mask'] = np.stack([*mask], 1)
    >>> kwargs['Observation'] = Observation
    >>> kwargs['Speech_image'] = Speech_image
    >>> kwargs['Noise_image'] = Noise_image
    >>> kwargs['speech_source'] = speech_source
    >>> kwargs['istft'] = istft
    >>> pprint(get_multi_speaker_metrics(**kwargs).as_dict())
    {'pesq': array([1.996, 2.105]),
     'stoi': array([0.8425774 , 0.86015112]),
     'mir_eval_sxr_sdr': array([13.82179099, 11.37128002]),
     'mir_eval_sxr_sir': array([21.39419702, 18.52582023]),
     'mir_eval_sxr_sar': array([14.68805087, 12.3606874 ]),
     'mir_eval_sxr_selection': array([0, 1]),
     'invasive_sxr_sdr': array([17.17792759, 14.49937822]),
     'invasive_sxr_sir': array([18.9065789 , 16.07738463]),
     'invasive_sxr_snr': array([22.01439067, 19.66127281])}
    >>> pprint(get_multi_speaker_metrics(**kwargs, postfilter='mask_mul').as_dict())
    {'pesq': array([2.235, 2.271]),
     'stoi': array([0.84173865, 0.85532424]),
     'mir_eval_sxr_sdr': array([14.17958101, 11.69826193]),
     'mir_eval_sxr_sir': array([29.62978561, 26.10579693]),
     'mir_eval_sxr_sar': array([14.3099193, 11.8692283]),
     'mir_eval_sxr_selection': array([0, 1]),
     'invasive_sxr_sdr': array([24.00659296, 20.80162802]),
     'invasive_sxr_sir': array([27.13945978, 24.21115858]),
     'invasive_sxr_snr': array([26.89769041, 23.44632734])}
    >>> pprint(get_multi_speaker_metrics(**kwargs, bf_algorithm='ch0', postfilter='mask_mul').as_dict())
    {'pesq': array([1.969, 2.018]),
     'stoi': array([0.81097215, 0.80093435]),
     'mir_eval_sxr_sdr': array([10.2343187 ,  8.29797827]),
     'mir_eval_sxr_sir': array([16.84226656, 14.64059341]),
     'mir_eval_sxr_sar': array([11.3932819 ,  9.59180288]),
     'mir_eval_sxr_selection': array([0, 1]),
     'invasive_sxr_sdr': array([14.70258429, 11.87061145]),
     'invasive_sxr_sir': array([14.74794743, 11.92701556]),
     'invasive_sxr_snr': array([34.53605847, 30.76351885])}

    >>> mask = ideal_ratio_mask(np.abs([*Speech_image, Noise_image])[:, 0])
    >>> kwargs['mask'] = np.stack([*mask], 1)
    >>> kwargs['speech_source'] = example['audio_data']['speech_image'][:, 0]
    >>> pprint(get_multi_speaker_metrics(**kwargs, bf_algorithm='ch0', postfilter='mask_mul').as_dict())
    {'pesq': array([3.471, 3.47 ]),
     'stoi': array([0.96011783, 0.96072581]),
     'mir_eval_sxr_sdr': array([13.50013349, 10.59091527]),
     'mir_eval_sxr_sir': array([17.67436882, 14.76824653]),
     'mir_eval_sxr_sar': array([15.66698718, 12.82478905]),
     'mir_eval_sxr_selection': array([0, 1]),
     'invasive_sxr_sdr': array([15.0283757 , 12.18546349]),
     'invasive_sxr_sir': array([15.07095641, 12.23764194]),
     'invasive_sxr_snr': array([35.13536337, 31.41445774])}

    >>> mask = phase_sensitive_mask(np.array([*Speech_image, Noise_image])[:, 0])
    >>> kwargs['mask'] = np.stack([*mask], 1)
    >>> kwargs['speech_source'] = example['audio_data']['speech_image'][:, 0]
    >>> pprint(get_multi_speaker_metrics(**kwargs, bf_algorithm='ch0', postfilter='mask_mul').as_dict())
    {'pesq': array([3.965, 3.968]),
     'stoi': array([0.98172316, 0.98371817]),
     'mir_eval_sxr_sdr': array([17.08649852, 14.51167667]),
     'mir_eval_sxr_sir': array([25.39489935, 24.17276323]),
     'mir_eval_sxr_sar': array([17.79271334, 15.0251782 ]),
     'mir_eval_sxr_selection': array([0, 1]),
     'invasive_sxr_sdr': array([14.67450877, 12.21865275]),
     'invasive_sxr_sir': array([14.77642923, 12.32843497]),
     'invasive_sxr_snr': array([31.02059848, 28.2459515 ])}
    >>> mask = ideal_complex_mask(np.array([*Speech_image, Noise_image])[:, 0])
    >>> kwargs['mask'] = np.stack([*mask], 1)
    >>> kwargs['speech_source'] = example['audio_data']['speech_image'][:, 0]
    >>> pprint(get_multi_speaker_metrics(**kwargs, bf_algorithm='ch0', postfilter='mask_mul').as_dict())
    {'pesq': array([4.549, 4.549]),
     'stoi': array([1., 1.]),
     'mir_eval_sxr_sdr': array([149.04269346, 147.03728106]),
     'mir_eval_sxr_sir': array([170.73079352, 168.36046824]),
     'mir_eval_sxr_sar': array([149.07223578, 147.06942287]),
     'mir_eval_sxr_selection': array([0, 1]),
     'invasive_sxr_sdr': array([12.32048218,  9.61471296]),
     'invasive_sxr_sir': array([12.41346788,  9.69274082]),
     'invasive_sxr_snr': array([29.06057363, 27.10901422])}

    """
    _, N = speech_source.shape
    K = mask.shape[-2]
    D, T, F = Observation.shape

    assert K < 10, (K, mask.shape, N, D, T, F)
    assert D < 30, (K, N, D, T, F)

    psds = get_power_spectral_density_matrix(
        rearrange(Observation, 'd t f -> f d t', d=D, t=T, f=F),
        rearrange(mask, 't k f -> f k t', k=K, t=T, f=F),
    )  # shape: f, ktarget, d, d

    assert psds.shape == (F, K, D, D), (psds.shape, (F, K, D, D))

    beamformers = list()
    for k_target in range(K):
        target_psd = psds[:, k_target]
        distortion_psd = np.sum(np.delete(psds, k_target, axis=1), axis=1)

        beamformers.append(
            get_single_source_bf_vector(
                bf_algorithm,
                target_psd_matrix=target_psd,
                noise_psd_matrix=distortion_psd,
            )
        )
    beamformers = np.stack(beamformers, axis=1)
    assert beamformers.shape == (F, K, D), (beamformers.shape, (F, K, D))

    def postfiler_fn(Signal):
        if postfilter is None:
            return Signal
        elif postfilter == 'mask_mul':
            return Signal * rearrange(mask, 't k f -> k f t', k=K, t=T, f=F)
        else:
            raise ValueError(postfilter)

    Speech_prediction = apply_beamforming_vector(
        vector=rearrange(beamformers, 'f k d -> k f d', k=K, d=D, f=F),
        mix=rearrange(Observation, 'd t f -> f d t', d=D, t=T, f=F),
    )
    Speech_prediction = postfiler_fn(Speech_prediction)
    speech_prediction = istft(rearrange(Speech_prediction, 'k f t -> k t f', k=K, t=T, f=F), num_samples=N)

    if Speech_image is None:
        speech_contribution = None
    else:
        Speech_contribution = apply_beamforming_vector(
            vector=rearrange(beamformers, 'f k d -> k f d', k=K, d=D, f=F),
            mix=rearrange(Speech_image, '(ksource k) d t f -> ksource k f d t', k=1, d=D, t=T, f=F),
        )
        Speech_contribution = postfiler_fn(Speech_contribution)
        # ksource in [K-1, K]
        speech_contribution = istft(rearrange(Speech_contribution, 'ksource k f t -> ksource k t f', k=K, t=T, f=F), num_samples=N)

    if Noise_image is None:
        noise_contribution = None
    else:
        Noise_contribution = apply_beamforming_vector(
            vector=rearrange(beamformers, 'f k d -> k f d', k=K, d=D, f=F),
            mix=rearrange(Noise_image, '(k d) t f -> k f d t', k=1, d=D, t=T, f=F),
        )
        Noise_contribution = postfiler_fn(Noise_contribution)
        noise_contribution = istft(rearrange(Noise_contribution, 'k f t -> k t f', k=K, t=T, f=F), num_samples=N)

    metric = OutputMetrics(
            speech_prediction=speech_prediction,
            speech_source=speech_source,
            speech_contribution=speech_contribution,
            noise_contribution=noise_contribution,
            sample_rate=8000,
            enable_si_sdr=False,
    )

    return metric


@experiment.capture
def get_dataset(dataset, json_path):
    """
    >>> from IPython.lib.pretty import pprint
    >>> np.set_string_function(lambda a: f'array(shape={a.shape}, dtype={a.dtype})')
    >>> pprint(get_dataset('cv_dev93')[0])  # doctest: +ELLIPSIS
    {...
     'example_id': '0_4k6c0303_4k4c0319',
     ...
     'snr': 23.287502642941252,
     'dataset': 'cv_dev93',
     'audio_data': {'observation': array(shape=(6, 93389), dtype=float64),
      'speech_source': array(shape=(2, 93389), dtype=float64),
      'speech_reverberation_early': array(shape=(2, 6, 93389), dtype=float64),
      'speech_reverberation_tail': array(shape=(2, 6, 93389), dtype=float64),
      'noise_image': array(shape=(6, 93389), dtype=float64),
      'speech_image': array(shape=(2, 6, 93389), dtype=float64),
      'Speech_source': array(shape=(2, 733, 257), dtype=complex128),
      'Speech_reverberation_early': array(shape=(2, 6, 733, 257), dtype=complex128),
      'Speech_reverberation_tail': array(shape=(2, 6, 733, 257), dtype=complex128),
      'Speech_image': array(shape=(2, 6, 733, 257), dtype=complex128),
      'Noise_image': array(shape=(6, 733, 257), dtype=complex128),
      'Observation': array(shape=(6, 733, 257), dtype=complex128)}}

    """
    db = SmsWsj(json_path=json_path)
    ds = db.get_dataset(dataset)
    ds = ds.map(AudioReader())

    def calculate_stfts(ex):
        ex['audio_data']['Speech_source'] = stft(ex['audio_data']['speech_source'])
        ex['audio_data']['Speech_reverberation_early'] = stft(ex['audio_data']['speech_reverberation_early'])
        ex['audio_data']['Speech_reverberation_tail'] = stft(ex['audio_data']['speech_reverberation_tail'])
        ex['audio_data']['Speech_image'] = stft(ex['audio_data']['speech_image'])
        ex['audio_data']['Noise_image'] = stft(ex['audio_data']['noise_image'])
        ex['audio_data']['Observation'] = stft(ex['audio_data']['observation'])
        return ex

    return ds.map(calculate_stfts)


@experiment.capture
def stft(
        signal,
        *,
        stft_size=512,
        stft_shift=128,
        stft_window_length=None,
        stft_window='hann',
):
    return _stft(
        signal,
        size=stft_size,
        shift=stft_shift,
        window_length=stft_window_length,
        window=stft_window,
    )


@experiment.capture
def istft(
        signal,
        num_samples,
        *,
        stft_size=512,
        stft_shift=128,
        stft_window_length=None,
        stft_window='hann',
):
    time_signal = _istft(
        signal,
        size=stft_size,
        shift=stft_shift,
        window_length=stft_window_length,
        window=stft_window,
        # num_samples=num_samples,  # this stft does not support num_samples
    )

    pad = True
    if pad:
        assert time_signal.shape[-1] >= num_samples, (time_signal.shape, num_samples)
        assert time_signal.shape[-1] < num_samples + stft_shift, (time_signal.shape, num_samples)
        time_signal = time_signal[..., :num_samples]
    else:
        raise ValueError(
            pad,
            'When padding is False in the stft, the signal is cutted.'
            'This operation can not be inverted.',
        )
    return time_signal


def get_scores(
        ex,
        mask,

        Observation='Observation',

        beamformer='mvdr_souden',
        postfilter = None,
):
    """
    Calculate the scores, where the prediction/estimated signal is tested
    against the source/desired signal.
    This function is for oracle test to figure out, which metric can work with
    source signal.

    SI-SDR does not work, when the desired signal is the signal before the
    room impulse response and give strange results, when the channel is
    changed.

    Example:

        >>> from IPython.lib.pretty import pprint
        >>> ex = get_dataset('cv_dev93')[0]
        >>> mask = get_mask_from_oracle(ex, 'IBM')
        >>> metric, result = get_scores(ex, mask)
        >>> pprint(result)
        {'pesq': array([2.014, 1.78 ]),
         'stoi': array([0.68236465, 0.61319396]),
         'mir_eval_sxr_sdr': array([10.23933413, 10.01566298]),
         'invasive_sxr_sdr': array([15.76439393, 13.86230425])}
    """

    if Observation == 'Observation':
        metric = get_multi_speaker_metrics(
            mask=rearrange(mask, 'k t f -> t k f'),  # T Ktarget F
            Observation=ex['audio_data'][Observation],  # D T F (stft signal)
            speech_source=ex['audio_data']['speech_source'],  # Ksource N (time signal)
            Speech_image=ex['audio_data']['Speech_image'],  # Ksource D T F (stft signal)
            Noise_image=ex['audio_data']['Noise_image'],  # D T F (stft signal)
            istft=istft,  # callable(signal, num_samples=num_samples)
            bf_algorithm=beamformer,
            postfilter=postfilter,  # [None, 'mask_mul']
        )
    else:
        assert mask is None, mask
        assert beamformer == 'ch0', beamformer
        assert postfilter is None, postfilter
        metric = OutputMetrics(
            speech_prediction=ex['audio_data'][Observation][:, 0],
            speech_source=ex['audio_data']['speech_source'],
            # speech_contribution=speech_contribution,
            # noise_contribution=noise_contribution,
            sample_rate=8000,
            enable_si_sdr=False,
        )

    result = metric.as_dict()
    del result['mir_eval_selection']
    del result['mir_eval_sar']
    del result['mir_eval_sir']
    if 'invasive_sxr_sir' in result:
        del result['invasive_sir']
        del result['invasive_snr']

    return metric, result


@experiment.capture
def get_mask_from_cacgmm(
        ex,  # (D, T, F)
        weight_constant_axis=-1,
):  # (K, T, F)
    """

    Args:
        observation:

    Returns:

    >>> from nara_wpe.utils import stft
    >>> y = get_dataset('cv_dev93')[0]['audio_data']['observation']
    >>> Y = stft(y, size=512, shift=128)
    >>> get_mask_from_cacgmm(Y).shape
    (3, 813, 257)

    """
    Observation = ex['audio_data']['Observation']
    Observation = rearrange(Observation, 'd t f -> f t d')

    trainer = CACGMMTrainer()

    initialization: 'F, K, T' = initializer.iid.dirichlet_uniform(
        Observation,
        num_classes=3,
        permutation_free=False,
    )

    pa = DHTVPermutationAlignment.from_stft_size(512)

    affiliation = trainer.fit_predict(
        Observation,
        initialization=initialization,
        weight_constant_axis=weight_constant_axis,
        inline_permutation_aligner=pa if weight_constant_axis != -1 else None
    )

    mapping = pa.calculate_mapping(
        rearrange(affiliation, 'f k t ->k f t'))

    affiliation = rearrange(pa.apply_mapping(
        rearrange(affiliation, 'f k t ->k f t'), mapping
    ), 'k f t -> k t f')

    return affiliation


def get_mask_from_oracle(
        ex,
        mask_estimator

):  # (K, T, F)
    """

    Args:
        ex:
        mask_estimator:

    Returns:

    >>> ex = get_dataset('cv_dev93')[0]
    >>> mask = get_mask_from_oracle(ex, 'ICM_0_early')
    >>> mask.shape
    (2, 733, 257)

    >>> obs = np.sum(ex['audio_data']['speech_reverberation_early'] + ex['audio_data']['speech_reverberation_tail'], axis=0) + ex['audio_data']['noise_image']
    >>> np.testing.assert_allclose(ex['audio_data']['observation'], obs, atol=1e-7)
    >>> Obs = np.sum(ex['audio_data']['Speech_reverberation_early'] + ex['audio_data']['Speech_reverberation_tail'], axis=0) + ex['audio_data']['Noise_image']
    >>> np.testing.assert_allclose(ex['audio_data']['Observation'], Obs, atol=2e-7)

    >>> Speech_reverberation_early_0 = ex['audio_data']['Observation'][0] * mask
    >>> np.testing.assert_allclose(Speech_reverberation_early_0, ex['audio_data']['Speech_reverberation_early'][:, 0], atol=1e-13, rtol=1e-13)
    >>> speech_reverberation_early_0 = istft(Speech_reverberation_early_0, num_samples=obs.shape[-1])
    >>> np.testing.assert_allclose(speech_reverberation_early_0, ex['audio_data']['speech_reverberation_early'][:, 0], atol=1e-13, rtol=1e-13)

    >>> mask = get_mask_from_oracle(ex, 'ICM_0')
    >>> mask.shape
    (2, 733, 257)
    >>> Speech_reverberation_early_0 = ex['audio_data']['Observation'][0] * mask
    >>> np.testing.assert_allclose(Speech_reverberation_early_0, ex['audio_data']['Speech_image'][:, 0], atol=1e-13, rtol=1e-13)

    >>> mask = get_mask_from_oracle(ex, 'IBM')
    >>> mask.shape
    (3, 733, 257)

    >>> mask = get_mask_from_oracle(ex, 'IRM')
    >>> mask.shape
    (3, 733, 257)

    """
    from pb_bss.extraction.mask_module import (
        ideal_ratio_mask,
        ideal_complex_mask,
        ideal_binary_mask,
    )

    if mask_estimator == 'ICM_0_early':
        # K, D, T, F =  ex['audio_data']['Speech_reverberation_early'].shape
        return ideal_complex_mask(
            [
            *ex['audio_data']['Speech_reverberation_early'][..., 0, :, :],
            # np.sum(ex['audio_data']['Speech_reverberation_tail'][..., 0, :, :]
            # , axis=0) + ex['audio_data']['Noise_image'][..., 0, :, :]
            ex['audio_data']['Observation'][0, :, :]
            - np.sum(
                ex['audio_data']['Speech_reverberation_early'][..., 0, :, :],
                axis=0)
            ]
        )[:-1, ...]
    elif mask_estimator == 'ICM_0':
        return ideal_complex_mask(
            [
            *ex['audio_data']['Speech_image'][..., 0, :, :],
            ex['audio_data']['Observation'][0, :, :]
            - np.sum(
                ex['audio_data']['Speech_image'][..., 0, :, :],
                axis=0)
            ]
        )[:-1, ...]
    elif mask_estimator in ['IBM', 'IRM']:
        signal = np.sqrt(
            np.abs(
                (np.array([
                    *ex['audio_data']['Speech_image'],
                    ex['audio_data']['Noise_image']
                ]) ** 2)
            ).sum(axis=1)
        )
        if mask_estimator == 'IRM':
            return mask_module.ideal_ratio_mask(signal)
        elif mask_estimator == 'IBM':
            mask = mask_module.ideal_binary_mask(signal)
            mask = np.clip(mask, 1e-10, 1 - 1e-10)
            return mask
        else:
            raise NotImplementedError(mask_module)
    else:
        raise NotImplementedError(mask_module)


@experiment.automain
def main(
        _run,
        out,
        mask_estimator,
        Observation,
        beamformer,
        postfilter,
        normalize_audio=True,
):
    if dlp_mpi.IS_MASTER:
        from sacred.commands import print_config
        print_config(_run)

    ds = get_dataset()

    data = []

    out = Path(out)

    for ex in dlp_mpi.split_managed(ds.sort(), allow_single_worker=True):

        if mask_estimator is None:
            mask = None
        elif mask_estimator == 'cacgmm':
            mask = get_mask_from_cacgmm(ex)
        else:
            mask = get_mask_from_oracle(ex, mask_estimator)

        metric, score = get_scores(
            ex,
            mask,
            Observation=Observation,
            beamformer=beamformer,
            postfilter=postfilter,
        )
        
        est0, est1 = metric.speech_prediction_selection
        dump_audio(est0, out / ex['dataset'] / f"{ex['example_id']}_0.wav", normalize=normalize_audio)
        dump_audio(est1, out / ex['dataset'] / f"{ex['example_id']}_1.wav", normalize=normalize_audio)

        data.append(dict(
            example_id=ex['example_id'],
            value=score,
            dataset=ex['dataset'],
        ))

        # print(score, repr(score))

    data = dlp_mpi.gather(data)

    if dlp_mpi.IS_MASTER:
        data = [
            entry
            for worker_data in data
            for entry in worker_data
        ]

        data = {  # itertools.groupby expect an order
            dataset: list(subset)
            for dataset, subset in from_list(data).groupby(
                lambda ex: ex['dataset']
            ).items()
        }

        for dataset, sub_data in data.items():
            print(f'Write details to {out}.')
            dump_json(sub_data, out / f'{dataset}_scores.json')

        for dataset, sub_data in data.items():
            summary = {}
            for k in sub_data[0]['value'].keys():
                m = np.mean([
                    d['value'][k]
                    for d in sub_data
                ])
                print(dataset, k, m)
                summary[k] = m
            dump_json(summary, out / f'{dataset}_summary.json')
