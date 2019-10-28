"""

### Description how to start this script

# Define how you want to parallelize.
# ccsalloc is an HPC scheduler, and this command requests 100 workers and each has 2GB memory
run="ccsalloc --group=hpc-prf-nt1 --res=rset=100:mem=2g:ncpus=1 --tracefile=ompi.%reqid.trace -t 2h ompi -- "
# To run on the machine, where you are logged in, use mpiexec:
run="mpiexec -np 16 "

# To start the experiment you can then execute the following commands.
# They will generate two files in your current working directory.
${run} python -m sms_wsj.examples.metric_target_comparison with dataset=cv_dev93 out=oracle_experiment_dev.json  # takes approx 30 min with 16 workers
${run} python -m sms_wsj.examples.metric_target_comparison with dataset=test_eval92 out=oracle_experiment_eval.json  # takes approx 40 min with 16 workers

# Display the summary again from the json:
python -m sms_wsj.examples.metric_target_comparison summary with out=oracle_experiment_dev.json
python -m sms_wsj.examples.metric_target_comparison summary with out=oracle_experiment_eval.json

"""

import json

from IPython.lib.pretty import pprint
import numpy as np
import pandas as pd
import sacred

from pb_bss.evaluation.wrapper import OutputMetrics
import dlp_mpi

from sms_wsj.io import dump_json
from sms_wsj.database import SmsWsj, AudioReader

experiment = sacred.Experiment('Oracle Experiment')


@experiment.config
def config():
    dataset = 'cv_dev93'  # or 'test_eval92'
    out = None  # json file to write the detailed results


@experiment.capture
def get_dataset(dataset):
    """
    >>> np.set_string_function(lambda a: f'array(shape={a.shape}, dtype={a.dtype})')
    >>> pprint(get_dataset('cv_dev93')[0])  # doctest: +ELLIPSIS
    {'audio_path': ...,
     ...,
     'example_id': '4k0c0301_4k6c030t_0',
     ...,
     'kaldi_transcription': ...,
     ...,
     'audio_data': {'speech_source': array(shape=(2, 103650), dtype=float64),
      'rir': array(shape=(2, 6, 8192), dtype=float64),
      'speech_image': array(shape=(2, 6, 103650), dtype=float64),
      'speech_reverberation_early': array(shape=(2, 6, 103650), dtype=float64),
      'speech_reverberation_tail': array(shape=(2, 6, 103650), dtype=float64),
      'noise_image': array(shape=(1, 1), dtype=float64),
      'observation': array(shape=(6, 103650), dtype=float64)},
     'snr': 29.749852569493584}
    """
    db = SmsWsj()
    ds = db.get_dataset(dataset)
    ds = ds.map(AudioReader())
    return ds


def get_scores(ex, prediction, source):
    """
    Calculate the scores, where the prediction/estimated signal is tested
    against the source/desired signal.
    This function is for oracle test to figure out, which metric can work with
    source signal.

    Example:
        SI-SDR does not work, when the desired signal is the signal befor the
        room impulse response and give strange results, when the channel is
        changed.

    >>> pprint(get_scores(get_dataset('cv_dev93')[0], 'image_0', 'early_0'))
    {'pesq': array([2.861]),
     'stoi': array([0.97151566]),
     'mir_eval_sxr_sdr': array([13.39136665]),
     'si_sdr': array([10.81039897])}
    >>> pprint(get_scores(get_dataset('cv_dev93')[0], 'image_0', 'source'))
    {'pesq': array([2.234]),
     'stoi': array([0.8005423]),
     'mir_eval_sxr_sdr': array([12.11446204]),
     'si_sdr': array([-20.05244551])}
    >>> pprint(get_scores(get_dataset('cv_dev93')[0], 'image_0', 'image_1'))
    {'pesq': array([3.608]),
     'stoi': array([0.92216845]),
     'mir_eval_sxr_sdr': array([9.55425598]),
     'si_sdr': array([-0.16858895])}
    """
    def get_signal(ex, name):
        assert isinstance(ex, dict), ex
        assert 'audio_data' in ex, ex
        assert isinstance(ex['audio_data'], dict), ex
        if name == 'source':
            return ex['audio_data']['speech_source'][:]
        elif name == 'early_0':
            return ex['audio_data']['speech_reverberation_early'][:, 0]
        elif name == 'early_1':
            return ex['audio_data']['speech_reverberation_early'][:, 1]
        elif name == 'image_0':
            return ex['audio_data']['speech_image'][:, 0]
        elif name == 'image_1':
            return ex['audio_data']['speech_image'][:, 1]
        elif name == 'image_0_noise':
            return ex['audio_data']['speech_image'][:, 0] + \
                   ex['audio_data']['noise_image'][0]
        elif name == 'image_1_noise':
            return ex['audio_data']['speech_image'][:, 1] + \
                   ex['audio_data']['noise_image'][0]
        else:
            raise ValueError(name)

    speech_prediction = get_signal(ex, prediction)
    speech_source = get_signal(ex, source)

    metric = OutputMetrics(
        speech_prediction=speech_prediction,
        speech_source=speech_source,
        sample_rate=8000,
        enable_si_sdr=True,
    )

    result = metric.as_dict()
    del result['mir_eval_sxr_selection']
    del result['mir_eval_sxr_sar']
    del result['mir_eval_sxr_sir']

    return result


@experiment.command
def summary(out):
    if dlp_mpi.IS_MASTER:
        if isinstance(out, str):
            assert out.endswith('.json'), out
            with open(out, 'r') as fd:
                data = json.load(fd)
        else:
            data = out

        df = pd.DataFrame(data)

        def force_order(df, key):
            # https://stackoverflow.com/a/28686885/5766934
            df[key] = df[key].astype("category")
            df[key].cat.set_categories(pd.unique(df['source']), inplace=True)

        force_order(df, 'source')
        force_order(df, 'prediction')

        with pd.option_context(
                "display.precision", 2,
                'display.width', 200,
        ):
            # print(pd.pivot_table(
            #     df,
            #     index=['prediction', 'source'],
            #     columns='score_name',
            #     values='value',
            #     aggfunc=np.mean,  # average over examples
            # ))
            print()
            print(pd.pivot_table(
                df.query('score_name == "mir_eval_sxr_sdr"'),
                index=['prediction'],
                columns=['score_name', 'source'],
                values='value',
                aggfunc=np.mean,  # average over examples
            ))
            print()
            print(pd.pivot_table(
                df.query('score_name == "si_sdr"'),
                index=['prediction'],
                columns=['score_name', 'source'],
                values='value',
                aggfunc=np.mean,  # average over examples
            ))


@experiment.automain
def main(_run, out):
    if dlp_mpi.IS_MASTER:
        from sacred.commands import print_config
        print_config(_run)

    ds = get_dataset()

    data = []

    for ex in dlp_mpi.split_managed(ds.sort(), allow_single_worker=True):
        for prediction in [
            'source',
            'early_0',
            'early_1',
            'image_0',
            'image_1',
            'image_0_noise',
            'image_1_noise',
        ]:
            for source in [
                'source',
                'early_0',
                'early_1',
                'image_0',
                'image_1',
                'image_0_noise',
                'image_1_noise',
            ]:
                scores = get_scores(ex, prediction=prediction, source=source)
                for score_name, score_value in scores.items():
                    data.append(dict(
                        score_name=score_name,
                        prediction=prediction,
                        source=source,
                        example_id=ex['example_id'],
                        value=score_value,
                    ))

    data = dlp_mpi.gather(data)

    if dlp_mpi.IS_MASTER:
        data = [
            entry
            for worker_data in data
            for entry in worker_data
        ]

        if out is not None:
            assert isinstance(out, str), out
            assert out.endswith('.json'), out
            print(f'Write details to {out}.')
            dump_json(data, out)

        summary(data)
