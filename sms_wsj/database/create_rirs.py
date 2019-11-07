"""Call instructions:

# When you do not have MPI:
python -m sms_wsj.database.create_rirs with database_path=/Users/lukas/Downloads/temp_wsj_bss debug=True
python -m sms_wsj.database.create_rirs with database_path=temp_wsj_bss debug=True

# When you have MPI:
mpiexec -np 3 python -m sms_wsj.database.create_rirs with database_path=temp_wsj_bss debug=True

"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile
from sacred import Experiment
from sms_wsj.reverb.reverb_utils import generate_rir
from sms_wsj.reverb.scenario import generate_random_source_positions
from sms_wsj.reverb.scenario import generate_sensor_positions
from sms_wsj.reverb.scenario import sample_from_random_box

import dlp_mpi

experiment = Experiment(Path(__file__).stem)


@experiment.config
def config():
    debug = False
    database_path = ""

    # Either set it to zero or above 0.15 s. Otherwise, RIR contains NaN.
    sound_decay_time_range = dict(low=0.2, high=0.5)

    geometry = dict(
        number_of_sources=4,
        number_of_sensors=6,
        sensor_shape="circular",
        center=[[4.], [3.], [1.5]],  # m
        scale=0.1,  # m
        room=[[8], [6], [3]],  # m
        random_box=[[0.4], [0.4], [0.4]],  # m
    )

    datasets = dict(
        train_si284=dict(
            count=33561,  # 33561 unique non-pp utterances
        ),
        cv_dev93=dict(
            count=491 * 2,  # 491 unique non-pp utterances
        ),
        test_eval92=dict(
            count=333 * 4,  # 333 unique non-pp utterances
        ),
    )

    sample_rate = 8000
    filter_length = 2 ** 13  # 1.024 seconds when sample_rate == 8000


def get_rng(dataset, example_id):
    string = f"{dataset}_{example_id}"
    seed = (
            int(hashlib.sha256(string.encode("utf-8")).hexdigest(),
                16) % 2 ** 32
    )
    return np.random.RandomState(seed=seed)


@experiment.command
def scenarios(
        database_path,
        datasets,
        geometry,
        sound_decay_time_range,
        debug,
):
    if not dlp_mpi.IS_MASTER:
        # It is enough, when one process generates the scenarios.json
        return

    assert len(database_path) > 0, "Database path can not be empty."
    database_path = Path(database_path).expanduser().resolve()
    scenario_json = Path(database_path) / "scenarios.json"

    print(f'from: random')
    print(f'to:   {database_path}')

    database = defaultdict(lambda: defaultdict(dict))
    for dataset, dataset_config in datasets.items():
        for example_id in range(dataset_config["count"]):
            if debug and example_id >= 2:
                break

            example_id = str(example_id)
            rng = get_rng(dataset, example_id)
            room_dimensions = sample_from_random_box(
                geometry["room"], geometry["random_box"], rng=rng
            )
            center = sample_from_random_box(
                geometry["center"], geometry["random_box"], rng=rng
            )
            source_positions = generate_random_source_positions(
                center=center,
                sources=geometry["number_of_sources"],
                rng=rng,
            )
            sensor_positions = generate_sensor_positions(
                shape=geometry["sensor_shape"],
                center=center,
                scale=geometry["scale"],
                number_of_sensors=geometry["number_of_sensors"],
                rotate_x=rng.uniform(0, 0.01 * 2 * np.pi),
                rotate_y=rng.uniform(0, 0.01 * 2 * np.pi),
                rotate_z=rng.uniform(0, 2 * np.pi),
            )
            sound_decay_time = rng.uniform(**sound_decay_time_range)
            database['datasets'][dataset][example_id] = {
                'room_dimensions': room_dimensions,
                'sound_decay_time': sound_decay_time,
                'source_position': source_positions,
                'sensor_position': sensor_positions,
            }
            # Use round to make it easier to read the numbers.
            # The rounding at this position is allowed, because all values are
            # independent of each other.
            # This introduces a small jitter on all positions.
            database['datasets'][dataset][example_id] = {
                k: np.round(v, decimals=3)
                for k, v in database['datasets'][dataset][example_id].items()
            }
            database['datasets'][dataset][example_id].update({
                k: v.tolist()
                for k, v in database['datasets'][dataset][example_id].items()
                if isinstance(v, np.ndarray)
            })
            database['datasets'][dataset][example_id]['example_id'] = \
                example_id

    scenario_json.parent.mkdir(exist_ok=False, parents=True)
    with scenario_json.open('w') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)


@experiment.command
def rirs(
        database_path,
        datasets,
        sample_rate,
        filter_length,
):
    database_path = Path(database_path)

    if dlp_mpi.IS_MASTER:
        scenario_json = database_path / "scenarios.json"
        with scenario_json.open() as f:
            database = json.load(f)
        for dataset in datasets:
            dataset_path = database_path / dataset
            dataset_path.mkdir(parents=True, exist_ok=True)
    else:
        database = None
    database = dlp_mpi.bcast(database)

    for dataset_name, dataset in database['datasets'].items():

        for _example_id, example in dlp_mpi.split_managed(
                list(sorted(dataset.items())),
                progress_bar=True,
                is_indexable=True,
        ):
            h = generate_rir(
                room_dimensions=example['room_dimensions'],
                source_positions=example['source_position'],
                sensor_positions=example['sensor_position'],
                sound_decay_time=example['sound_decay_time'],
                sample_rate=sample_rate,
                filter_length=filter_length,
                sensor_orientations=None,
                sensor_directivity=None,
                sound_velocity=343
            )
            assert not np.any(
                np.isnan(h)
            ), f"{np.sum(np.isnan(h))} values of {h.size} are NaN."

            K, D, T = h.shape
            directory = database_path / dataset_name / _example_id
            directory.mkdir(parents=False, exist_ok=False)

            for k in range(K):
                # Although storing as np.float64 does not allow every reader
                # to access the files, it does not require normalization and
                # we are unsure how much precision is needed for RIRs.
                with soundfile.SoundFile(
                        str(directory / f"h_{k}.wav"), subtype='DOUBLE',
                        samplerate=sample_rate, mode='w', channels=h.shape[1]
                ) as f:
                    f.write(h[k, :, :].T)

        dlp_mpi.barrier()

        print(f'RANK={dlp_mpi.RANK}, SIZE={dlp_mpi.SIZE}:'
              f' Finished {dataset_name}.')


@experiment.automain
def main():
    scenarios()
    rirs()
