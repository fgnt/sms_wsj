"""Call instructions:

# When you do not have MPI:
python -m nt.database.wsj_bss.create_files with database_path=/Users/lukas/Downloads/temp_wsj_bss debug=True
python -m nt.database.wsj_bss.create_files with database_path=temp_wsj_bss debug=True

# When you have MPI:
mpiexec -np 3 python -m nt.database.wsj_bss.create_files with database_path=temp_wsj_bss debug=True

# When on PC2:
# Multiply your needs with mpiprocs.
# 6h is barely enough with rset=200:mpiprocs=1:ncpus=1:mem=2g:vmem=2g
# and creating all datasets.
rm -rf /scratch/hpc-prf-nt2/ldrude/storage_root/project_dc/data/wsj_bss_v2
TARGET_PATH=/scratch/hpc-prf-nt2/ldrude/storage_root/project_dc/data/wsj_bss_v2
mkdir -p $TARGET_PATH
ccsalloc \
    --res="rset=200:mpiprocs=1:ncpus=1:mem=2g:vmem=2g" \
    --join --stdout=$TARGET_PATH/python_%A_%a.out \
    --tracefile=$TARGET_PATH/trace_%reqid.trace \
    -t 6h \
    -N create_files \
    cbj.ompi -- \
    python -m nt.database.wsj_bss.create_files with database_path=$TARGET_PATH

# Synchronize the files:
rsync \
    --recursive \
    --info=progress2 \
    --compress \
    pc2:/scratch/hpc-prf-nt2/ldrude/storage_root/project_dc/data/wsj_bss \
    /net/vol/ldrude/projects/2017/project_dc_storage/data/

time cbj.copy tar \
    pc2:/scratch/hpc-prf-nt2/ldrude/storage_root/project_dc/data/wsj_bss \
    /net/vol/ldrude/projects/2017/project_dc_storage/data/
"""
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from sacred import Experiment

from paderbox.io import dump_audio
from paderbox.io import dump_json
from paderbox.reverb.reverb_utils import generate_rir
from paderbox.reverb.scenario import generate_random_source_positions
from paderbox.reverb.scenario import generate_sensor_positions
from paderbox.reverb.scenario import sample_from_random_box
from paderbox.utils.mpi import COMM
from paderbox.utils.mpi import IS_MASTER
from paderbox.utils.mpi import RANK
from paderbox.utils.mpi import SIZE
from paderbox.utils.mpi import map_unordered
from paderbox.database.keys import *

experiment = Experiment(Path(__file__).stem)


@experiment.config
def config():
    debug = False
    database_path = ""

    # Either set it to zero or above 0.15 s. Otherwise, RIR contains NaN.
    sound_decay_time_range = dict(low=0.2, high=0.5)

    geometry = dict(
        number_of_sources=2,
        number_of_sensors=6,
        sensor_shape="circular",
        center=[[4.], [3.], [1.5]],  # m
        scale=0.1,  # m
        room=[[8], [6], [3]],  # m
        random_box=[[0.4], [0.4], [0.4]],  # m
    )

    datasets = dict(
        train_si284=dict(
            count=30000,  # 33561 unique non-pp utterances
            minimum_angular_distance=15,  # degree
            maximum_angular_distance=None,  # degree
        ),
        cv_dev93=dict(
            count=500,  # 491 unique non-pp utterances
            minimum_angular_distance=15,  # degree
            maximum_angular_distance=None,  # degree
        ),
        test_eval92=dict(
            count=1500,  # 333 unique non-pp utterances
            minimum_angular_distance=15,  # degree
            maximum_angular_distance=None,  # degree
        ),
        # test_eval92_narrow=dict(
        #     count=5 * 333,
        #     minimum_angular_distance=None,  # degree
        #     maximum_angular_distance=45,  # degree
        # ),
        # test_eval92_wide=dict(
        #     count=5 * 333,
        #     minimum_angular_distance=90,  # degree
        #     maximum_angular_distance=None,  # degree
        # ),
    )

    sample_rate = 8000
    filter_length = 2 ** 13


def get_rng(dataset, example_id):
    string = f"{dataset}_{example_id}"
    seed = (
        int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 2 ** 32
    )
    return np.random.RandomState(seed=seed)


@experiment.automain
def main(
    database_path,
    datasets,
    geometry,
    sound_decay_time_range,
    sample_rate,
    filter_length,
    debug,
):
    assert len(database_path) > 0, "Database path can not be empty."
    database_path = Path(database_path)

    if IS_MASTER:
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
            source_positions_center = sample_from_random_box(
                geometry["center"], geometry["random_box"], rng=rng
            )
            source_positions = generate_random_source_positions(
                center=source_positions_center,
                sources=geometry["number_of_sources"],
                rng=rng,
            )
            sensor_positions_center = sample_from_random_box(
                geometry["center"], geometry["random_box"], rng=rng
            )
            sensor_positions = generate_sensor_positions(
                shape=geometry["sensor_shape"],
                center=sensor_positions_center,
                scale=geometry["scale"],
                number_of_sensors=geometry["number_of_sensors"],
                rotate_x=rng.uniform(0, 0.01 * 2 * np.pi),
                rotate_y=rng.uniform(0, 0.01 * 2 * np.pi),
                rotate_z=rng.uniform(0, 2 * np.pi),
            )
            sound_decay_time = rng.uniform(**sound_decay_time_range)
            database[DATASETS][dataset][example_id] = {
                ROOM_DIMENSIONS: room_dimensions,
                SOUND_DECAY_TIME: sound_decay_time,
                SOURCE_POSITION: source_positions,
                SENSOR_POSITION: sensor_positions,
            }
            database[DATASETS][dataset][example_id] = {
                k: np.round(v, decimals=3)
                for k, v in database[DATASETS][dataset][example_id].items()
            }
            database[DATASETS][dataset][example_id][EXAMPLE_ID] = example_id
    if IS_MASTER:
        dump_json(database, database_path / "scenarios.json")

    if IS_MASTER:
        for dataset in datasets:
            dataset_path = database_path / dataset
            dataset_path.mkdir(parents=True, exist_ok=True)

    # TODO: Add either broadcasting or synchronize a checksum for savety.

    # Non-masters may need the folders before the master created them.
    COMM.Barrier()

    for dataset_name, dataset in database[DATASETS].items():
        print(f'RANK={RANK}, SIZE={SIZE}: Starting {dataset_name}.')

        def workload(_example_id):
            example = dataset[_example_id]
            h = generate_rir(
                room_dimensions=example[ROOM_DIMENSIONS],
                source_positions=example[SOURCE_POSITION],
                sensor_positions=example[SENSOR_POSITION],
                sound_decay_time=example[SOUND_DECAY_TIME],
                sample_rate=sample_rate,
                filter_length=filter_length,
                sensor_orientations=None,
                sensor_directivity=None,
                sound_velocity=343,
                algorithm="habets",
            )
            assert not np.any(
                np.isnan(h)
            ), f"{np.sum(np.isnan(h))} values of {h.size} are NaN."

            K, D, T = h.shape
            directory = database_path / dataset_name / _example_id
            directory.mkdir(parents=False, exist_ok=False)

            for k in range(K):
                # Although storing as np.float64 does not allow every reader
                # to access the files, it doe not require normalization and
                # we are unsure how much precision is needed for RIRs.
                dump_audio(
                    h[k, :, :],
                    directory / f"h_{k}.wav",
                    sample_rate=sample_rate,
                    dtype=None,
                    normalize=False,
                )

        for _ in map_unordered(workload, dataset, progress_bar=False):
            pass

        # for example_id in list(dataset.keys())[RANK::SIZE]:
        #     workload(example_id)

        # for example_id in share_round_robin(dataset):
        #     workload(example_id)

        print(f'RANK={RANK}, SIZE={SIZE}: Finished {dataset_name}.')
