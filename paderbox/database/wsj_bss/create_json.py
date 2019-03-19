"""
NT:
python -m paderbox.database.wsj_bss.create_json
scp wsj_bss.json jenkins@ntjenkins:/net/vol/jenkins/jsons/

PC2:
python -m paderbox.database.wsj_bss.create_json \
    --json-path=$NT_DATABASE_JSONS_DIR/wsj_bss.json
"""
from paderbox.database.wsj import PUNCTUATION_SYMBOLS
from collections import defaultdict
from pathlib import Path
import numpy as np
import click
from copy import copy
from paderbox.database.helper import (
    dump_database_as_json,
    click_common_options,
    check_audio_files_exist
)
from paderbox.database.keys import *
from paderbox.database.wsj import WSJ_8kHz
from paderbox.io.data_dir import wsj_bss
from paderbox.database import JsonDatabase


@click.command()
@click_common_options("wsj_bss.json", wsj_bss)
def main(json_path, database_path):
    json_path = Path(json_path)
    if json_path.is_file():
        raise FileExistsError(json_path)

    setup = dict(
        train_si284=dict(source_dataset_name="train_si284"),
        cv_dev93=dict(source_dataset_name="cv_dev93"),
        test_eval92=dict(source_dataset_name="test_eval92"),
    )

    rir_db = JsonDatabase(database_path / "scenarios.json")
    source_db = WSJ_8kHz()
    target_db = dict()
    target_db[DATASETS] = defaultdict(dict)

    rng = np.random.RandomState(0)
    for dataset_name in setup.keys():
        source_dataset_name = setup[dataset_name]["source_dataset_name"]
        source_iterator = source_db.get_iterator_by_names(source_dataset_name)
        print(f'length of source {dataset_name}: {len(source_iterator)}')
        source_iterator = source_iterator.filter(
            filter_fn=filter_punctuation_pronunciation, lazy=False
        )
        print(
            f'length of source {dataset_name}: {len(source_iterator)} '
            '(after punctuation filter)'
        )

        rir_iterator = rir_db.get_iterator_by_names(dataset_name)
        print(f'length of rir {dataset_name}: {len(rir_iterator)}')

        for rir_example in rir_iterator:
            example = None
            while example is None:
                example = get_randomized_example(
                    rir_example,
                    source_iterator,
                    rng,
                    dataset_name,
                    database_path,
                )
            target_db[DATASETS][dataset_name][example[EXAMPLE_ID]] = example

    print("Check that all wav files in the json exist.")
    check_audio_files_exist(target_db, speedup="thread")
    print("Finished check.")
    dump_database_as_json(json_path, target_db)


def filter_punctuation_pronunciation(example):
    transcription = example[KALDI_TRANSCRIPTION].split()
    return len(PUNCTUATION_SYMBOLS.intersection(transcription)) == 0


def get_randomized_example(
    rir_example, source_iterator, rng, dataset_name, database_path
):
    """Returns None, if example is rejected."""
    example = copy(rir_example)
    example[NUM_SPEAKERS] = len(example[SOURCE_POSITION][0])

    # Fixed selection of the first, to at least see each utterance once.
    # If we use more examples, utterances will be taken again.
    rir_id = rir_example[EXAMPLE_ID]
    source_examples = [source_iterator[int(rir_id) % len(source_iterator)]]
    for _ in range(1, example[NUM_SPEAKERS]):
        source_examples.append(
            source_iterator[rng.randint(0, len(source_iterator))]
        )

    example[SPEAKER_ID] = [ex[SPEAKER_ID] for ex in source_examples]
    if len(set(example[SPEAKER_ID])) < example[NUM_SPEAKERS]:
        return  # Rejection sampling

    example["source_id"] = [ex[EXAMPLE_ID] for ex in source_examples]

    for k in (GENDER, KALDI_TRANSCRIPTION):
        example[k] = [ex[k] for ex in source_examples]

    example[LOG_WEIGHTS] = rng.uniform(0, 5, size=(example[NUM_SPEAKERS],))
    example[LOG_WEIGHTS] -= np.mean(example[LOG_WEIGHTS])
    example[LOG_WEIGHTS] = example[LOG_WEIGHTS].tolist()

    # This way, at least the first speaker can have proper alignments, all other
    # speakers can not be used for ASR.
    example[NUM_SAMPLES] = {
        OBSERVATION: source_examples[0][NUM_SAMPLES],
        SPEECH_SOURCE: [ex[NUM_SAMPLES] for ex in source_examples]
    }
    example["offset"] = [0]
    for k in range(1, example[NUM_SPEAKERS]):
        excess_samples = (
            example[NUM_SAMPLES][OBSERVATION]
            - example[NUM_SAMPLES][SPEECH_SOURCE][k]
        )
        example["offset"].append(
            np.sign(excess_samples) * rng.randint(0, np.abs(excess_samples))
        )

    example[EXAMPLE_ID] = "_".join((*example["source_id"], rir_id))

    example[AUDIO_PATH] = dict()
    example[AUDIO_PATH][SPEECH_SOURCE] = [
        ex[AUDIO_PATH][OBSERVATION] for ex in source_examples
    ]
    example[AUDIO_PATH][RIR] = [
        database_path / dataset_name / rir_id / f"h_{k}.wav"
        for k in range(len(example[SOURCE_POSITION][0]))
    ]

    return example


if __name__ == "__main__":
    main()
