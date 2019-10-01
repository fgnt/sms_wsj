import os
import json
from collections import defaultdict
from copy import copy
from pathlib import Path

import numpy as np
import sacred
import soundfile
from lazy_dataset.database import JsonDatabase

ex = sacred.Experiment('Create sms_wsj json')

PUNCTUATION_SYMBOLS = set('''
    &AMPERSAND
    ,COMMA
    ;SEMI-COLON
    :COLON
    !EXCLAMATION-POINT
    ...ELLIPSIS
    -HYPHEN
    .PERIOD
    .DOT
    ?QUESTION-MARK

    .DECIMAL
    .PERCENT
    /SLASH

    'SINGLE-QUOTE
    "DOUBLE-QUOTE
    "QUOTE
    "UNQUOTE
    "END-OF-QUOTE
    "END-QUOTE
    "CLOSE-QUOTE
    "IN-QUOTES

    (PAREN
    (PARENTHESES
    (IN-PARENTHESIS
    (BRACE
    (LEFT-PAREN
    (PARENTHETICALLY
    (BEGIN-PARENS
    )CLOSE-PAREN
    )CLOSE_PAREN
    )END-THE-PAREN
    )END-OF-PAREN
    )END-PARENS
    )CLOSE-BRACE
    )RIGHT-PAREN
    )UN-PARENTHESES
    )PAREN

    {LEFT-BRACE
    }RIGHT-BRACE
'''.split())


def filter_punctuation_pronunciation(example):
    transcription = example['kaldi_transcription'].split()
    return len(PUNCTUATION_SYMBOLS.intersection(transcription)) == 0


def get_randomized_example(
    rir_example, source_iterator, rng, dataset_name, database_path
):
    """Returns None, if example is rejected."""
    example = copy(rir_example)
    example['num_speakers'] = len(example['source_position'][0])

    # Fixed selection of the first, to at least see each utterance once.
    # If we use more examples, utterances will be taken again.
    rir_id = rir_example['example_id']
    source_examples = [source_iterator[int(rir_id) % len(source_iterator)]]
    for _ in range(1, example['num_speakers']):
        source_examples.append(
            source_iterator[rng.randint(0, len(source_iterator))]
        )

    example['speaker_id'] = [exa['speaker_id'] for exa in source_examples]
    if len(set(example['speaker_id'])) < example['num_speakers']:
        return  # asserts that no speaker_id is used twice

    example["source_id"] = [exa['example_id'] for exa in source_examples]

    for k in ('gender', 'kaldi_transcription'):
        example[k] = [exa[k] for exa in source_examples]

    example['log_weights'] = rng.uniform(0, 5, size=(example['num_speakers'],))
    example['log_weights'] -= np.mean(example['log_weights'])
    example['log_weights'] = example['log_weights'].tolist()

    # This way, at least the first speaker can have proper alignments,
    # all other speakers can not be used for ASR.
    if isinstance(source_examples[0]['num_samples'], dict) \
            and 'observation' in source_examples[0]['num_samples']:
        # 16k case
        example['num_samples'] = {
            'observation': source_examples[0]['num_samples']['observation'],
            'speech_source': [exa['num_samples']['observation'] for exa in
                              source_examples]
        }
    else:
        # 8k case
        example['num_samples'] = {
            'observation': source_examples[0]['num_samples'],
            'speech_source': [exa['num_samples'] for exa in source_examples]
        }
    example["offset"] = [0]
    for k in range(1, example['num_speakers']):
        excess_samples = (
            example['num_samples']['observation']
            - example['num_samples']['speech_source'][k]
        )
        example["offset"].append(int(
            np.sign(excess_samples) * rng.randint(0, np.abs(excess_samples))
        ))
    example['example_id'] = "_".join((*example["source_id"], rir_id))

    example['audio_path'] = dict()
    example['audio_path']['speech_source'] = [
        exa['audio_path']['observation'] for exa in source_examples
    ]
    example['audio_path']['rir'] = [
        str(database_path / dataset_name / rir_id / f"h_{k}.wav")
        for k in range(len(example['source_position'][0]))
    ]

    return example


@ex.config
def config():
    rir_dir = None
    json_path = None
    wsj_json_path = None
    if rir_dir is None and 'RIR_DIR' in os.environ:
        rir_dir = os.environ['RIR_DIR']
    assert rir_dir is not None, 'You have to specify the rir dir'
    if wsj_json_path is None and 'WSJ_JSON' in os.environ:
        wsj_json_path = os.environ['WSJ_JSON']
    assert wsj_json_path is not None, 'You have to specify a wsj_json_path'
    if json_path is None and 'SMS_WSJ_JSON' in os.environ:
        json_path = os.environ['SMS_WSJ_JSON']
    assert json_path is not None, 'You have to specify a path for the new json'


@ex.automain
def main(json_path: Path, rir_dir: Path, wsj_json_path: Path):
    wsj_json_path = Path(wsj_json_path).expanduser().resolve()
    json_path = Path(json_path).expanduser().resolve()
    if json_path.exists():
        raise FileExistsError(json_path)
    rir_dir = Path(rir_dir).expanduser().resolve()
    assert wsj_json_path.is_file(), json_path
    assert rir_dir.exists(), rir_dir

    setup = dict(
        train_si284=dict(source_dataset_name="train_si284"),
        cv_dev93=dict(source_dataset_name="cv_dev93"),
        test_eval92=dict(source_dataset_name="test_eval92"),
    )

    rir_db = JsonDatabase(rir_dir / "scenarios.json")

    source_db = JsonDatabase(wsj_json_path)

    target_db = dict()
    target_db['datasets'] = defaultdict(dict)

    rng = np.random.RandomState(0)
    for dataset_name in setup.keys():
        source_dataset_name = setup[dataset_name]["source_dataset_name"]
        source_iterator = source_db.get_dataset(source_dataset_name)
        print(f'length of source {dataset_name}: {len(source_iterator)}')
        source_iterator = source_iterator.filter(
            filter_fn=filter_punctuation_pronunciation, lazy=False
        )
        print(
            f'length of source {dataset_name}: {len(source_iterator)} '
            '(after punctuation filter)'
        )

        rir_iterator = rir_db.get_dataset(dataset_name)
        print(f'length of rir {dataset_name}: {len(rir_iterator)}')

        info = soundfile.info(str(rir_dir / dataset_name / "0" / "h_0.wav"))
        frame_rate_rir = info.samplerate

        ex_wsj = source_iterator.random_choice(1, rng_state=rng)[0]
        info = soundfile.SoundFile(ex_wsj['audio_path']['observation'])
        frame_rate_wsj = info.samplerate
        assert frame_rate_rir == frame_rate_wsj, (
            frame_rate_rir, frame_rate_wsj)

        for rir_example in rir_iterator:
            example = None
            while example is None:
                example = get_randomized_example(
                    rir_example,
                    source_iterator,
                    rng,
                    dataset_name,
                    rir_dir,
                )
            ex_id = example['example_id']
            del example['example_id']
            target_db['datasets'][dataset_name][ex_id] = example
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with json_path.open('w') as f:
        json.dump(target_db, f, indent=4, ensure_ascii=False)
    print(f'{json_path} written')
