from collections import defaultdict
from pathlib import Path
import numpy as np
from copy import copy
import sacred
import json
import wave
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


@ex.config
def config():
    rir_dir = None
    json_path = None
    wsj_json = None
    assert rir_dir  is not None, 'You have to specify the rir dir'
    assert wsj_json is not None, 'You have to specify a path to the wsj.json'
    assert json_path is not None, 'You have to specify the path to write the json to'
    wsj_json = Path(wsj_json).expanduser().resolve()
    json_path = Path(json_path).expanduser().resolve()
    if json_path.exists():
        raise FileExistsError(json_path)
    rir_dir  = Path(rir_dir).expanduser().resolve()
    assert wsj_json.is_file(), json_path
    assert rir_dir.exists(), rir_dir 


@ex.automain
def main(json_path: Path, rir_dir: Path, wsj_json: Path):
    

    setup = dict(
        train_si284=dict(source_dataset_name="train_si284"),
        cv_dev93=dict(source_dataset_name="cv_dev93"),
        test_eval92=dict(source_dataset_name="test_eval92"),
    )

    rir_db = JsonDatabase(rir_dir / "scenarios.json")

    source_db = JsonDatabase(wsj_json)

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

        with wave.open(rir_dir / dataset_name / "0" / "h_0.wav", "rb") as f:
            frame_rate_rir = f.getframerate()

        ex_wsj = source_iterator.random_choice(1, rng_state=rng)[0]
        with wave.open(ex_wsj['audio_path']['observation'], "rb") as f:
            frame_rate_wsj = f.getframerate()
        assert frame_rate_rir == frame_rate_wsj, (frame_rate_rir, frame_rate_wsj)


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
            target_db['datasets'][dataset_name][example['example_id']] = example

    print("Finished check.")
    json.dump(target_db, json_path, create_path=True,
             indent=4, ensure_ascii=False)


def filter_punctuation_pronunciation(example):
    transcription = example['kaldi_transcription'].split()
    return len(PUNCTUATION_SYMBOLS.intersection(transcription)) == 0


def get_randomized_example(
    rir_example, source_iterator, rng, dataset_name, database_path
):
    """Returns None, if example is rejected."""
    example = copy(rir_example)
    example['num_speaker'] = len(example['source_position'][0])

    # Fixed selection of the first, to at least see each utterance once.
    # If we use more examples, utterances will be taken again.
    rir_id = rir_example['example_id']
    source_examples = [source_iterator[int(rir_id) % len(source_iterator)]]
    for _ in range(1, example['num_speaker']):
        source_examples.append(
            source_iterator[rng.randint(0, len(source_iterator))]
        )

    example['speaker_id'] = [ex['speaker_id'] for ex in source_examples]
    if len(set(example['speaker_id'])) < example['num_speaker']:
        return  # asserts that no speaker_id is used twice

    example["source_id"] = [ex['example_id'] for ex in source_examples]

    for k in ('gender', 'kaldi_transcription'):
        example[k] = [ex[k] for ex in source_examples]

    example['log_weights'] = rng.uniform(0, 5, size=(example['num_speaker'],))
    example['log_weights'] -= np.mean(example['log_weights'])
    example['log_weights'] = example['log_weights'].tolist()

    # This way, at least the first speaker can have proper alignments, all other
    # speakers can not be used for ASR.
    if isinstance(source_examples[0]['num_samples'], dict) \
            and 'observation' in source_examples[0]['num_samples']:
        # 16k case
        example['num_samples'] = {
            'observation': source_examples[0]['num_samples']['observation'],
            'speech_source': [ex['num_samples']['observation'] for ex in
                            source_examples]
        }
    else:
        # 8k case
        example['num_samples'] = {
            'observation': source_examples[0]['num_samples'],
            'speech_source': [ex['num_samples'] for ex in source_examples]
        }
    example["offset"] = [0]
    for k in range(1, example['num_speakers']):
        excess_samples = (
            example['num_samples']['observation']
            - example['num_samples']['speech_source'][k]
        )
        example["offset"].append(
            np.sign(excess_samples) * rng.randint(0, np.abs(excess_samples))
        )

    example['example_id'] = "_".join((*example["source_id"], rir_id))

    example['audio_path'] = dict()
    example['audio_path']['speech_source'] = [
        ex['audio_path']['observation'] for ex in source_examples
    ]
    example['audio_path']['rir'] = [
        database_path / dataset_name / rir_id / f"h_{k}.wav"
        for k in range(len(example['source_position'][0]))
    ]

    return example
