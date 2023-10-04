import os
import json
import functools
from collections import defaultdict
from copy import copy
from pathlib import Path
import warnings

import numpy as np
import sacred
import soundfile
from lazy_dataset.database import JsonDatabase

from sms_wsj.database.create_rirs import get_rng

ex = sacred.Experiment('Create intermediate SMS-WSJ json')

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


DEBUG_EXAMPLE_LIMIT = 10


def filter_punctuation_pronunciation(example):
    transcription = example['kaldi_transcription'].split()
    return len(PUNCTUATION_SYMBOLS.intersection(transcription)) == 0


def test_example_composition(a, b, speaker_ids):
    """

    Args:
        a: List of permutation example indices
        b: List of permutation example indices
        speaker_ids: Speaker id corresponding to an index

    Returns:

    >>> speaker_ids = np.array(['Alice', 'Bob', 'Carol', 'Carol'])
    >>> test_example_composition([0, 1, 2, 3], [2, 3, 1, 0], speaker_ids)
    >>> test_example_composition([0, 1, 2, 3], [0, 1, 2, 3], speaker_ids)
    Traceback (most recent call last):
    ...
    AssertionError: ('utterance duplicate', [0, 1, 2, 3], [0, 1, 2, 3])
    >>> test_example_composition([0, 1, 2, 3], [1, 0, 3, 2], speaker_ids)
    Traceback (most recent call last):
    ...
    AssertionError: ('speaker duplicate', 2)
    >>> test_example_composition([0, 1, 2, 3], [2, 3, 0, 1], speaker_ids)
    Traceback (most recent call last):
    ...
    AssertionError: ('duplicate pair', 2)



    """
    # Ensure that a speaker is not mixed with itself
    # This also ensures that an utterance is not mixed with itself
    assert np.all(speaker_ids[a] != speaker_ids[b]), ('speaker duplicate', len(a) - np.sum(speaker_ids[a] != speaker_ids[b]))

    # Ensure that any pair of utterances does not appear more than once
    tmp = [tuple(sorted(ab)) for ab in zip(a, b)]
    assert len(set(tuple(tmp))) == len(a), ('duplicate pair', len(a) - len(set(tuple(tmp))))


def extend_composition_example_greedy(rng, speaker_ids, example_compositions=None, tries=500):
    """

    Args:
        rng:
        speaker_ids: Speaker id corresponding to an index
        example_compositions:
        tries:

    Returns:

    >>> rng = np.random.RandomState(0)
    >>> speaker_ids = np.array(['Alice', 'Bob', 'Carol', 'Dave', 'Eve'])
    >>> comp = extend_composition_example_greedy(rng, speaker_ids)
    >>> comp
    array([[2],
           [0],
           [1],
           [3],
           [4]])
    >>> comp = extend_composition_example_greedy(rng, speaker_ids, comp)
    >>> comp
    array([[2, 3],
           [0, 4],
           [1, 2],
           [3, 0],
           [4, 1]])
    >>> comp = extend_composition_example_greedy(rng, speaker_ids, comp)
    >>> comp
    array([[2, 3, 1],
           [0, 4, 2],
           [1, 2, 3],
           [3, 0, 4],
           [4, 1, 0]])
    >>> speaker_ids[comp]
    array([['Carol', 'Dave', 'Bob'],
           ['Alice', 'Eve', 'Carol'],
           ['Bob', 'Carol', 'Dave'],
           ['Dave', 'Alice', 'Eve'],
           ['Eve', 'Bob', 'Alice']], dtype='<U5')
    """
    if example_compositions is None:
        example_compositions = np.arange(len(speaker_ids), dtype=int)
        example_compositions = rng.permutation(example_compositions)[:, None]
        return example_compositions

    assert example_compositions.ndim == 2, example_compositions.shape

    candidates = np.arange(len(speaker_ids), dtype=int)
    speaker_ids = np.array(speaker_ids)
    for _ in range(tries):
        candidates = rng.permutation(candidates)

        try:
            for i in range(len(candidates)):
                for _ in range(tries):
                    if any([
                        speaker_ids[entry_a] == speaker_ids[candidates[i]]
                        for entry_a in example_compositions[i]
                    ]):
                        candidates[i:] = rng.permutation(candidates[i:])
                    else:
                        break

            for tmp in example_compositions.T:
                test_example_composition(tmp, candidates, speaker_ids)

        except AssertionError:
            pass
        else:
            break
    else:
        raise RuntimeError(f'Couldn\'t find a valid speaker composition')

    return np.concatenate([example_compositions, candidates[:, None]], axis=-1)


def get_randomized_example(
    rir_example, source_examples, rng, dataset_name
):
    example_id = "_".join([
        rir_example['example_id'],
        *[source_ex["example_id"] for source_ex in source_examples],
    ])
    rng = get_rng(dataset_name, example_id)

    example = copy(rir_example)
    example['example_id'] = example_id
    example['dataset'] = dataset_name

    assert len(source_examples) <= len(example['source_position'][0])
    num_speakers = len(source_examples)

    # Remove unused source positions and rirs (i.e. the scenarios.json was
    # maybe generated with more speakers)
    example['source_position'] = [
        v[:num_speakers] for v in example['source_position']]
    example['audio_path']['rir'] = example['audio_path']['rir'][:num_speakers]

    example['num_speakers'] = num_speakers

    example['speaker_id'] = [exa['speaker_id'] for exa in source_examples]

    # asserts that no speaker_id is used twice
    assert len(set(example['speaker_id'])) == example['num_speakers']

    example["source_id"] = [exa['example_id'] for exa in source_examples]

    for k in ('gender', 'kaldi_transcription'):
        example[k] = [exa[k] for exa in source_examples]

    example['log_weights'] = rng.uniform(0, 5, size=(example['num_speakers'],))
    example['log_weights'] -= np.mean(example['log_weights'])
    example['log_weights'] = example['log_weights'].tolist()

    # This way, at least the first speaker can have proper alignments,
    # all other speakers can not be used for ASR.
    def _get_num_samples(num_samples):
        if isinstance(num_samples, dict):
            return num_samples['observation']
        else:
            return num_samples

    example['num_samples'] = dict()
    example['num_samples']['original_source'] = [
        _get_num_samples(exa['num_samples'])
        for exa in source_examples
    ]
    example['num_samples']['observation'] = max(
        example['num_samples']['original_source']
    )

    example["offset"] = []
    for k in range(example['num_speakers']):
        excess_samples = (
            example['num_samples']['observation']
            - example['num_samples']['original_source'][k]
        )
        assert excess_samples >= 0, excess_samples
        example["offset"].append(rng.randint(0, excess_samples + 1))

    example['audio_path']['original_source'] = [
        exa['audio_path']['observation'] for exa in source_examples
    ]
    # example['audio_path']['rir']: Already defined in rir_example.
    return example


def combine_rirs_and_sources(
        rir_dataset,
        source_dataset,
        num_speakers,
        dataset_name,
):
    # The keys of rir_dataset are integers. Sort the rirs based on this
    # integer.
    rir_dataset = rir_dataset.sort(sort_fn=functools.partial(sorted, key=int))

    assert len(rir_dataset) % len(source_dataset) == 0, (len(rir_dataset), len(source_dataset))
    repetitions = len(rir_dataset) // len(source_dataset)

    source_dataset = source_dataset.sort()
    source_dataset = list(source_dataset.tile(repetitions))

    speaker_ids = [example['speaker_id'] for example in source_dataset]

    rng = get_rng(dataset_name, 'example_compositions')

    composition_examples = None
    for _ in range(num_speakers):
        composition_examples = extend_composition_example_greedy(
            rng, speaker_ids, example_compositions=composition_examples,
        )

    ex_dict = dict()
    assert len(rir_dataset) == len(composition_examples), (len(rir_dataset), len(composition_examples))
    for rir_example, composition_example in zip(
            rir_dataset, composition_examples
    ):
        source_examples = [source_dataset[i] for i in composition_example]

        example = get_randomized_example(
            rir_example,
            source_examples,
            rng,
            dataset_name,
        )
        ex_dict[example['example_id']] = example

    return ex_dict


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

    num_speakers = 2
    debug = False  # If `True`, only creates a few examples per dataset.


@ex.automain
def main(
        json_path: Path,
        rir_dir: Path,
        wsj_json_path: Path,
        num_speakers: int,
        debug: bool,
):
    wsj_json_path = Path(wsj_json_path).expanduser().resolve()
    json_path = Path(json_path).expanduser().resolve()
    rir_dir = Path(rir_dir).expanduser().resolve()
    assert wsj_json_path.is_file(), json_path
    assert rir_dir.exists(), rir_dir

    # ToDo: What was the motivation for defining this "setup"?
    setup = dict(
        train_si284=dict(source_dataset_name="train_si284"),
        cv_dev93=dict(source_dataset_name="cv_dev93"),
        test_eval92=dict(source_dataset_name="test_eval92"),
    )

    rir_db = JsonDatabase(rir_dir / "scenarios.json")

    source_db = JsonDatabase(wsj_json_path)

    target_db = dict()
    target_db['datasets'] = defaultdict(dict)

    for dataset_name in setup.keys():
        source_dataset_name = setup[dataset_name]["source_dataset_name"]
        source_dataset = source_db.get_dataset(source_dataset_name)
        print(f'length of source {dataset_name}: {len(source_dataset)}')
        source_dataset = source_dataset.filter(
            filter_fn=filter_punctuation_pronunciation, lazy=False
        )
        print(
            f'length of source {dataset_name}: {len(source_dataset)} '
            '(after punctuation filter)'
        )

        def add_rir_path(rir_ex):
            assert 'audio_path' not in rir_ex, rir_ex
            example_id = rir_ex['example_id']
            rir_ex['audio_path'] = {'rir': [
                str(rir_dir / dataset_name / example_id / f"h_{k}.wav")
                for k in range(num_speakers)
            ]}
            return rir_ex

        rir_dataset = rir_db.get_dataset(dataset_name).map(add_rir_path)

        assert len(rir_dataset) % len(source_dataset) == 0, (
            f'To avoid a bias towards certain utterance the len '
            f'rir_dataset ({len(rir_dataset)}) should be an integer '
            f'multiple of len source_dataset ({len(source_dataset)}).'
        )

        print(f'length of rir {dataset_name}: {len(rir_dataset)}')

        probe_path = rir_dir / dataset_name / "0"
        available_speaker_positions = len(list(probe_path.glob('h_*.wav')))
        assert num_speakers <= available_speaker_positions, (
            f'Requested {num_speakers} num_speakers, while found only '
            f'{available_speaker_positions} rirs in {probe_path}.'
        )

        info = soundfile.info(str(rir_dir / dataset_name / "0" / "h_0.wav"))
        sample_rate_rir = info.samplerate

        ex_wsj = source_dataset.random_choice(1)[0]
        info = soundfile.SoundFile(ex_wsj['audio_path']['observation'])
        sample_rate_wsj = info.samplerate
        assert sample_rate_rir == sample_rate_wsj, (
            sample_rate_rir, sample_rate_wsj
        )

        if debug:
            rir_dataset = rir_dataset[:DEBUG_EXAMPLE_LIMIT]
            # Use step_size to avoid that only one speaker is in
            # source_iterator.
            step_size = len(source_dataset) // DEBUG_EXAMPLE_LIMIT
            source_dataset = source_dataset[::step_size]

        ex_dict = combine_rirs_and_sources(
            rir_dataset=rir_dataset,
            source_dataset=source_dataset,
            num_speakers=num_speakers,
            dataset_name=dataset_name,
        )

        target_db['datasets'][dataset_name] = ex_dict

    json_path.parent.mkdir(exist_ok=True, parents=True)
    with json_path.open('w') as f:
        json.dump(target_db, f, indent=2, ensure_ascii=False)
    print(f'{json_path} written.')
