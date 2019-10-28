import json
from pathlib import Path
import numpy as np
import soundfile


class NumpyEncoder(json.JSONEncoder):
    # https://stackoverflow.com/a/47626762/5766934
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dump_json(obj, file, indent=2):
    with open(file, 'w') as fd:
        json.dump(obj, fd, cls=NumpyEncoder, indent=indent)


def dump_audio(obj, file, samplerate=8000, mkdir=True, normalize=True):
    if normalize:
        # Correction, because the allowed values are in the range [-1, 1).
        # => "1" is not a vaild value
        correction = (2**15 - 1) / (2**15)
        obj = obj * (correction / np.amax(np.abs(obj)))

    if isinstance(file, Path):
        file = str(file)
    try:
        soundfile.write(
            file=file,
            data=obj,
            samplerate=samplerate,
        )
    except RuntimeError:
        if mkdir:
            # Assume mkdir is rarely nessesary, hence first try write
            Path(file).parent.mkdir(
                parents=True,
                exist_ok=True,  # Allow concurrent mkdir
            )
            soundfile.write(
                file=file,
                data=obj,
                samplerate=samplerate,
            )
        else:
            raise
