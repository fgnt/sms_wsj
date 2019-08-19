from pathlib import Path
git_root = Path(__file__).parent.parent.resolve().expanduser()
from .kaldi import get_kaldi_wer