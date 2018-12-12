from pathlib import Path
import re
import click
import time
import logging

from paderbox.io.data_dir import wsj
from paderbox.io.audioread import read_nist_wsj
from paderbox.io.audiowrite import audiowrite


def write_wavs(dst_dir, wsj_root: Path):

    dst_dir = Path(dst_dir)

    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=False)

    wsj_nist_files = list(wsj_root.rglob("*.wv1"))  # ignore .wv2 files since
    # they are not referenced in our database anyway

    logging.info(f"Writing {len(wsj_nist_files)} wav files...")

    for nist_file in wsj_nist_files:
        signal = read_nist_wsj(nist_file, expected_sample_rate=16000)[0]
        subdir = Path(re.search(
            f'{wsj}/(.+?).wv1', str(nist_file)).group(1)
                      ).parent
        wav_dir = dst_dir / subdir
        if not wav_dir.exists():
            wav_dir.mkdir(parents=True, exist_ok=False)
        wav_path = wav_dir / f'{nist_file.stem}.wav'
        audiowrite(signal, wav_path, sample_rate=16000)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    @click.command()
    @click.option('-d', '--dst-dir',
                  help="Directory which will store the converted WSJ wav files",
                  type=click.Path())
    @click.option('--wsj-root', default=wsj,
                  help='Root directory to WSJ database', type=click.Path())
    def main(dst_dir, wsj_root):
        logging.info(f"Start - {time.ctime()}")
        write_wavs(dst_dir, wsj_root)
        logging.info(f"Done - {time.ctime()}")

    main()
