"""
Example calls:
python -m paderbox.database.wsj.write_wav --dst-dir /net/vol/ldrude/share/wsj_8k --sample_rate 8000

mpiexec -np 20 python -m paderbox.database.wsj.write_wav --dst-dir /net/vol/ldrude/share/wsj_8k --sample_rate 8000

"""

from pathlib import Path
import click
import time
import logging
from tqdm import tqdm
import subprocess
import numpy as np
from paderbox.utils.mpi import COMM, RANK, SIZE, MASTER, IS_MASTER, bcast, barrier
import shutil

from paderbox.io.data_dir import wsj
from paderbox.io.audioread import read_nist_wsj
from paderbox.io.audiowrite import audiowrite


def resample_with_sox(x, rate_in, rate_out):
    if rate_in == rate_out:
        return x

    x = x.astype(np.float32)
    command = (
        f'sox -N -V1 -t f32 -r {rate_in} -c 1 - -t f32 -r {rate_out} -c 1 -'
    ).split()
    process = subprocess.run(
        command,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=x.tobytes(order="f")
    )
    return np.fromstring(process.stdout, dtype=np.float32)


def write_wavs(dst_dir: Path, wsj_root: Path, sample_rate):
    # Expect, that the dst_dir does not exist to make sure to not overwrite.
    if IS_MASTER:
        dst_dir.mkdir(parents=True, exist_ok=False)

    if IS_MASTER:
        for suffix in 'pl ndx ptx dot txt'.split():
            files = list(wsj_root.rglob(f"*.{suffix}"))
            logging.info(f"About to write {len(files)} {suffix} files.")
            for file in files:
                target = dst_dir / file.relative_to(wsj_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.is_file():
                    shutil.copy(file, target.parent)
            written_files = list(dst_dir.rglob(f"*.{suffix}"))
            logging.info(f"Writing {len(files)} {suffix} files.")
            assert len(written_files) == len(files), (files, written_files)

    if IS_MASTER:
        # Ignore .wv2 files since they are not referenced in our database
        # anyway
        wsj_nist_files = list(wsj_root.rglob("*.wv1"))
        logging.info(f"About to write {len(written_files)} wav files.")
    else:
        wsj_nist_files = None

    wsj_nist_files = bcast(wsj_nist_files)

    for nist_file in tqdm(wsj_nist_files[RANK::SIZE], disable=not IS_MASTER):
        nist_file: Path
        signal = read_nist_wsj(nist_file, expected_sample_rate=16000)[0]

        target = dst_dir / nist_file.with_suffix('.wav').relative_to(wsj_root)
        assert not target == nist_file, (nist_file, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        signal = resample_with_sox(signal, rate_in=16000, rate_out=sample_rate)
        audiowrite(signal, target, sample_rate=sample_rate)

    if IS_MASTER:
        created_files = list(wsj_root.rglob("*.wav"))
        logging.info(f"Written {len(written_files)} wav files.")
        assert len(wsj_nist_files) == len(created_files)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.INFO
    )

    @click.command()
    @click.option(
        '-d', '--dst-dir',
        help="Directory which will store the converted WSJ wav files",
        type=click.Path()
    )
    @click.option('--wsj-root', default=wsj,
                  help='Root directory to WSJ database', type=click.Path())
    @click.option('--sample_rate', default=16000,
                  help='16000 is the default.', type=int)
    def main(dst_dir, wsj_root, sample_rate):
        logging.info(f"Start - {time.ctime()}")

        wsj_root = Path(wsj_root).expanduser().resolve()
        dst_dir = Path(dst_dir).expanduser().resolve()
        assert not dst_dir == wsj_root, (wsj_root, dst_dir)

        write_wavs(dst_dir, wsj_root, sample_rate=sample_rate)
        logging.info(f"Done - {time.ctime()}")

    main()
