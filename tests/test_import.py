import os
import subprocess
import importlib
from pathlib import Path

import pytest

import sms_wsj


def get_module_name_from_file(file, package_path=Path(sms_wsj.__file__).parent):
    """
    >> import sms_wsj
    >> file = sms_wsj.io.__file__
    >> file  # doctest: +ELLIPSIS
    '.../sms_wsj/io.py'
    >> get_module_name_from_file(file)
    'sms_wsj.io'
    >> file = sms_wsj.database.__file__
    >> file  # doctest: +ELLIPSIS
    '.../sms_wsj/database/__init__.py'
    >> get_module_name_from_file(pb.transform.__file__)
    'sms_wsj.database'
    """

    assert package_path in file.parents, (package_path, file)
    file = file.relative_to(package_path.parent)
    parts = list(file.with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop(-1)
    module_path = '.'.join(parts)
    return module_path


@pytest.fixture(scope="session", autouse=True)
def dummy_kaldi_root(tmp_path_factory):
    kaldi_root = tmp_path_factory.mktemp("kaldi")
    (kaldi_root / 'src' / 'base' / '.depend.mk').mkdir(parents=True)

    if 'KALDI_ROOT' not in os.environ:
        os.environ.setdefault('KALDI_ROOT', str(kaldi_root))

    return kaldi_root


class TestImport:
    python_files = Path(sms_wsj.__file__).parent.glob('**/*.py')

    @pytest.mark.parametrize('py_file', [
            pytest.param(
                py_file,
                id=get_module_name_from_file(py_file))
            for py_file in python_files
    ])
    def test_import(self, py_file: Path, with_importlib=True):
        """
        Import `py_file` into the system

        Args:
            py_file: Python file to import
            with_importlib: If True, use `importlib` for importing. Else, use
                            `subprocess.run`: It is considerably slower but may
                             have better readable test output
        """
        import_name = get_module_name_from_file(py_file)
        suffix = Path(py_file).suffix
        try:
            if with_importlib:
                _ = importlib.import_module(import_name)
            else:
                _ = subprocess.run(
                    ['python', '-c', f'import {import_name}'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    universal_newlines=True,
                )
        except (
                ImportError,
                ModuleNotFoundError,
                subprocess.CalledProcessError,
        ) as e:
            try:
                err = e.stderr
            except AttributeError:
                err = 'See Traceback above'
            assert False, f'Cannot import file "{import_name}{suffix}" \n\n' \
                          f'stderr: {err}'
