"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To use a consistent encoding
from codecs import open
# Always prefer setuptools over distutils
from distutils.core import setup
from os import path
from setuptools import find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sms_wsj',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.0',

    description='Python package for the spatialized multi speaker (SMS) wsj'
                ' database',
    long_description=long_description,

    # The project's main homepage.
    url='http://ei.upb.de/nt/',

    # Author details
    author='Department of Communications Engineering',
    author_email='sek@nt.upb.de',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Researcher Source Separation',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='database, multi channel, multi speaker, simulated',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'dlp_mpi',
        'lazy_dataset',
        'sacred',
        'sh',
        'soundfile',
        # TODO scipy>=1.3.0
    ],

    # Installation problems in a clean, new environment:
    # 1. `cython` and `scipy` must be installed manually before using
    # `pip install`
    # 2. `pyzmq` has to be installed manually, otherwise `pymatbridge` will
    # complain

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'all': [
            'pytest',
            'IPython',
            'matplotlib',
            'scipy',
            'pandas',
            'einops',
            'nara_wpe',
            'paderbox',
            'pb_bss @ git+https://github.com/fgnt/pb_bss',  # Used for the table, that is shown in the publication
        ]
    },
)
