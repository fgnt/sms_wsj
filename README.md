# SMS-WSJ: A database for in-depth analysis of multi-channel source separation algorithms

![Example ID](doc/images/room.svg)

This repository includes the scripts required to create the SMS-WSJ database, a spatial clustering baseline for separation,
and a baseline ASR system using Kaldi (http://github.com/kaldi-asr/kaldi).

## Why does this database exist?

In multi-speaker ASR the [WSJ0-2MIX database](https://www.merl.com/demos/deep-clustering) and the spatialized version thereof are widely used.
Observing that research in multi-speaker ASR is often hard to compare because some researchers pretrain on WSJ, while others train only on WSJ0-2MIX or create other sub-lists of WSJ we decided to use a fixed file list which is suitable for training an ASR system without additional audio data.
Punctuation pronunciation utterances are filtered to further facilitate end-to-end ASR experiments.

Further, we argue that the tooling around [WSJ0-2MIX database](https://www.merl.com/demos/deep-clustering) and the spatialized version thereof is very limited.
Therefore, we provide a spatial clustering baseline and a Kaldi ASR baseline.
Researchers can now easily improve parts of the pipeline while ensuring that they can fairly compare with baseline results reported in the associated Arxiv paper.

## How can I cite this work? Where are baseline results?
The associated paper can be found here: https://arxiv.org/abs/1910.13934
If you are using this code please cite the paper as follows:

```
@Article{SmsWsj19,
  author    = {Drude, Lukas and Heitkaemper, Jens and Boeddeker, Christoph and Haeb-Umbach, Reinhold},
  title     = {{SMS-WSJ}: Database, performance measures, and baseline recipe for multi-channel source separation and recognition},
  journal   = {arXiv preprint arXiv:1910.13934},
  year      = {2019},
}
```

## Installation

Does not work with Windows.

Clone this repository and install the package:
```bash
$ git clone https://github.com/fgnt/sms_wsj.git
$ cd sms_wsj
$ pip install --user -e ./
```

Set your KALDI_ROOT environment variable:
```bash
$ export KALDI_ROOT=/path/to/kaldi
```
We assume that the Kaldi WSJ baseline has been created with the `run.sh` script.
This is important to be able to use the Kaldi language model.
To build the ASR baseline the structures created during the first stage of
the `run.sh` script are required.
The ASR baseline uses the language models created during the same stage.
Afterwards you can create the database:
```bash
$ make WSJ_DIR=/path/to/wsj SMS_WSJ_DIR=/path/to/write/db/to
```
If desired the number of parallel jobs may be specified using the additonal
input num_jobs. Per default `nproc --all` parallel jobs are used.


The RIRs are downloaded by default, to generate them yourself see [here](#q-i-want-to-generate-the-rirs-myself-how-can-i-do-that).


Use the following command to train the baseline ASR model:
```bash
$ python -m sms_wsj.train_baseline_asr with egs_path=$KALDI_ROOT/egs/ json_path=/path/to/sms_wsj.json
```
The script has been tested with the KALDI Git hash "7637de77e0a77bf280bef9bf484e4f37c4eb9475"


## FAQ
### Q: How large is the disc capacity required for the database?
A: The total disc usage is 442.1 GiB.  

directory         | disc usage
:------------------|--------------:
tail              |      120.1 GiB  
early             |      120.1 GiB  
observation       |      60.0 GiB  
noise             |      60.0 GiB  
rirs              |      52.6 GiB  
wsj_8k_zeromean   |      29.2 GiB  
sms_wsj.json      |      139,7 MiB  
wsj_8k.json       |      31,6 MiB  

### Q: How many hours takes the database creation?
A: Using 32 cores the database creation without recalculating the RIRs takes around 4 hours.

### Q: What does the example ID `0_4k6c0303_4k4c0319` mean?
A: The example ID is a unique identifier for an example (sometime also known as utterance ID).
The example ID is a composition of the sperakers, the utterances and an scenario counter:

![Example ID](doc/images/example_id.svg)
=======

### Q: What to do if Kaldi uses Python 3 instead of Python 2?
The Python code in this repository requires Python 3.6. However, Kaldi runs
on Python 2.7. To solve this mismatch Kaldi has to be forced to switch the
Python version using the `path.sh`. Therefore, add the follwing line to
the `${KALDI_ROOT}/tools/envh.sh` file:
```
export PATH=path/to/your/python2/bin/:${PATH}
```

### Q: I want to generate the RIRs myself. How can I do that?
To generate the RIRs you can run the following command:
```bash
$ mpiexec -np $(nproc --all) python -m sms_wsj.database.create_rirs with database_path=cache/rirs
```
The expected runtime will be around `1900/(ncpus - 1)` hours.
When you have access to an HPC system, you can replace `mpiexec -np $(nproc --all)` with an HPC command.
It is enough, when each job has access to 2GB RAM.
