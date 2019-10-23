# SMS-WSJ: A database for in-depth analysis of multi-channel source separation algorithms

This repository includes the scripts required to create the sms_wsj database
and a baseline asr system using KALDI (http://github.com/kaldi-asr/kaldi).

If you are using this code please cite the following paper:

```
@Article{SmsWsj19,
  author    = {Lukas Drude, Jens Heitkaemper, Christoph Boeddeker, Reinhold Haeb-Umbach},
  title     = {{SMS-WSJ: Database, performance measures, and baseline recipe for multi-channel source separation and recognition}},
  year      = {2019},
}
```

## Installation

Does not work with Windows.

Clone this Repo and install the package 
```bash
$ git clone https://github.com/fgnt/sms_wsj.git
$ cd sms_wsj
$ pip install --user -e ./
```

Get the RIR by downloading them (recommended)
```bash

```
or create them yourself using
```bash
$ make rirs RIR_DIR=/path/to/write/rirs/to
```
Then Set your KALDI_ROOT
```bash
$ export KALDI_ROOT=/path/to/kaldi
```
Afterwards you can create the database:
```bash
$ make SMS_WSJ_DIR=/path/to/write/db/to WSJ_DIR=/path/to/wsj
```
If desired the number of parallel jobs may be specified using the additonal
input num_jobs. Per default 16 parallel jobs are used.

Use the following command to train the baseline asr model:
```bash
$ python -m sms_wsj.train_baseline_asr with egs_path=$KALDI_ROOT/egs/ json_path=/path/to/sms_wsj.json
```
For more informationon possible 
The script has been tested with the KALDI hash "7637de77e0a77bf280bef9bf484e4f37c4eb9475"


## FAQ
### Q: What does the example id `0_4k0c0301_4k6c030t` mean?
A: The example id is a unique identifier for an example (sometime also knwon as utterance id).
The example id is a composition of the sperakers, the utterances and an scenario counter:

![Example ID](doc/images/example_id.svg)
=======

### Q: What to do if kaldi uses python3 instead of python2??

Add the follwing line to the ${KALDI_ROOT}/tools/envh.sh file:
```
export PATH=path/to/your/python2/bin/:${PATH}
```
