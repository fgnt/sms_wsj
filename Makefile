SHELL = /bin/bash

# Uses environment variable WSJ_DIR if it is defined, otherwise falls back to default /net/fastdb/wsj
WSJ_DIR ?= /net/fastdb/wsj
WSJ0_DIR ?= $(WSJ_DIR)
WSJ1_DIR ?= $(WSJ_DIR)
SMS_WSJ_DIR ?= cache
RIR_DIR = $(SMS_WSJ_DIR)/rirs
JSON_DIR ?= $(SMS_WSJ_DIR)
WSJ_8K_ZEROMEAN_DIR ?= $(SMS_WSJ_DIR)/wsj_8k_zeromean
WRITE_ALL = True # If True the reverberated data will be calculated on the fly and not saved to SMS_WSJ_DIR
num_jobs = $(shell nproc --all)
DEBUG = False # If True, create less entries in sms_wsj.json just for debugging.
# Example for call on the paderborn parallel computing center
# ccsalloc --res=rset=1:mem=2G:ncpus=8 -t 4h make all --num_jobs=8

export OMP_NUM_THREADS = 1
export MKL_NUM_THREADS = 1

all: sms_wsj

cache:
	mkdir cache

wsj_8k_zeromean: $(WSJ_8K_ZEROMEAN_DIR)
$(WSJ_8K_ZEROMEAN_DIR): $(WSJ_DIR)
	@echo creating $(WSJ_8K_ZEROMEAN_DIR)
	@echo using $(num_jobs) parallel jobs
	mpiexec -np ${num_jobs} python -m sms_wsj.database.wsj.write_wav \
	with dst_dir=$(WSJ_8K_ZEROMEAN_DIR) wsj0_root=$(WSJ0_DIR) wsj1_root=$(WSJ1_DIR) sample_rate=8000

wsj_8k_zeromean.json: $(JSON_DIR)/wsj_8k_zeromean.json
$(JSON_DIR)/wsj_8k_zeromean.json: $(WSJ_8K_ZEROMEAN_DIR) | $(JSON_DIR)
	@echo creating $(JSON_DIR)/wsj_8k_zeromean.json
	python -m sms_wsj.database.wsj.create_json \
	with json_path=$(JSON_DIR)/wsj_8k_zeromean.json database_dir=$(WSJ_8K_ZEROMEAN_DIR) as_wav=True

sms_wsj.json: $(JSON_DIR)/sms_wsj.json
$(JSON_DIR)/sms_wsj.json: $(RIR_DIR) $(JSON_DIR)/wsj_8k_zeromean.json | $(JSON_DIR)
	@echo creating $(JSON_DIR)/sms_wsj.json
	python -m sms_wsj.database.create_json \
	with json_path=$(JSON_DIR)/sms_wsj.json rir_dir=$(RIR_DIR) wsj_json_path=$(JSON_DIR)/wsj_8k_zeromean.json debug=$(DEBUG)

sms_wsj: $(SMS_WSJ_DIR)/sms_wsj
$(SMS_WSJ_DIR)/sms_wsj: $(JSON_DIR)/sms_wsj.json | $(SMS_WSJ_DIR)
	@echo creating $(SMS_WSJ_DIR) files
	@echo This amends the sms_wsj.json with the new paths.
	@echo using $(num_jobs) parallel jobs
	mpiexec -np ${num_jobs} python -m sms_wsj.database.write_files \
	with dst_dir=$(SMS_WSJ_DIR) json_path=$(JSON_DIR)/sms_wsj.json write_all=$(WRITE_ALL) new_json_path=$(JSON_DIR)/sms_wsj.json debug=$(DEBUG)

# The room impuls responses can be downloaded, so that they do not have to be created
# however if you want to recreate them use "make rirs RIR_DIR=/path/to/storage/"
rirs:
	@echo creating $(RIR_DIR)
	git clone https://github.com/boeddeker/rir-generator.git ./sms_wsj/reverb/rirgen_rep
	pip install -e sms_wsj/reverb/rirgen_rep/python/
	mpiexec -np ${num_jobs} python -m sms_wsj.database.create_rirs \
	with database_path=$(RIR_DIR)

# To manually download and extract the rirs, execute the following after downloading all files from https://zenodo.org/record/3517889
# cat $(RIR_DIR)/sms_wsj.tar.gz.* > $(RIR_DIR)/sms_wsj.tar.gz
# tar -C $(RIR_DIR)/ -zxvf $(RIR_DIR)/sms_wsj.tar.gz
$(RIR_DIR):
	@echo "RIR directory does not exist, starting download, to recreate the RIRs use 'make rirs'."
	mkdir -p $(RIR_DIR)
	echo $(RIR_DIR)
	wget -qO- https://zenodo.org/record/3517889/files/sms_wsj.tar.gz.parta{a,b,c,d,e} \
	| tar -C $(RIR_DIR)/ -zx --checkpoint=10000 --checkpoint-action=echo="%u/5530000 %c"

$(JSON_DIR):
	@echo "JSON_DIR is wrongly set or directory does not exist."
	@echo "Please specify an existing JSON_DIR directory using the  variable, JSON_DIR =" $(JSON_DIR)
	exit 1

$(WSJ_DIR):
	@echo "WSJ directory does not exist."
	@echo "Please specify an existing WSJ directory using the WSJ_DIR variable, WSJ_DIR =" $(WSJ_DIR)
	exit 1
