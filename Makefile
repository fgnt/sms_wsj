# uses environment variable WSJ_DIR if it is defined, otherwise falls back to default /net/fastdb/wsj
WSJ_DIR ?= /net/fastdb/wsj
SMS_WSJ_DIR ?= cache
RIR_DIR = $(SMS_WSJ_DIR)/rirs
JSON_PATH ?= cache
WSJ_8k_DIR ?= (SMS_WSJ_DIR)/wsj_8k
on_the_fly = False # if True the reverberated data will be calculated on the fly and not saved to SMS_WSJ_DIR
num_jobs = 16

cache:
	mkdir cache

wsj_8k: $(SMS_WSJ_DIR)
	echo `type python`
	mpiexec -n ${num_jobs} python -m sms_wsj.wsj.write_resampled_wav --dst-dir $(WSJ_8k_DIR) --wsj-root $(WSJ_DIR)

wsj.json: $(WSJ_8k_DIR)
	echo creating $(WSJ_8k_DIR)
	python -m sms_wsj.wsj.create_json -j $(JSON_PATH)/wsj.json -db $(WSJ_8k_DIR) --wav

# The room impuls responses can be downloaded, so that they do not have to be created
# however if you want to recreate them use make rirs SMS_WSJ_DIR=/path/to/storage/
rirs:
	echo creating $(RIR_DIR)
	mpiexec -np $(num_jobs) python -m sms_wsj.create_files with database_path=$(RIR_DIR)

sms_wsj.json: $(SMS_WSJ_DIR)/wsj_8k $(RIR_DIR)
	echo creating $(JSON_PATH)/sms_wsj.json
	python -m sms_wsj.create_json -j $(JSON_PATH)/sms_wsj.json -db $(RIR_DIR) --wsj-json-path $(WSJ_8k_DIR)

sms_wsj: $(JSON_PATH)/sms_wsj.json
	echo creating $(SMS_WSJ_DIR)/wsj_8k