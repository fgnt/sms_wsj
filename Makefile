# uses environment variable WSJ_DIR if it is defined, otherwise falls back to default /net/fastdb/wsj
WSJ_DIR ?= /net/fastdb/wsj
SMS_WSJ_DIR ?= cache
RIR_DIR = $(SMS_WSJ_DIR)/rirs
JSON_PATH ?= cache
export NT_DATABASE_JSONS_DIR=$(JSON_PATH)
WSJ_8k_DIR ?= $(SMS_WSJ_DIR)/wsj_8k
on_the_fly = False # if True the reverberated data will be calculated on the fly and not saved to SMS_WSJ_DIR
num_jobs = 16

cache:
	mkdir cache

wsj_8k: $(SMS_WSJ_DIR)
	echo creating $(WSJ_8k_DIR)
	mpiexec -n ${num_jobs} python -m sms_wsj.database.wsj.write_wav with dst_dir=$(WSJ_8k_DIR) wsj_root=$(WSJ_DIR) sample_rate=8000

wsj.json: $(WSJ_8k_DIR)
	echo creating $(JSON_PATH)/wsj_8k.json
	python -m sms_wsj.wsj.create_json with json_path=$(JSON_PATH)/wsj_8k.json database_dir=$(WSJ_8k_DIR) as_wav=True

# The room impuls responses can be downloaded, so that they do not have to be created
# however if you want to recreate them use make rirs SMS_WSJ_DIR=/path/to/storage/
rirs:
	echo creating $(RIR_DIR)
	mpiexec -np $(num_jobs) python -m paderbox.database.wsj_bss.create_files with database_path=$(RIR_DIR)

sms_wsj.json: $(SMS_WSJ_DIR)/wsj_8k $(RIR_DIR)
	echo creating $(JSON_PATH)/sms_wsj.json
	python -m paderbox.database.wsj_bss.create_json with json_path=$(JSON_PATH)/sms_wsj.json rir_dir=$(RIR_DIR) wsj_json_path=$(JSON_PATH)/wsj_8k.json

sms_wsj: $(JSON_PATH)/sms_wsj.json
	echo creating $(SMS_WSJ_DIR) files
	mpiexec -np $(num_jobs) python -m sms_wsj.database.write_files with dst_dir $(RIR_DIR) json_path=$(JSON_PATH)/sms_wsj.json write_all=True new_json_path=$(JSON_PATH)/sms_wsj.json