# uses environment variable WSJ_DIR if it is defined, otherwise falls back to default /net/fastdb/wsj
WSJ_DIR ?= /net/fastdb/wsj
SMS_WSJ_DIR ?= cache
RIR_DIR = $(SMS_WSJ_DIR)/rirs
JSON_PATH ?= cache
WSJ_8K_DIR ?= $(SMS_WSJ_DIR)/wsj_8k
on_the_fly = False # if True the reverberated data will be calculated on the fly and not saved to SMS_WSJ_DIR
num_jobs = 16

all: sms_wsj

cache:
	mkdir cache

wsj_8k: $(WSJ_8K_DIR)
$(WSJ_8K_DIR): $(WSJ_DIR)
	echo creating $(WSJ_8K_DIR)
	mpiexec -n ${num_jobs} python -m sms_wsj.database.wsj.write_wav with dst_dir=$(WSJ_8K_DIR) wsj_root=$(WSJ_DIR) sample_rate=8000

wsj_8k.json: $(WSJ_8K_DIR) $(JSON_PATH)
$(JSON_PATH)/wsj_8k.json: $(WSJ_8K_DIR)
	echo creating $(JSON_PATH)/wsj_8k.json
	python -m sms_wsj.database.wsj.create_json with json_path=$(JSON_PATH)/wsj_8k.json database_dir=$(WSJ_8K_DIR) as_wav=True

sms_wsj.json: $(JSON_PATH)/sms_wsj.json $(JSON_PATH)
$(JSON_PATH)/sms_wsj.json: $(JSON_PATH)/wsj_8k.json $(RIR_DIR)
	echo creating $(JSON_PATH)/sms_wsj.json
	python -m sms_wsj.database.create_json with json_path=$(JSON_PATH)/sms_wsj.json rir_dir=$(RIR_DIR) wsj_json_path=$(JSON_PATH)/wsj_8k.json

sms_wsj: $(SMS_WSJ_DIR)/sms_wsj $(SMS_WSJ_DIR)
$(SMS_WSJ_DIR)/sms_wsj: $(JSON_PATH)/sms_wsj.json $(SMS_WSJ_DIR)
	echo creating $(SMS_WSJ_DIR) files
	mpiexec -np $(num_jobs) python -m sms_wsj.database.write_files with dst_dir=$(SMS_WSJ_DIR) json_path=$(JSON_PATH)/sms_wsj.json write_all=True new_json_path=$(JSON_PATH)/sms_wsj.json

# The room impuls responses can be downloaded, so that they do not have to be created
# however if you want to recreate them use "make rirs SMS_WSJ_DIR=/path/to/storage/"
rirs:
	echo creating $(RIR_DIR)
	mpiexec -np $(num_jobs) python -m sms.database.create_rirs with database_path=$(RIR_DIR)

$(RIR_DIR):
	echo "RIR directory does not exist, either download it from ... or use 'make rirs' to create it."
	echo "Use the RIR_DIR variable to point to the RIR directory, RIR_DIR =" $(RIR_DIR)
	exit 1

$(WSJ_DIR):
	echo "WSJ directory does not exist."
	echo "Please specify a existing WSJ directory using the WSJ_DIR variable, WSJ_DIR =" $(WSJ_DIR)
	exit 1
