SEED = 1
NOW :=`date "+%F-%T"`
CURRDIR := `pwd`
G ?= 0

DATA_DIR=$(CURRDIR)/datasets
CHECKPOINT_DIR=$(CURRDIR)/checkpoints
RESULT_DIR=$(CURRDIR)/results

all:
	NV_GPU=$(G) ./run_and_time.sh $(M) $(G) | tee benchmark-$(NOW).log