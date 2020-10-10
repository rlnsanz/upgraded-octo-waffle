#!/bin/bash

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark

echo "running benchmark run level 1"
./run.sh $1 1

echo "running benchmark run level 2"
./run.sh $1 2

echo "running benchmark run level 3"
./run.sh $1 3

echo "running benchmark run level 4"
./run.sh $1 4

sleep 3
ret_code=$?; if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"