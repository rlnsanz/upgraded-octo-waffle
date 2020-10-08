#!/bin/bash

set -e

INIT='weak'
NGPUS='8'
MODEL='squeezenet'
TBLOG=''

echo $1
echo $2
case ${1:-"base"} in

    judy)
    TBLOG="-owner judy -loglvl $2"
    MOUD=''
    MODEL="${MODEL}log"
    cp train_raw.py train.py
    ;;

    chuck)
    TBLOG="-owner chuck -loglvl $2"
    MOUD=''
    MODEL="${MODEL}log"
    cp train_raw.py train.py
    ;;

    mike)
    TBLOG='-owner mike -loglvl 4'
    MOUD=''
    MODEL="${MODEL}log"
    cp train_raw.py train.py
    ;;

    flor)
    TBLOG="-owner flor -loglvl $2"
    MOUD=''
    MODEL="${MODEL}log"
    cp train_raw.py train.py
    ;;

    base)
    MOUD=''
    cp train_raw.py train.py
    ;;

    naive)
    MOUD=''
    cp train_manualchkpt.py train.py
    ;;

    exec)
    MOUD="--flor=name:$MODEL"
    cp train_transformed.py train.py
    ;;

    reexec)
    MOUD="--flor=name:$MODEL,mode:reexec,memo:blessed.json,predinit:$INIT"
    cp train_transformed.py train.py
    ;;

    mikeparallel)
    MOUD="--flor=name:$MODEL,mode:reexec,memo:blessed.json,predinit:$INIT,pid:$2,ngpus:$NGPUS"
    MODEL="${MODEL}log"
    cp train_parallel_mike_log.py train.py
    ;;

    parallel)
    MOUD="--flor=name:$MODEL,mode:reexec,memo:blessed.json,predinit:$INIT,pid:$2,ngpus:$NGPUS"
    cp train_parallel.py train.py
    ;;

esac

python3 train.py $MOUD -net $MODEL $TBLOG
