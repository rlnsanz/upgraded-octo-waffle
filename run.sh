#!/bin/bash

set -e

INIT='strong'
NGPUS='4'
MODEL='squeezenet'

echo $1
echo $2
case ${1:-"base"} in

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

    parallel)
    MOUD="--flor=name:$MODEL,mode:reexec,memo:blessed.json,predinit:$INIT,pid:$2,ngpus:$NGPUS"
    cp train_parallel.py train.py
    ;;

esac

python3 train.py $MOUD -net $MODEL