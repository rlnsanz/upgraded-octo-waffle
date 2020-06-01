#!/bin/bash

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)

#SBATCH --cpus-per-task=4 # number of cores per task

#SBATCH --gres=gpu:1

#SBATCH --nodelist=XXX # if you need specific nodes

#SBATCH -t 2-10:30 # time requested (D-HH:MM)

#SBATCH --output=slurm_output/exec-%j.out

echo pwd: $(pwd)
echo hostname: $(hostname)
echo Starting job ...
source activate ml

echo STARTED: $(date +"%T.%6N")

export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=0 python ../../train_transformed_parallel.py \
-net resnet152 \
--flor=name:RESNET,mode:reexec,predinit:strong,memo:memo.json,ngpus:4,pid:0 &

CUDA_VISIBLE_DEVICES=1 python ../../train_transformed_parallel.py \
-net resnet152 \
--flor=name:RESNET,mode:reexec,predinit:strong,memo:memo.json,ngpus:4,pid:1 &

CUDA_VISIBLE_DEVICES=2 python ../../train_transformed_parallel.py \
-net resnet152 \
--flor=name:RESNET,mode:reexec,predinit:strong,memo:memo.json,ngpus:4,pid:2 &

CUDA_VISIBLE_DEVICES=3 python ../../train_transformed_parallel.py \
-net resnet152 \
--flor=name:RESNET,mode:reexec,predinit:strong,memo:memo.json,ngpus:4,pid:3

echo COMPLETED: $(date +"%T.%6N")
