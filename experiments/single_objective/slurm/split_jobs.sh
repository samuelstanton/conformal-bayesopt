#!/bin/bash
dimInd=$(( $1 % 3 ))
seed=$(( $1 % 25 ))
task="levy"
batchsize=1
dims=(5 10 20)
dim=${dims[$dimInd]}

DATA_DIR=/scratch/ss13641/code/remote/conformal-bayesopt/experiments/single_objective/data/$2
mkdir -p $DATA_DIR

pwd
echo $DATA_DIR
echo $seed $dim

python ./runner.py --seed=$seed --n_batch=50 \
       --problem=${task} --dim=$dim --batch_size=${batchsize} \
       --output=${DATA_DIR}/${task}${dim}_q${batchsize}_${seed}.pt

