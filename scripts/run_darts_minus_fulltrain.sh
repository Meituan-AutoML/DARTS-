#!/bin/sh
# RDARTS Evaluation
SLURM_ARRAY_JOB_ID=0
SPACE='s5'
DATASET='cifar100'
DROP_PROB=0.0
WEIGHT_DECAY=0.0003
#CONFIG="--layers 20 --init_channels 36"
CONFIG=""

for i in $(seq 0 2);do
    let j=$i
    let SLURM_ARRAY_TASK_ID=$i
    echo $i $j
    python src/evaluation/train.py --data /home/work/dataset/cifar $CONFIG --gpu $j --cutout --auxiliary --job_id $SLURM_ARRAY_JOB_ID --task_id $i --seed 1 --space $SPACE --dataset $DATASET --search_dp $DROP_PROB --search_wd $WEIGHT_DECAY --search_task_id $i --archs_config_file ./experiments/search_logs/darts_minus_arch.yaml > train_darts_minus_$DATASET-$SPACE-$DROP_PROB-$WEIGHT_DECAY-task-$i.log  2>&1 &
done


#SGAS Evaluation
# gpu=5
# for i in $(seq 0 1);do
#     echo $i $gpu
#     export CUDA_VISIBLE_DEVICES=$gpu && python src/eval/train.py --data /home/work/dataset/cifar --cutout --auxiliary --arch DARTS_MINUS_C10_LINEAR_S3_$i >& DARTS_MINUS_C10_LINEAR_S3_${i}_fulltrain.log &
#     let gpu=($gpu+1)%8
# done