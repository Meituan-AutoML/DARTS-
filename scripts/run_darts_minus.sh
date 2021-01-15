#!/bin/sh
SLURM_ARRAY_JOB_ID=0
SPACE='s3'
DATASET='cifar10'
WEIGHT_DECAY=0.0003
DROP_PROB=0.0
beta=1
DECAY_TYPE='linear'
gpu=0

for i in $(seq 0 3);do
    let SLURM_ARRAY_TASK_ID=$i
    echo $i $gpu
    # don't compute hessian
    python src/search/train_search.py --data /home/work/dataset/cifar --gpu $gpu --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --seed $i --report_freq_hessian 50 --space $SPACE --dataset $DATASET --drop_path_prob $DROP_PROB --skip_beta $beta --decay $DECAY_TYPE --auxiliary_skip  > train-search-darts-minus-$SPACE-$DATASET-job-$SLURM_ARRAY_JOB_ID-task-$SLURM_ARRAY_TASK_ID-drop-$DROP_PROB-seed-$i-skip-beta_$beta-decay-${DECAY_TYPE}.log 2>&1 &
    let gpu=($gpu+1)%8
done
