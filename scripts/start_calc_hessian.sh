#!/bin/sh
# Local Test
# python src/search/calc_hessian.py --disable_cuda --data /Users/bo/dataset/cifar/ --space s3 --auxiliary_skip --decay linear --task_id 3 --ev_start_epoch 50

# Server
job=0
ss='s2'
dataset='cifar10'
decay='linear'
beta=1

for i in $(seq 0 2);do
    let gpu=$i
    python src/search/calc_hessian.py --compute_hessian --report_freq_hessian 2 --gpu $gpu --data /home/work/dataset/cifar/ --dataset $dataset --space $ss --auxiliary_skip --decay $decay --skip_beta $beta --job_id $job --task_id $i --ev_start_epoch 0 >& hessian_${ss}_${dataset}_job_${job}_task_${i}_beta_${beta}_decay_${decay}.log &
done