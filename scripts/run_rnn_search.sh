#!/bin/sh
gpu=0
for i in $(seq 0 1);do
    let gpu=$gpu+$i
    echo $i $gpu
    python src/rnn/train_search.py --data /home/work/dataset/penn --gpu $gpu --seed $i >& darts_train_search_ptb_seed_${i}.log &
    sleep 1
done
