# Local test (if cuda not present), make sure you have .pth file available under experiments/search_logs/s2/cifar10/0.0_0.0003-1
# python src/landscape/vis_loss_landscape.py --checkpoint_epoch 50  --disable_cuda --auxiliary_skip --decay linear --data /Users/bo/dataset/cifar/ --space s2  --task_id 2 --test_infer

# server
space='s3'
dataset='cifar10'
ep=47 # the epoch of best validation acc 
gpu=0
for task_id in $(seq 0 0);do
    echo $gpu $task_id
    #darts-
    python src/landscape/vis_loss_landscape.py --checkpoint_epoch $ep --gpu $gpu --auxiliary_skip --decay linear --data /home/work/dataset/cifar/ --space $space --dataset $dataset --task_id $task_id --job_id 0 --test_infer &> landscape_${space}_${dataset}_task_${task_id}_ep${ep}.log &
    #darts vanilla
    #python src/landscape/vis_loss_landscape.py --checkpoint_epoch $ep --gpu $gpu --auxiliary_skip --skip_beta 0 --data /home/work/dataset/cifar/ --space $space --dataset $dataset --task_id $task_id --job_id 0 --test_infer &> landscape_${space}_${dataset}_task_${task_id}_ep${ep}.log &
done