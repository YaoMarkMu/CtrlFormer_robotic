pip install einops
for sd in 0 1 2 3 4 5 6
do
    CUDA_VISIBLE_DEVICES=$sd python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=$sd scale=0.0425 tag=trans_same&
    sleep 5
done

CUDA_VISIBLE_DEVICES=7 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=7 scale=0.0425  tag=trans_same
sleep 72h

###############################################################################################################

pip install einops
for sd in 0 1 2 3 4 5 6
do
    CUDA_VISIBLE_DEVICES=$sd python3 train.py  batch_size=512 action_repeat=2 env_index=1 seed=$sd scale=0.0425  tag=trans_same&
    sleep 5
done

CUDA_VISIBLE_DEVICES=7 python3 train.py  batch_size=512 action_repeat=2 env_index=1 seed=7 scale=0.0425  tag=trans_same
sleep 72h

###############################################################################################################

pip install einops
for sd in 0 1 2 3 4 5 6
do
    CUDA_VISIBLE_DEVICES=$sd python3 train.py  batch_size=512 action_repeat=2 env_index=2 seed=$sd scale=0.0425  tag=trans_same&
    sleep 5
done

CUDA_VISIBLE_DEVICES=7 python3 train.py  batch_size=512 action_repeat=2 env_index=2 seed=7 scale=0.0425  tag=trans_same
sleep 72h

###############################################################################################################

pip install einops
for sd in 0 1 2 3 4 5 6
do
    CUDA_VISIBLE_DEVICES=$sd python3 train.py  batch_size=512 action_repeat=8 env_index=3 seed=$sd scale=0.0425  tag=trans_same&
    sleep 5
done

CUDA_VISIBLE_DEVICES=7 python3 train.py  batch_size=512 action_repeat=8 env_index=3 seed=7 scale=0.0425  tag=trans_same
sleep 72h

###############################################################################################################

pip install einops
for sd in 0 1 2 3 4 5 6
do
    CUDA_VISIBLE_DEVICES=$sd python3 train.py  batch_size=512 action_repeat=4 env_index=4 seed=$sd scale=0.0425  tag=trans_same&
    sleep 5
done

CUDA_VISIBLE_DEVICES=7 python3 train.py  batch_size=512 action_repeat=4 env_index=4 seed=7 scale=0.0425  tag=trans_same
sleep 72h