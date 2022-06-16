CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=0 encoder_conf=0.01 &
CUDA_VISIBLE_DEVICES=1 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=0 encoder_conf=0.005 &
CUDA_VISIBLE_DEVICES=2 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=0 encoder_conf=0.0025 &
CUDA_VISIBLE_DEVICES=3 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=0 encoder_conf=0.001 &
CUDA_VISIBLE_DEVICES=4 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=0 encoder_conf=0.0005 &
CUDA_VISIBLE_DEVICES=5 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=0 encoder_conf=0.00025 





CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=1 encoder_conf=0.01 &
CUDA_VISIBLE_DEVICES=1 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=1 encoder_conf=0.005 &
CUDA_VISIBLE_DEVICES=2 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=1 encoder_conf=0.0025 &
CUDA_VISIBLE_DEVICES=3 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=1 encoder_conf=0.001 &
CUDA_VISIBLE_DEVICES=4 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=1 encoder_conf=0.0005 &
CUDA_VISIBLE_DEVICES=5 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=1 encoder_conf=0.00025 




CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=2 encoder_conf=0.01 &
CUDA_VISIBLE_DEVICES=1 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=2 encoder_conf=0.005 &
CUDA_VISIBLE_DEVICES=2 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=2 encoder_conf=0.0025 &
CUDA_VISIBLE_DEVICES=3 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=2 encoder_conf=0.001 &
CUDA_VISIBLE_DEVICES=4 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=2 encoder_conf=0.0005 &
CUDA_VISIBLE_DEVICES=5 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=2 encoder_conf=0.00025


CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=3 encoder_conf=0.01 &
CUDA_VISIBLE_DEVICES=1 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=3 encoder_conf=0.005 &
CUDA_VISIBLE_DEVICES=2 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=3 encoder_conf=0.0025 &
CUDA_VISIBLE_DEVICES=3 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=3 encoder_conf=0.001 &
CUDA_VISIBLE_DEVICES=4 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=3 encoder_conf=0.0005 &
CUDA_VISIBLE_DEVICES=5 python train.py env=cartpole_swingup batch_size=64 action_repeat=8 seed=3 encoder_conf=0.00025










#CUDA_VISIBLE_DEVICES=0 python train.py env=cheetah_run batch_size=64 action_repeat=4 seed=0
#CUDA_VISIBLE_DEVICES=0 python train.py env=finger_spin batch_size=64 action_repeat=2 seed=0
#CUDA_VISIBLE_DEVICES=0 python train.py env=walker_walk batch_size=64 action_repeat=2 seed=0
#CUDA_VISIBLE_DEVICES=0 python train.py env=ball_in_cup_catch batch_size=64 action_repeat=4 seed=0

