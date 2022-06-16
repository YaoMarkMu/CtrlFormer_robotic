# MOREformer: Multi-Objective REpresentation learning via multi-path transformer for reinforcement learning

This is a PyTorch implementation of **MOREformer**.

The whole framework is shown as follow:
![MOREformer Framework](pngs/framework.PNG)

## Requirements

We assume you have access to a gpu that can run CUDA 11.2 and Driver 460.91.03.

python 3.6.13
torch  1.9.1
gcc    9.2

Then, the simplest way to install all required dependencies is to create an anaconda environment by running


```

conda env create -f conda_env.yml

```

After the instalation ends you can activate your environment with
```

conda activate MOREformer

```

## Rebuttal Contrastive   cons_ab分支
```python
CUDA_VISIBLE_DEVICES=0 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=0  tag=cnn_same  scale=0.0425&
CUDA_VISIBLE_DEVICES=1 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=1  tag=cnn_same  scale=0.0425&
CUDA_VISIBLE_DEVICES=2 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=2  tag=cnn_same  scale=0.0425&
CUDA_VISIBLE_DEVICES=3 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=3  tag=cnn_same  scale=0.0425&
CUDA_VISIBLE_DEVICES=4 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=4  tag=cnn_same  scale=0.0425&
CUDA_VISIBLE_DEVICES=5 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=5  tag=cnn_same  scale=0.0425&
CUDA_VISIBLE_DEVICES=6 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=6  tag=cnn_same  scale=0.0425&
CUDA_VISIBLE_DEVICES=7 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=7  tag=cnn_same  scale=0.0425

sleep 72h
```

## Rebuttal LR   Main分支
```python
CUDA_VISIBLE_DEVICES=0 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=0 scale=0.0425 tag=trans_same&
CUDA_VISIBLE_DEVICES=1 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=1 scale=0.0425 tag=trans_same&
CUDA_VISIBLE_DEVICES=2 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=2 scale=0.0425 tag=trans_same&
CUDA_VISIBLE_DEVICES=3 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=3 scale=0.0425 tag=trans_same&
CUDA_VISIBLE_DEVICES=4 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=4 scale=0.0425 tag=trans_same&
CUDA_VISIBLE_DEVICES=5 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=5 scale=0.0425 tag=trans_same&
CUDA_VISIBLE_DEVICES=6 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=6 scale=0.0425 tag=trans_same&
CUDA_VISIBLE_DEVICES=7 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=7 scale=0.0425 tag=trans_same

sleep 72h

CUDA_VISIBLE_DEVICES=0 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=0 scale=0.025 tag=trans_same&
CUDA_VISIBLE_DEVICES=1 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=1 scale=0.025 tag=trans_same&
CUDA_VISIBLE_DEVICES=2 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=2 scale=0.025 tag=trans_same&
CUDA_VISIBLE_DEVICES=3 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=3 scale=0.025 tag=trans_same&
CUDA_VISIBLE_DEVICES=4 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=4 scale=0.025 tag=trans_same&
CUDA_VISIBLE_DEVICES=5 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=5 scale=0.025 tag=trans_same&
CUDA_VISIBLE_DEVICES=6 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=6 scale=0.025 tag=trans_same&
CUDA_VISIBLE_DEVICES=7 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=7 scale=0.025 tag=trans_same

sleep 72h

CUDA_VISIBLE_DEVICES=0 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=0 scale=0.085 tag=trans_same&
CUDA_VISIBLE_DEVICES=1 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=1 scale=0.085 tag=trans_same&
CUDA_VISIBLE_DEVICES=2 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=2 scale=0.085 tag=trans_same&
CUDA_VISIBLE_DEVICES=3 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=3 scale=0.085 tag=trans_same&
CUDA_VISIBLE_DEVICES=4 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=4 scale=0.085 tag=trans_same&
CUDA_VISIBLE_DEVICES=5 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=5 scale=0.085 tag=trans_same&
CUDA_VISIBLE_DEVICES=6 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=6 scale=0.085 tag=trans_same&
CUDA_VISIBLE_DEVICES=7 python3 train.py  batch_size=512 action_repeat=2 env_index=0 seed=7 scale=0.085 tag=trans_same

sleep 72h
```

## Testing multi_task

```python
CUDA_VISIBLE_DEVICES=0 python train.py  batch_size=512 action_repeat=2 env_index=0 seed=0 &
CUDA_VISIBLE_DEVICES=1 python train.py  batch_size=512 action_repeat=2 env_index=0 seed=1 &
CUDA_VISIBLE_DEVICES=2 python train.py  batch_size=512 action_repeat=2 env_index=0 seed=2 &
CUDA_VISIBLE_DEVICES=3 python train.py  batch_size=512 action_repeat=2 env_index=0 seed=3 &
CUDA_VISIBLE_DEVICES=4 python train.py  batch_size=512 action_repeat=2 env_index=0 seed=4 &
```

```python
CUDA_VISIBLE_DEVICES=5 python train.py  batch_size=512 action_repeat=2 env_index=1 seed=0 &
CUDA_VISIBLE_DEVICES=6 python train.py  batch_size=512 action_repeat=2 env_index=1 seed=1 &
CUDA_VISIBLE_DEVICES=7 python train.py  batch_size=512 action_repeat=2 env_index=1 seed=2 &
CUDA_VISIBLE_DEVICES=8 python train.py  batch_size=512 action_repeat=2 env_index=1 seed=3 &
CUDA_VISIBLE_DEVICES=9 python train.py  batch_size=512 action_repeat=2 env_index=1 seed=4 &
```


```python
CUDA_VISIBLE_DEVICES=10 python train.py  batch_size=512 action_repeat=2 env_index=2 seed=0 &
CUDA_VISIBLE_DEVICES=11 python train.py  batch_size=512 action_repeat=2 env_index=2 seed=1 &
CUDA_VISIBLE_DEVICES=12 python train.py  batch_size=512 action_repeat=2 env_index=2 seed=2 &
CUDA_VISIBLE_DEVICES=13 python train.py  batch_size=512 action_repeat=2 env_index=2 seed=3 &
CUDA_VISIBLE_DEVICES=14 python train.py  batch_size=512 action_repeat=2 env_index=2 seed=4 &
```

```python
CUDA_VISIBLE_DEVICES=15 python train.py  batch_size=512 action_repeat=8 env_index=3 seed=0 &
CUDA_VISIBLE_DEVICES=16 python train.py  batch_size=512 action_repeat=8 env_index=3 seed=1 &
CUDA_VISIBLE_DEVICES=17 python train.py  batch_size=512 action_repeat=8 env_index=3 seed=2 &
CUDA_VISIBLE_DEVICES=18 python train.py  batch_size=512 action_repeat=8 env_index=3 seed=3 &
CUDA_VISIBLE_DEVICES=19 python train.py  batch_size=512 action_repeat=8 env_index=3 seed=4 &
```

```python
CUDA_VISIBLE_DEVICES=20 python train.py  batch_size=512 action_repeat=8 env_index=4 seed=0 &
CUDA_VISIBLE_DEVICES=21 python train.py  batch_size=512 action_repeat=8 env_index=4 seed=1 &
CUDA_VISIBLE_DEVICES=22 python train.py  batch_size=512 action_repeat=8 env_index=4 seed=2 &
CUDA_VISIBLE_DEVICES=23 python train.py  batch_size=512 action_repeat=8 env_index=4 seed=3 &
CUDA_VISIBLE_DEVICES=24 python train.py  batch_size=512 action_repeat=8 env_index=4 seed=4 &
```

```python
CUDA_VISIBLE_DEVICES=25 python train.py  batch_size=512 action_repeat=4 env_index=5 seed=0 &
CUDA_VISIBLE_DEVICES=26 python train.py  batch_size=512 action_repeat=4 env_index=5 seed=1 &
CUDA_VISIBLE_DEVICES=27 python train.py  batch_size=512 action_repeat=4 env_index=5 seed=2 &
CUDA_VISIBLE_DEVICES=28 python train.py  batch_size=512 action_repeat=4 env_index=5 seed=3 &
CUDA_VISIBLE_DEVICES=29 python train.py  batch_size=512 action_repeat=4 env_index=5 seed=4 &
```



## Testing
```python


CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=400 action_repeat=8 seed=0 lr=2e-4

CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=400 action_repeat=8 seed=0 lr=5e-4

CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=400 action_repeat=8 seed=0 lr=8e-4

CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=400 action_repeat=8 seed=0 lr=1e-3

CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=400 action_repeat=8 seed=0 lr=2e-3

CUDA_VISIBLE_DEVICES=0 python train.py env=cartpole_swingup batch_size=400 action_repeat=8 seed=0 lr=5e-3
```

## Instructions
To train the MOREformer in the 6 representative DMControl tasks run on the i-th gpu with seed $i$ just run sh scripts/test[i].sh, for example
```

sh scripts/test0.sh

```
To train the MOREformer on the `Cartpole-Swingup` task run
```

python train.py env=cartpole_swingup

```

To get the state-of-the-art performance run
```

python train.py env=cartpole_swingup batch_size=512 action_repeat=8

```

This will produce the `runs` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. To launch tensorboard run
```

tensorboard --logdir runs

```

#### IMPORTANT: All the dropout operators in vision transformer are removed in this repo since it is not suit for online RL tasks. 

#### IMPORTANT: please use a batch size of 512 to reproduce the results in the paper. Hovewer, with a smaller batch size it still works well.

#### IMPORTANT: And if action_repeat is used the effective number of env steps needs to be multiplied by action_repeat in the result graphs. This is a common practice for a fair comparison. The hyper-parameters of action_repeat for different task is set as follow:
|  Env   | action_repeat  |
|  ----  | ----  |
| cartpole_swingup  | 8 |
| reacher_easy  | 4 |
| cheetah_run  | 4 |
| finger_spin  | 2 |
| walker_walk  | 2 |
| ball_in_cup_catch  | 4 |


The console output is also available in a form:
```

| train | E: 5 | S: 5000 | R: 11.4359 | D: 66.8 s | BR: 0.0581 | ALOSS: -1.0640 | CLOSS: 0.0996 | TLOSS: -23.1683 | TVAL: 0.0945 | AENT: 3.8132

```
a training entry decodes as
```

train - training episode
E - total number of episodes
S - total number of environment steps
R - episode return
D - duration in seconds
BR - average reward of a sampled batch
ALOSS - average loss of the actor
CLOSS - average loss of the critic
TLOSS - average loss of the temperature parameter
TVAL - the value of temperature
AENT - the actor's entropy

```
while an evaluation entry
```

| eval | E: 20 | S: 20000 | R: 10.9356

```
contains
```

E - evaluation was performed after E episodes
S - evaluation was performed after S environment steps
R - average episode return computed over `num_eval_episodes` (usually 10)

```


## Current Performance
![Performance](pngs/Performance.PNG)



## Benchmarks
### The PlaNet Benchmark
**MOREformer** demonstrates the state-of-the-art performance on a set of challenging image-based tasks from the DeepMind Control Suite (Tassa et al., 2018). We compare against PlaNet (Hafner et al., 2018), SAC-AE (Yarats et al., 2019), SLAC (Lee et al., 2019), CURL (Srinivas et al., 2020), and an upper-bound performance SAC States (Haarnoja et al., 2018). This follows the benchmark protocol established in PlaNet (Hafner et al., 2018).
![The PlaNet Benchmark](pngs/planet_bench.png)

### The Dreamer Benchmark
**MOREformer** demonstrates the state-of-the-art performance on an extended set of challenging image-based tasks from the DeepMind Control Suite (Tassa et al., 2018), following the benchmark protocol from Dreamer (Hafner et al., 2019). We compare against Dreamer (Hafner et al., 2019) and an upper-bound performance SAC States (Haarnoja et al., 2018).
![The Dreamer Benchmark](pngs/dreamer_bench.png)


## Acknowledgements
We used [timm](https://github.com/rwightman/pytorch-image-models) for basic model of vision transformer.
We used [kornia](https://github.com/kornia/kornia) for data augmentation.
