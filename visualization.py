import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np

import dmc2gym
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


from vit_grad_rollout import VITAttentionGradRollout

# model = torch.hub.load('facebookresearch/deit:main', 
# 'deit_tiny_patch16_224', pretrained=True)
# mask = grad_rollout(input_tensor)
# grad_rollout = VITAttentionGradRollout(model, discard_ratio=0.9, head_fusion='max')



def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=cfg.image_size,
                       width=cfg.image_size,
                       frame_skip=cfg.action_repeat,
                       camera_id=camera_id)

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def visual(self):
        if self.cfg.load_pretrain:
            load_path="../../../models"
            print("start load")
            self.agent.actor.load_state_dict(torch.load(load_path+"/actor.pth"))
            self.agent.critic.load_state_dict(torch.load(load_path+"/critic.pth"))
            self.agent.critic_target.load_state_dict(torch.load(load_path+"/critic_target.pth"))
            print("load pretrained model done")
        obs = self.env.reset()
        obs_tensor = torch.as_tensor(obs).float().cuda()
        obs_tensor = torch.unsqueeze(obs_tensor,0)
        vis_model = self.agent.critic.encoder
        grad_rollout = VITAttentionGradRollout(vis_model, discard_ratio=0.9)
        grad_rollout(obs_tensor)
        #print(mask.shape)
        
@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.visual()


if __name__ == '__main__':
    main()
