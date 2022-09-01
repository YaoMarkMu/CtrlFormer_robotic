import copy
import math
import os
import pickle as pkl
import sys
import time
import copy
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

        self.logger = Logger(self.work_dir+",env="+cfg.env,
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
            self.work_dir+",env="+cfg.env if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)
    
    def update_optimizer(self, scale=0.01):
        self.agent.update_optimizer(scale)
    
    def reuse_head(self, reuse_model):
        for reuse, new in zip(reuse_model.actor.named_parameters(), self.agent.actor.named_parameters()):
            reuse_name=reuse[0]
            reuse_param=reuse[1]
            new_name = new[0]
            new_param = new[1]
            if "trunk" in reuse_name:
                print(reuse_name)
                new_param.data.copy_(reuse_param.data)

        for reuse, new in zip(reuse_model.critic.named_parameters(), self.agent.critic.named_parameters()):
            reuse_name=reuse[0]
            reuse_param=reuse[1]
            new_name=new[0]
            new_param=new[1]
            if "Q1" in reuse_name or "Q2" in reuse_name:
                print(reuse_name)
                new_param.data.copy_(reuse_param.data)
        for reuse, new in zip(reuse_model.critic.named_parameters(), self.agent.critic_target.named_parameters()):
            reuse_name=reuse[0]
            reuse_param=reuse[1]
            new_name=new[0]
            new_param=new[1]
            if "Q1" in reuse_name or "Q2" in reuse_name:
                print(reuse_name)
                new_param.data.copy_(reuse_param.data)
                
    def run(self,reuse,round,reuse_model): 
        print(self.cfg.scale)
        if reuse:
            self.agent.log_alpha=torch.tensor(np.log(0.0001)).to("cuda")
            self.agent.log_alpha.requires_grad = True
            self.update_optimizer(scale=self.cfg.scale)
            self.agent.set_reuse()
            if round==1:
                self.reuse_head(reuse_model) 
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step*self.cfg.action_repeat <= self.cfg.num_train_steps*round:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            #print(self.cfg.load_pretrain)
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)
            if self.step*self.cfg.action_repeat%20000==0:
                print("save the {}-th model".format(self.step))
                self.agent.save_model(self.work_dir+",env="+self.cfg.env,self.step)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    multi_round=False
    from train import Workspace as W
    cfg1=copy.deepcopy(cfg)
    cfg2=copy.deepcopy(cfg)
    cfg1.env = cfg1.envs[cfg1.env_index][0]
    cfg1.token_index=0 
    workspace_0 = W(cfg1)
    cfg2.env = cfg2.envs[cfg2.env_index][1]
    cfg2.token_index=1
    workspace_1 = W(cfg2)
    reuse=False
    reuse_model=None
    ### you can get a more powerful model via multi_round training
    if multi_round:
        for i in range(10):
            print("################## current is task 0 ###################")
            workspace_0.run(reuse,round=i+1,reuse_model=reuse_model)
            reuse=True
            reuse_model=workspace_0.agent
            print("################## current is task 1 ###################")
            workspace_1.run(reuse,round=i+1,reuse_model=reuse_model)
    else:
        print("################## current is task 0 ###################")
        workspace_0.run(reuse,round=1,reuse_model=reuse_model)
        reuse=True
        reuse_model=workspace_0.agent
        print("################## current is task 1 ###################")
        workspace_1.run(reuse,round=1,reuse_model=reuse_model)

    


    
    
    


if __name__ == '__main__':
    main()
