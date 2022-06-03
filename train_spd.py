import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
import dmc2gym
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer_spd import ReplayBuffer
# from video import VideoRecorder

torch.backends.cudnn.benchmark = True

def make_env(cfg):
    """Helper function to create dm_control environment"""
    domain_name = cfg.env.split('_')[0]
    task_name = '_'.join(cfg.env.split('_')[1:])
    print("domain_name :", domain_name)
    print("task_name :", task_name)
    camera_id = 2 if domain_name == 'quadruped' else 0
    
    env = dmc2gym.make(
       domain_name=domain_name,
       task_name=task_name,
       resource_files=cfg.train_resource_files,
       img_source=cfg.train_img_source,
       total_frames=cfg.total_frames,
       seed=cfg.seed,
       visualize_reward=False,
       from_pixels=True,
       height=cfg.image_size,
       width=cfg.image_size,
       frame_skip=cfg.action_repeat
    )

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def make_eval_1_env(cfg):
    """Helper function to create dm_control environment"""
    domain_name = cfg.env.split('_')[0]
    task_name = '_'.join(cfg.env.split('_')[1:])
    
    eval_env = dmc2gym.make(
       domain_name=domain_name,
       task_name=task_name,
       resource_files=cfg.eval_resource_1_files,
       img_source=cfg.eval_img_1_source,
       total_frames=cfg.total_frames,
       seed=2,
       visualize_reward=False,
       from_pixels=True,
       height=cfg.image_size,
       width=cfg.image_size,
       frame_skip=cfg.action_repeat
    )
    eval_env = utils.FrameStack(eval_env, k=cfg.frame_stack)

    return eval_env

def make_eval_2_env(cfg):
    """Helper function to create dm_control environment"""
    domain_name = cfg.env.split('_')[0]
    task_name = '_'.join(cfg.env.split('_')[1:])
    
    eval2_env = dmc2gym.make(
       domain_name=domain_name,
       task_name=task_name,
       resource_files=cfg.eval_resource_2_files,
       img_source=cfg.eval_img_2_source,
       total_frames=cfg.total_frames,
       seed=1,
       visualize_reward=False,
       from_pixels=True,
       height=cfg.image_size,
       width=cfg.image_size,
       frame_skip=cfg.action_repeat
    )
    eval2_env = utils.FrameStack(eval2_env, k=cfg.frame_stack)

    return eval2_env

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat,
                             )
                             
        self.model_dir = self.work_dir + '/models'                             
        os.makedirs(self.model_dir, exist_ok=True)
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)
        self.eval_1_env = make_eval_1_env(cfg)
        self.eval_2_env = make_eval_2_env(cfg)
        
        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        # adv
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.batch_size,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)
        # self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.eval_1_env.reset()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.eval_1_env.step(action)
                # self.video_recorder.record(self.eval_env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward_de', average_episode_reward, self.step)

        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.eval_2_env.reset()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.eval_2_env.step(action)
                # self.video_recorder.record(self.eval_env)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward_ge', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    if self.step % 100000 == 0:
                        self.agent.save(self.model_dir, self.step)

                self.logger.log('train/episode_reward', episode_reward, self.step)

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
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger, self.step)
            next_obs, reward, done, info = self.env.step(action)
            # allow infinite bootstrap      
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config_spd.yaml', strict=True)
def main(cfg):
    from train_spd import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
