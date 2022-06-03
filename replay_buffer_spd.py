import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import augmentation

aug_to_func = {    
        'crop': augmentation.Crop,
        'random-conv': augmentation.RandomConv,
        'grayscale': augmentation.Grayscale,
        'cutout-color': augmentation.CutoutColor,
        'color-jitter': augmentation.ColorJitter,
}

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, batch_size, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_list = [aug_to_func[t](batch_size=batch_size) for t in list(aug_to_func.keys())]
        
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        random_aug_idx = np.random.randint(1, len(self.aug_list), size=1)
        
        self.aug_trans = self.aug_list[random_aug_idx[0]]
        
        ###
        self.aug_trans_randcrop = self.aug_list[0]

        clean_obses = self.obses[idxs]
        clean_next_obses = self.next_obses[idxs]
        
        
        aug_obses = clean_obses.copy()
        aug_next_obses = clean_next_obses.copy()
        

        clean_obses = torch.as_tensor(clean_obses, device=self.device).float()
        clean_next_obses = torch.as_tensor(clean_next_obses, device=self.device).float()

        aug_obses = torch.as_tensor(aug_obses, device=self.device).float()
        aug_next_obses = torch.as_tensor(aug_next_obses, device=self.device).float()

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        #####
        clean_obses = self.aug_trans_randcrop.do_augmentation(clean_obses)
        clean_next_obses = self.aug_trans_randcrop.do_augmentation(clean_next_obses)
        aug_obses = self.aug_trans_randcrop.do_augmentation(aug_obses)
        aug_next_obses = self.aug_trans_randcrop.do_augmentation(aug_next_obses)
        #####
        clean_obses, clean_next_obses, aug_obses, aug_next_obses = clean_obses/ 255., clean_next_obses/255., aug_obses/255., aug_next_obses/255.
        aug_obses = self.aug_trans.do_augmentation(aug_obses)
        aug_next_obses = self.aug_trans.do_augmentation(aug_next_obses)
        ###
        

        return clean_obses, actions, rewards, clean_next_obses, not_dones_no_max, aug_obses, aug_next_obses