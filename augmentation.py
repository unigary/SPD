import numpy as np
import torch
import torch.nn as nn
import numbers
import random
import time
import kornia
from kornia.color.hsv import hsv_to_rgb, rgb_to_hsv


def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3
    
    imgs = imgs.view([b,frames,3,h,w])
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + imgs[:, :, 2, ...] * 0.114 
    
    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float().to(device) # broadcast tiling
    return imgs

class Grayscale(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def do_augmentation(self, images, p=.3):
        device = images.device
        in_type = images.type()
        ####
        images = images * 255.
        images = images.type(torch.uint8)
        # images: [B, C, H, W]
        bs, channels, h, w = images.shape
        images = images.to(device)
        gray_images = grayscale(images)
        rnd = np.random.uniform(0., 1., size=(images.shape[0],))
        mask = rnd <= p
        mask = torch.from_numpy(mask)
        frames = images.shape[1] // 3
        images = images.view(*gray_images.shape)
        mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
        mask = mask.type(images.dtype).to(device)
        mask = mask[:, :, None, None, None]
        out = mask * gray_images + (1 - mask) * images
        ####
        out = out.view([bs, -1, h, w]).type(in_type) / 255.
        return out

class Cutout(object):
    """
    Cutout Augmentation
    """
    def __init__(self, batch_size, min_cut=10, max_cut=30):
        self.batch_size = batch_size
        self.min_cut = min_cut
        self.max_cut = max_cut
        
    def do_augmentation(self, imgs):
        device = imgs.device
        n, c, h, w = imgs.shape
        imgs = imgs.cpu().numpy()
        w1 = np.random.randint(self.min_cut, self.max_cut, n)
        h1 = np.random.randint(self.min_cut, self.max_cut, n)
        
        cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
        for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
            cut_img = img.copy()
            cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
            cutouts[i] = cut_img
        cutouts = torch.as_tensor(cutouts, device=device)
        return cutouts
        
       
    def change_randomization_params(self, index_):
        pass

    def change_randomization_params_all(self):
        pass
        
    def print_parms(self):
        pass
        
        
class CutoutColor(object):
    """
    Cutout-Color Augmentation
    """
    def __init__(self, batch_size, box_min=10, box_max=30, pivot_h=12, pivot_w=24, obs_dtype='uint8', *_args, **_kwargs):
        self.box_min = box_min
        self.box_max = box_max
        self.batch_size = batch_size
        
    def do_augmentation(self, imgs):
        device = imgs.device
        imgs = imgs.cpu().numpy()
        n, c, h, w = imgs.shape
        w1 = np.random.randint(self.box_min, self.box_max, n)
        h1 = np.random.randint(self.box_min, self.box_max, n)
        
        cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
        ####
        rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
        # rand_box = np.random.randint(0, 255, size=(n, c))
        for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
            cut_img = img.copy()
            # add random box
            cut_img[:, h11:h11 + h11, w11:w11 + w11] = np.tile(
                rand_box[i].reshape(-1,1,1),                                                
                (1,) + cut_img[:, h11:h11 + h11, w11:w11 + w11].shape[1:])     
            cutouts[i] = cut_img
    
        cutouts = torch.as_tensor(cutouts, device=device)
        return cutouts
        
    def change_randomization_params(self, index_):
        pass

    def change_randomization_params_all(self):
        pass
        
    def print_parms(self):
        pass
     
class multiCutoutColor(object):
    """
    Cutout-Color Augmentation
    """
    def __init__(self, batch_size, box_min=5, box_max=10, pivot_h=12, pivot_w=24, obs_dtype='uint8', *_args, **_kwargs):
        self.box_min = box_min
        self.box_max = box_max
        self.batch_size = batch_size
        
    def do_augmentation(self, imgs):
        device = imgs.device
        imgs = imgs.cpu().numpy()
        n, c, h, w = imgs.shape
        box_num = 10
        
        pivot_w_ = np.random.randint(0, 74, (n, box_num))
        pivot_h_ = np.random.randint(0, 30, (n, box_num))
        w1 = np.random.randint(self.box_min, self.box_max, (n,box_num))
        h1 = np.random.randint(self.box_min, self.box_max, (n,box_num))
        cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
        rand_box = np.random.randint(0, 255, size=(n, box_num, c)) / 255.

        for i, (img, w11, h11, pivot_w, pivot_h) in enumerate(zip(imgs, w1, h1, pivot_w_, pivot_h_)):
            cut_img = img.copy()
            cut_img1 = img.copy()
            # add random box
            for j in range(box_num):
                cut_img[:, pivot_h[j]+h11[j]:pivot_h[j]+h11[j] + h11[j], pivot_w[j]+w11[j]:pivot_w[j]+w11[j] + w11[j]] = np.tile(
                    rand_box[i][j].reshape(-1,1,1),                                                
                    (1,) + cut_img[:, pivot_h[j]+h11[j]:pivot_h[j]+h11[j] + h11[j], pivot_w[j]+w11[j]:pivot_w[j]+w11[j] + w11[j]].shape[1:])     
            cutouts[i] = 0.7 * cut_img1 + 0.3 * cut_img
    
        cutouts = torch.as_tensor(cutouts, device=device)
        return cutouts
        
    def change_randomization_params(self, index_):
        pass

    def change_randomization_params_all(self):
        pass
        
    def print_parms(self):
        pass


class Flip(object):
    """
    Flip Augmentation
    """
    def __init__(self, batch_size, p=0.2, *_args, **_kwargs):
        self.batch_size = batch_size
        self.p = p
        
    def do_augmentation(self, images):
        device = images.device
        bs, channels, h, w = images.shape
        
        images = images.to(device)

        flipped_images = images.flip([3])
        
        rnd = np.random.uniform(0., 1., size=(images.shape[0],))
        mask = rnd <= self.p
        mask = torch.from_numpy(mask)
        frames = images.shape[1] #// 3
        images = images.view(*flipped_images.shape)
        mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
        
        mask = mask.type(images.dtype).to(device)
        mask = mask[:, :, None, None]
        
        out = mask * flipped_images + (1 - mask) * images

        out = out.view([bs, -1, h, w])
        return out
    
    def change_randomization_params(self, index_):
        pass

    def change_randomization_params_all(self):
        pass
        
    def print_parms(self):
        pass
        

class Rotate(object):
    """
    Rotate Augmentation
    """
    def __init__(self, batch_size, *_args, **_kwargs):
        self.batch_size = batch_size
    
    def change_randomization_params(self, index_):
        pass
        
    def change_randomization_params_all(self):
        pass
        
    def print_parms(self):
        pass
    
    def do_augmentation(self, images, p=.3):
        device = images.device
        # images: [B, C, H, W]
        bs, channels, h, w = images.shape
        
        images = images.to(device)

        rot90_images = images.rot90(1,[2,3])
        rot180_images = images.rot90(2,[2,3])
        rot270_images = images.rot90(3,[2,3])    
        
        rnd = np.random.uniform(0., 1., size=(images.shape[0],))
        rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
        mask = rnd <= p
        mask = rnd_rot * mask
        mask = torch.from_numpy(mask).to(device)
        
        frames = images.shape[1]
        masks = [torch.zeros_like(mask) for _ in range(4)]
        for i,m in enumerate(masks):
            m[torch.where(mask==i)] = 1
            m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(device)
            m = m[:,:,None,None]
            masks[i] = m
        
        
        out = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

        out = out.view([bs, -1, h, w])
        return out
        

class Crop(object):
    """
    Crop Augmentation
    """
    def __init__(self, batch_size, *_args, **_kwargs):
        self.batch_size = batch_size 

    def do_augmentation(self, x):
        aug_trans = nn.Sequential(nn.ReplicationPad2d(4),kornia.augmentation.RandomCrop((84, 84)))
        return aug_trans(x)

    def change_randomization_params(self, index_):
        pass

    def change_randomization_params_all(self):
        pass

    def print_parms(self):
        pass


class RandomConv(object):
    """
    Random-Conv Augmentation
    """
    def __init__(self, batch_size, *_args, **_kwargs):
        self.batch_size = batch_size 
    
    def do_augmentation(self, x):
        _device = x.device
        img_h, img_w = x.shape[2], x.shape[3]
        num_stack_channel = x.shape[1]
        num_batch = x.shape[0]
        num_trans = num_batch
        batch_size = int(num_batch / num_trans)
        
        # initialize random covolution
        with torch.no_grad():
            rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
            
            for trans_index in range(num_trans):
                torch.nn.init.xavier_normal_(rand_conv.weight.data)
                temp_x = x[trans_index*batch_size:(trans_index+1)*batch_size]
                temp_x = temp_x.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
                rand_out = rand_conv(temp_x)
                if trans_index == 0:
                    total_out = rand_out
                else:
                    total_out = torch.cat((total_out, rand_out), 0)
            total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
        return total_out

    def change_randomization_params(self, index_):
        pass

    def change_randomization_params_all(self):
        pass

    def print_parms(self):
        pass


class ColorJitter(nn.Module):
    def __init__(self, batch_size,
                 brightness=(0.6, 1.4),
                 contrast=(0.6, 1.4),
                 saturation=(0.6, 1.4),
                 hue=(-0.5, 0.5)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.batch_size = batch_size

    def do_augmentation(self, x):
        """ Returns jittered images.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        # check if channel can be devided by three
        if x.shape[1] % 3 > 0:
            raise ValueError('color jitter is used with stacked RGB images')

        # flag for transformation order
        is_transforming_rgb_first = np.random.randint(2)

        # (batch, C, W, H) -> (batch, stack, 3, W, H)
        flat_rgb = x.view(x.shape[0], -1, 3, x.shape[2], x.shape[3])

        if is_transforming_rgb_first:
            # transform contrast
            flag_rgb = self._transform_contrast(flat_rgb)

        # (batch, stack, 3, W, H) -> (batch * stack, 3, W, H)
        rgb_images = flat_rgb.view(-1, 3, x.shape[2], x.shape[3])

        # RGB -> HSV
        hsv_images = rgb_to_hsv(rgb_images)

        # apply same transformation within the stacked images
        # (batch * stack, 3, W, H) -> (batch, stack, 3, W, H)
        flat_hsv = hsv_images.view(x.shape[0], -1, 3, x.shape[2], x.shape[3])

        # transform hue
        flat_hsv = self._transform_hue(flat_hsv)
        # transform saturate
        flat_hsv = self._transform_saturate(flat_hsv)
        # transform brightness
        flat_hsv = self._transform_brightness(flat_hsv)

        # (batch, stack, 3, W, H) -> (batch * stack, 3, W, H)
        hsv_images = flat_hsv.view(-1, 3, x.shape[2], x.shape[3])

        # HSV -> RGB
        rgb_images = hsv_to_rgb(hsv_images)

        # (batch * stack, 3, W, H) -> (batch, stack, 3, W, H)
        flat_rgb = rgb_images.view(x.shape[0], -1, 3, x.shape[2], x.shape[3])

        if not is_transforming_rgb_first:
            # transform contrast
            flat_rgb = self._transform_contrast(flat_rgb)

        return flat_rgb.view(*x.shape)


    def _transform_hue(self, hsv):
        scale = torch.empty(hsv.shape[0], 1, 1, 1, device=hsv.device)
        scale = scale.uniform_(*self.hue) * 255.0 / 360.0
        hsv[:, :, 0, :, :] = (hsv[:, :, 0, :, :] + scale) % 1
        return hsv

    def _transform_saturate(self, hsv):
        scale = torch.empty(hsv.shape[0], 1, 1, 1, device=hsv.device)
        scale.uniform_(*self.saturation)
        hsv[:, :, 1, :, :] *= scale
        return hsv.clamp(0, 1)

    def _transform_brightness(self, hsv):
        scale = torch.empty(hsv.shape[0], 1, 1, 1, device=hsv.device)
        scale.uniform_(*self.brightness)
        hsv[:, :, 2, :, :] *= scale
        return hsv.clamp(0, 1)

    def _transform_contrast(self, rgb):
        scale = torch.empty(rgb.shape[0], 1, 1, 1, 1, device=rgb.device)
        scale.uniform_(*self.contrast)
        means = rgb.mean(dim=(3, 4), keepdims=True)
        return ((rgb - means) * (scale + means)).clamp(0, 1)
