import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import hydra
import random

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs):
        if obs.max() > 1.:
            obs = obs / 255.
        
        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.head(h)
        if not self.output_logits:
            out = torch.tanh(out)

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        pass
    

class InverseForwardDynamicsModel(nn.Module):
    def __init__(self, encoder_cfg, feature_dim, action_shape, hidden_dim):
        super().__init__()
        self.encoder = hydra.utils.instantiate(encoder_cfg)
        
        self.fc_inverse = nn.Linear(2*feature_dim, hidden_dim)
        self.ln_inverse = nn.LayerNorm(hidden_dim)
        self.head_inverse = nn.Linear(hidden_dim, action_shape[0])

        self.fc_forward = nn.Linear(action_shape[0]+feature_dim, hidden_dim)
        self.ln_forward = nn.LayerNorm(hidden_dim)
        self.head_forward = nn.Linear(hidden_dim, feature_dim)

        self.apply(utils.weight_init)
    
    def forward(self, h_clean, h_next_clean, h_aug, h_next_aug):        
        joint_h_g = torch.cat([h_aug, h_next_aug], dim=1)
        joint_h_c = torch.cat([h_clean, h_next_clean], dim=1)

        pred_action_g = torch.relu(self.ln_inverse(self.fc_inverse(joint_h_g)))
        pred_action_g = torch.tanh(self.head_inverse(pred_action_g))

        pred_action_c = torch.relu(self.ln_inverse(self.fc_inverse(joint_h_c)))
        pred_action_c = torch.tanh(self.head_inverse(pred_action_c))
        
        joint_s_a_g = torch.cat([h_aug, pred_action_c], dim=1)
        joint_s_a_c = torch.cat([h_clean, pred_action_g], dim=1)

        pred_next_state_g = torch.relu(self.ln_forward(self.fc_forward(joint_s_a_g)))
        pred_next_state_g = torch.tanh(self.head_forward(pred_next_state_g))

        pred_next_state_c = torch.relu(self.ln_forward(self.fc_forward(joint_s_a_c)))
        pred_next_state_c = torch.tanh(self.head_forward(pred_next_state_c))

        return pred_action_g, pred_action_c, pred_next_state_g, pred_next_state_c


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth, log_std_bounds): # obs_shape, image_pad
        super().__init__()
        self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim, 2 * action_shape[0], hidden_depth)

        self.outputs = dict()        
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        pass
        

class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth):
        super().__init__()
        self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)
        
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, aug_obs, action, detach_encoder=False):
        assert aug_obs.size(0) == action.size(0)
        
        aug_obs = self.encoder(aug_obs, detach=detach_encoder)

        obs_action = torch.cat([aug_obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2
        
    def log(self, logger, step):
        pass


class Discriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

        self.apply(utils.weight_init)

    def forward(self, obs):
        D_critic = torch.relu(self.ln(self.fc(obs)))
        D_critic = torch.tanh(self.head(D_critic))

        return D_critic


class SPDAgent(object):
    def __init__(self, obs_shape, action_shape, action_range, device, encoder_cfg, discriminator_cfg,
                 critic_cfg, actor_cfg, inv_cfg, discount, init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        self.discriminator = hydra.utils.instantiate(discriminator_cfg).to(self.device)

        self.inv = hydra.utils.instantiate(inv_cfg).to(self.device)
        self.inv.encoder.copy_conv_weights_from(self.critic.encoder)
        self.encoder.copy_conv_weights_from(self.critic.encoder)
        
        self.inv_optimizer = torch.optim.Adam(self.inv.parameters(), lr=lr)
        
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)
        self.discriminator.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, aug_obs, action, reward, aug_next_obs, not_done, logger, step):
        with torch.no_grad():
            dist_aug = self.actor(aug_next_obs)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(aug_next_obs, next_action_aug)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q = reward + (not_done * self.discount * target_V)
        
        current_Q1, current_Q2 = self.critic(aug_obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, aug_obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(aug_obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(aug_obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        
        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_adv(self, weak_obs, strong_obs):
        self.encoder_optimizer.zero_grad()
        weak_feature_g = self.encoder(weak_obs)
        strong_feature_g = self.encoder(strong_obs)
        weak_imgs_critic_g = self.discriminator(weak_feature_g).detach()
        strong_imgs_critic_g = self.discriminator(strong_feature_g)
        generator_loss = - torch.mean(torch.log(torch.sigmoid(-weak_imgs_critic_g + strong_imgs_critic_g)))
        (0.001*generator_loss).backward()
        self.encoder_optimizer.step()
    
        self.discriminator_optimizer.zero_grad()
        weak_feature_d = self.encoder(weak_obs)
        strong_feature_d = self.encoder(strong_obs)
        weak_imgs_critic_d = self.discriminator(weak_feature_d)
        strong_imgs_critic_d = self.discriminator(strong_feature_d.detach())

        discriminator_loss = - torch.mean(torch.log(torch.sigmoid(weak_imgs_critic_d - strong_imgs_critic_d)))
        (0.001*discriminator_loss).backward()
        self.discriminator_optimizer.step()

    def update_inv(self, weak_obs, weak_next_obs, strong_obs, strong_next_obs, action):
        h_weak, h_next_weak, h_strong, h_next_strong = self.encoder(weak_obs), self.encoder(weak_next_obs), self.encoder(strong_obs), self.encoder(strong_next_obs)
        pred_action_g, pred_action_c, pred_next_state_g, pred_next_state_c = self.inv(h_weak, h_next_strong, h_strong, h_next_weak)
        
        inv_loss = 0.5 * (F.mse_loss(pred_action_g, action.detach()) + F.mse_loss(pred_action_c, action.detach()))
        forward_loss = - 0.5 * (F.cosine_similarity(pred_next_state_g, h_next_strong.detach(), dim=-1).mean() + F.cosine_similarity(pred_next_state_c, h_next_weak.detach(), dim=-1).mean())
        total_loss = inv_loss + forward_loss
        self.inv_optimizer.zero_grad()
        (0.1*total_loss).backward()
        self.inv_optimizer.step()

    def update(self, replay_buffer, logger, step):
        weak_obs, action, reward, weak_next_obs, not_done, strong_obs, strong_next_obs = replay_buffer.sample(self.batch_size)
        self.update_adv(weak_obs, strong_obs)
        self.update_inv(weak_obs, weak_next_obs, strong_obs, strong_next_obs, action)
        self.update_critic(weak_obs, action, reward, weak_next_obs, not_done, logger, step)
        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(weak_obs, logger, step)
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.encoder.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.encoder.load_state_dict(
            torch.load('%s/encoder_%s.pt' % (model_dir, step))
        )