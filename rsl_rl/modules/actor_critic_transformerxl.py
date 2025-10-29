# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP, TransformerXL
from rsl_rl.utils import unpad_trajectories


class ActorCriticTransformerXL(nn.Module):
    is_recurrent = False 
    is_transformerxl = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_dependent_std=False,
        transformer_model_dim: int = 256,
        transformer_num_layers: int = 4,
        transformer_num_heads: int = 8,
        transformer_ff_multiplier: float = 4.0,
        transformer_dropout: float = 0.0,
        memory_length: int = 128,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformerXL.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys())),
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.state_dependent_std = state_dependent_std

        # get observation dimensions
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "TransformerXL module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "TransformerXL module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # transformer backbones (separate caches for actor and critic)
        self.obs_project_actor = nn.Linear(num_actor_obs, transformer_model_dim)
        self.obs_project_critic = nn.Linear(num_critic_obs, transformer_model_dim)

        self.transformerxl = TransformerXL(
            transformer_model_dim,
            model_dim=transformer_model_dim,
            depth=transformer_num_layers,
            heads=transformer_num_heads,
            ff_multiplier=transformer_ff_multiplier,
            dropout=transformer_dropout,
            memory_length=memory_length,
        )


        # actor
        if self.state_dependent_std:
            self.actor = MLP(transformer_model_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(transformer_model_dim, num_actions, actor_hidden_dims, activation)

        # actor normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # critic
        self.critic = MLP(transformer_model_dim, 1, critic_hidden_dims, activation)

        # critic normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )

        self.distribution = None
        Normal.set_default_validate_args(False)

        self._actor_state = None
        self._actor_inference_state = None
        self._critic_state = None

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None, hidden_states=None):
        if dones is None:
            self._actor_state = None
            self._actor_inference_state = None
            self._critic_state = None
            return

        dones_bool = dones.to(dtype=torch.bool)

        self._actor_state = self._mask_done_entries(self._actor_state, dones_bool)
        self._actor_inference_state = self._mask_done_entries(self._actor_inference_state, dones_bool)
        self._critic_state = self._mask_done_entries(self._critic_state, dones_bool)


    def forward(self):
        raise NotImplementedError
    
    def _mask_done_entries(self, state, dones):
        if not state or dones is None:
            return state

        cache, abs_pos = state
        if dones.dim() > 1:
            dones = dones.flatten()

        keep_mask = (~dones).view(-1, 1, 1, 1)

        masked_cache = [
            None if layer is None else (k * keep_mask, v * keep_mask)
            for layer in cache
            for k, v in [layer]  # unpack safely inside comprehension
        ]

        return masked_cache, abs_pos


    def _apply_transformer(self, obs, state_attr):
        is_streaming = obs.dim() < 3
        state = getattr(self, state_attr) 

        # Adapt input to the format TransformerXL's layers expect ([batch, seq, dim]),
        # bypassing the flawed shape handling inside TransformerXL.forward.
        if is_streaming:
            # Manually convert [batch, dim] to [batch, 1, dim].
            obs = obs.unsqueeze(1)
        else:
            # Manually convert [seq, batch, dim] to [batch, seq, dim].
            obs = obs.permute(1, 0, 2)
        print(state_attr)
        print(obs.shape)
        features, new_state = self.transformerxl(obs, state=state)

        # Adapt output back to original format
        if is_streaming:
            features = features.squeeze(1)
        else:
            features = features.permute(1, 0, 2)

        if new_state is not None:
            setattr(self, state_attr, new_state)
        return features

    def _update_distribution(self, features):
        if self.state_dependent_std:
            mean_and_std = self.actor(features)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        else:
            mean = self.actor(features)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(
                    f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
                )
        self.distribution = Normal(mean, std)

    def act(self, obs, masks=None, hidden_states=None):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        obs = self.obs_project_actor(obs)
        features = self._apply_transformer(
            obs,
            "_actor_state",
        )
        # if masks is not None:
        #     features = unpad_trajectories(features, masks)
        self._update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        obs = self.obs_project_actor(obs)
        features = self._apply_transformer(
            obs,
            "_actor_inference_state",
        )
        # if masks is not None:
        #     features = unpad_trajectories(features, masks)
        if self.state_dependent_std:
            return self.actor(features)[..., 0, :]
        else:
            return self.actor(features)

    def evaluate(self, obs, masks=None, hidden_states=None):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        obs = self.obs_project_critic(obs)
        features = self._apply_transformer(
            obs,
            "_critic_state",
        )
        # if masks is not None:
        #     features = unpad_trajectories(features, masks)
        return self.critic(features)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return None, None

    def detach_hidden_states(self, dones=None):
        return None, None

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
