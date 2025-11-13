# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization, TransformerXL

_default = object()


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
        transformer_pos_bias_max: float = 2.0,
        transformer_norm_type: str = "rms",
        **kwargs,
    ):
        # print(memory_length)
        # print("-"*60)
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
        # Project raw observations to transformer model dimension using a
        # single hidden layer (2x model dim).
        self.obs_project_actor = MLP(num_actor_obs, transformer_model_dim, [transformer_model_dim * 2], activation)
        self.obs_project_critic = MLP(num_critic_obs, transformer_model_dim, [transformer_model_dim * 2], activation)

        self.transformerxl = TransformerXL(
            transformer_model_dim,
            model_dim=transformer_model_dim,
            depth=transformer_num_layers,
            heads=transformer_num_heads,
            ff_multiplier=transformer_ff_multiplier,
            dropout=transformer_dropout,
            memory_length=memory_length,
            pos_bias_max=transformer_pos_bias_max,
            norm_type=transformer_norm_type,
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
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        self.rollout_actor_cache = None
        self.inference_actor_cache = None
        self.rollout_critic_cache = None

        self.training_actor_cache = None
        self.training_critic_cache = None

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
        pass
        # dones_bool = dones.to(dtype=torch.bool)

        # self.rollout_actor_cache = self._mask_done_entries(self.rollout_actor_cache, dones_bool)
        # self.inference_actor_cache = self._mask_done_entries(self.inference_actor_cache, dones_bool)
        # self.rollout_critic_cache = self._mask_done_entries(self.rollout_critic_cache, dones_bool)
        # self.training_actor_cache = self._mask_done_entries(self.training_actor_cache, dones_bool)
        # self.training_critic_cache = self._mask_done_entries(self.training_critic_cache, dones_bool)

    def forward(self):
        raise NotImplementedError

    def _mask_done_entries(self, state, dones):
        if state is None or dones is None:
            return state

        cache = state
        if dones.dim() > 1:
            dones = dones.flatten()

        keep_mask = (~dones).view(-1, 1, 1, 1)

        masked_cache = []
        for layer in cache:
            if layer is None:
                masked_cache.append(None)
            else:
                k, v = layer
                masked_k = k * keep_mask if k is not None else None
                masked_v = v * keep_mask if v is not None else None
                masked_cache.append((masked_k, masked_v))
        return masked_cache

    def _apply_transformer(self, obs, state):
        """Run the Transformer-XL backbone while keeping tensor layouts explicit.

        - Streaming mode (collecting data): obs is [num_envs, feature_dim].
        - Training mode: obs is [time, num_envs, feature_dim].

        Transformer-XL expects [batch, seq, dim], i.e. [num_envs, time, dim].
        """
        if obs.dim() == 2:
            # [B, D] -> [B, 1, D] so TXL treats the step as a length-1 sequence.
            transformer_input = obs.unsqueeze(1)
            squeeze_output = True
        elif obs.dim() == 3:
            # [T, B, D] -> [B, T, D] so attention runs along the time axis.
            transformer_input = obs.permute(1, 0, 2).contiguous()
            squeeze_output = False
        else:
            raise ValueError(f"Unsupported observation rank {obs.dim()}; expected 2 or 3.")

        features, new_state = self.transformerxl(transformer_input, state=state)
        # print(transformer_input.shape)
        if squeeze_output:
            # [B, 1, D] -> [B, D]
            features = features.squeeze(1)
        else:
            # [B, T, D] -> [T, B, D] to match the caller's expected layout.
            features = features.permute(1, 0, 2).contiguous()
        # self.print_state(state=new_state)
        return features, new_state

    def _update_distribution(self, features):
        if self.state_dependent_std:
            mean_and_std = self.actor(features)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(features)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, masks=None, hidden_states=_default):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        obs = self.obs_project_actor(obs)
        features, state = self._apply_transformer(
            obs,
            self.rollout_actor_cache if hidden_states is _default else self.training_actor_cache,
        )
        if hidden_states is _default:
            self.rollout_actor_cache = state
        else:
            self.training_actor_cache = state

        self._update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        obs = self.obs_project_actor(obs)
        features, self.inference_actor_cache = self._apply_transformer(
            obs,
            self.inference_actor_cache,
        )

        if self.state_dependent_std:
            return self.actor(features)[..., 0, :]
        else:
            return self.actor(features)

    def evaluate(self, obs, masks=None, hidden_states=_default):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        obs = self.obs_project_critic(obs)
        features, state = self._apply_transformer(
            obs,
            self.rollout_critic_cache if hidden_states is _default else self.training_critic_cache,
        )
        if masks == "compute_returns":
            pass
        elif hidden_states is _default:
            self.rollout_critic_cache = state
        else:
            self.training_critic_cache = state

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

    def get_transformerxl_state(self):
        return self.training_actor_cache, self.training_critic_cache

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

    def print_state(self, state):
        cache = state
        if cache and cache[0] is not None:
            k, v = cache[0]
            print(f"Key shape: {k.shape}")
            print("=" * 20)
