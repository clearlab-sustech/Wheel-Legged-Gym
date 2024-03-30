# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCriticSequence(nn.Module):
    is_recurrent = False
    is_sequence = True

    def __init__(
        self,
        num_obs,
        num_critic_obs,
        num_actions,
        num_encoder_obs,
        latent_dim,
        encoder_hidden_dims=[256, 256, 256],
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        orthogonal_init=False,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticVAE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ActorCriticSequence, self).__init__()

        self.orthogonal_init = orthogonal_init
        self.latent_dim = latent_dim

        activation = get_activation(activation)

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_encoder_obs, encoder_hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[l], self.latent_dim)
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(encoder_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
            else:
                encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
                encoder_layers.append(activation)
                # actor_layers.append(torch.nn.LayerNorm(actor_hidden_dims[l + 1]))
        self.encoder = nn.Sequential(*encoder_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs + self.latent_dim, actor_hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
                actor_layers.append(activation)
                # actor_layers.append(torch.nn.LayerNorm(actor_hidden_dims[l + 1]))
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(critic_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(critic_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
                critic_layers.append(activation)
                # critic_layers.append(torch.nn.LayerNorm(critic_hidden_dims[l + 1]))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, observation_history):
        self.latent = self.encoder(observation_history)
        mean = self.actor(torch.cat((observations, self.latent.detach()), dim=-1))
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, observation_history, **kwargs):
        self.update_distribution(observations, observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_latent(self, **kwargs):
        return self.latent

    def act_inference(self, observations, observation_history):
        self.latent = self.encoder(observation_history)
        actions_mean = self.actor(torch.cat((observations, self.latent), dim=-1))
        return actions_mean, self.latent

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def encode(self, observation_history, **kwargs):
        latent = self.encoder(observation_history)
        return latent


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
