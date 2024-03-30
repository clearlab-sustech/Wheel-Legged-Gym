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

from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import (
    WheelLeggedCfg,
    WheelLeggedCfgPPO,
)


class WheelLeggedVMCCfg(WheelLeggedCfg):
    class env(WheelLeggedCfg.env):
        num_privileged_obs = (
            WheelLeggedCfg.env.num_observations + 7 * 11 + 3 + 6 * 7 + 3 + 3
        )

    class control(WheelLeggedCfg.control):
        action_scale_theta = 0.5
        action_scale_l0 = 0.1
        action_scale_vel = 10.0

        l0_offset = 0.175
        feedforward_force = 40.0  # [N]

        kp_theta = 50.0  # [N*m/rad]
        kd_theta = 3.0  # [N*m*s/rad]
        kp_l0 = 900.0  # [N/m]
        kd_l0 = 20.0  # [N*s/m]

        # PD Drive parameters:
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}  # [N*m/rad]
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.5}  # [N*m*s/rad]

    class normalization(WheelLeggedCfg.normalization):
        class obs_scales(WheelLeggedCfg.normalization.obs_scales):
            l0 = 5.0
            l0_dot = 0.25

    class noise(WheelLeggedCfg.noise):
        class noise_scales(WheelLeggedCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1


class WheelLeggedVMCCfgPPO(WheelLeggedCfgPPO):

    class algorithm(WheelLeggedCfgPPO.algorithm):
        kl_decay = (
            WheelLeggedCfgPPO.algorithm.desired_kl - 0.002
        ) / WheelLeggedCfgPPO.runner.max_iterations

    class runner(WheelLeggedCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged_vmc"
