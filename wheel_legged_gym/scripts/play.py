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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 20
    env_cfg.env.fail_to_terminal_time_s = 3
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.friction_range = [0.1, 0.2]
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)

    logger = Logger(env.dt)
    robot_index = 21  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    latent = None

    CoM_offset_compensate = True
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)

    for i in range(1000 * int(env.max_episode_length)):
        if ppo_runner.alg.actor_critic.is_sequence:
            actions, latent = policy(obs, obs_history)
        else:
            actions = policy(obs.detach())

        env.commands[:, 0] = 2.5
        env.commands[:, 2] = 0.18  # + 0.07 * np.sin(i * 0.01)
        env.commands[:, 3] = 0

        if CoM_offset_compensate:
            if i > 200 and i < 600:
                vel_cmd[:] = 2.5 * np.clip((i - 200) * 0.05, 0, 1)
            else:
                vel_cmd[:] = 0
            vel_err_intergral += (
                (vel_cmd - env.base_lin_vel[:, 0])
                * env.dt
                * ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5)
            )
            vel_err_intergral = torch.clip(vel_err_intergral, -0.5, 0.5)
            env.commands[:, 0] = vel_cmd + vel_err_intergral

        obs, _, rews, dones, infos, obs_history = env.step(actions)
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    WHEEL_LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale
                    + env.default_dof_pos[robot_index, joint_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_yaw": env.commands[robot_index, 1].item(),
                    "command_height": env.commands[robot_index, 2].item(),
                    "base_height": env.base_height[robot_index].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            if CoM_offset_compensate:
                logger.log_states({"command_x": vel_cmd[robot_index].item()})
            else:
                logger.log_states({"command_x": env.commands[robot_index, 0].item()})
            if latent is not None:
                logger.log_states(
                    {
                        "est_lin_vel_x": latent[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": latent[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": latent[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
                if latent.shape[1] > 3 and env_cfg.noise.add_noise:
                    logger.log_states(
                        {
                            "base_vel_yaw_obs": obs[robot_index, 2].item()
                            / env.cfg.normalization.obs_scales.ang_vel,
                            "dof_pos_obs": obs[robot_index, 9 + joint_index].item()
                            / env.cfg.normalization.obs_scales.dof_pos
                            + env.default_dof_pos[robot_index, joint_index].item(),
                            "dof_vel_obs": obs[robot_index, 15 + joint_index].item()
                            / env.cfg.normalization.obs_scales.dof_vel,
                        }
                    )
                    logger.log_states(
                        {
                            "base_vel_yaw_est": latent[robot_index, 3 + 2].item()
                            / env.cfg.normalization.obs_scales.ang_vel,
                            "dof_pos_est": latent[
                                robot_index, 3 + 9 + joint_index
                            ].item()
                            / env.cfg.normalization.obs_scales.dof_pos
                            + env.default_dof_pos[robot_index, joint_index].item(),
                            "dof_vel_est": latent[
                                robot_index, 3 + 15 + joint_index
                            ].item()
                            / env.cfg.normalization.obs_scales.dof_vel,
                        }
                    )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
