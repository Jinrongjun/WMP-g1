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

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from datetime import datetime

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False
    # env_cfg.commands.heading_command = True

    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0., 0.]  # kg
    env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False # =True For A1-amp
    # env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]


    env_cfg.terrain.mesh_type = 'plane'  # 'plane', 'slope', 'stair', 'gap', 'climb', 'crawl', 'tilt'
    # if(env_cfg.terrain.mesh_type == 'plane'):
    #     env_cfg.rewards.scales.feet_edge = 0
    #     env_cfg.rewards.scales.feet_stumble = 0


    if(args.terrain not in ['slope', 'stair', 'gap', 'climb', 'crawl', 'tilt']):
        print('terrain should be one of slope, stair, gap, climb, crawl, and tilt, set to climb as default')
        args.terrain = 'climb'
    env_cfg.terrain.terrain_proportions = {
        'slope': [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0],
        'stair': [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
        'gap': [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
        'climb': [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
        'tilt': [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
        'crawl': [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
     }[args.terrain]

    env_cfg.commands.ranges.lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0, 0]

    env_cfg.commands.ranges.flat_lin_vel_x = [0.6, 0.6]
    env_cfg.commands.ranges.flat_lin_vel_y = [-0.0, -0.0]
    env_cfg.commands.ranges.flat_ang_vel_yaw = [0.0, 0.0]

    env_cfg.depth.use_camera = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True

    if env_cfg.env.env_name == 'g1':
        train_cfg.runner.load_run = 'play'
    else:
        train_cfg.runner.load_run = 'play_27dof'


    train_cfg.runner.checkpoint = -1
    if 'g1' in env_cfg.env.env_name :
        ppo_runner, train_cfg = task_registry.make_wmp_runner_g1(env=env, name=args.task, args=args, train_cfg=train_cfg)
    else:
        ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.15)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        args.run_name = "g1"
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))
    img_idx = 0

    history_length = 5
    # trajectory_history, obs_without_command indexes modified for G1
    trajectory_history = torch.zeros(size=(env.num_envs, history_length, env.num_obs -
                                                    env.privileged_dim - env.height_dim - 3), device=env.device)
    obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                        obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
    
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

    world_model = ppo_runner._world_model.to(env.device)
    wm_latent = wm_action = None
    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_update_interval = env.cfg.depth.update_interval
    wm_action_history = torch.zeros(size=(env.num_envs, wm_update_interval, env.num_actions),
                                    device=env.device)
    wm_obs = {
        "prop": obs[:, :env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim], # modified for G1
        "is_first": wm_is_first,
    }

    if (env.cfg.depth.use_camera):
        wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                      device=world_model.device)

    wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)

    total_reward = 0
    not_dones = torch.ones((env.num_envs,), device=env.device)
    for i in range(1*int(env.max_episode_length) + 3):
        if (env.global_counter % wm_update_interval == 0):
            if (env.cfg.depth.use_camera):
                wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)

            wm_embed = world_model.encoder(wm_obs)
            wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
            wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
            wm_is_first[:] = 0

        history = trajectory_history.flatten(1).to(env.device)
        actions = policy(obs.detach(), history.detach(), wm_feature.detach())


        obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())

        not_dones *= (~dones)
        total_reward += torch.mean(rews * not_dones)

        # update world model input
        wm_action_history = torch.concat(
            (wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
        wm_obs = {
            "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim].to(world_model.device), # modified for G1
            "is_first": wm_is_first,
        }
        if (env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                          device=world_model.device)

        reset_env_ids = reset_env_ids.cpu().numpy()
        if (len(reset_env_ids) > 0):
            wm_action_history[reset_env_ids, :] = 0
            wm_is_first[reset_env_ids] = 1

        wm_action = wm_action_history.flatten(1)


        # process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                            obs[:, env.privileged_dim + 9:-env.height_dim]),
                                            dim=1)
        trajectory_history = torch.concat(
            (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            lootat = env.root_states[8, :3]
            camara_position = lootat.detach().cpu().numpy() + [0, 1, 0]
            env.set_camera(camara_position, lootat)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])
    if RENDER:
        video.release()
    print('total reward:', total_reward)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    RENDER = True
    args = get_args()
    args.rl_device = args.sim_device
    play(args)