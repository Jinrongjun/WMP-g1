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

import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# MOTION_FILES = glob.glob('datasets/mocap_motions/*')  # a list of csv file paths

MOTION_FILES = glob.glob('datasets/g1/walk.csv') # modified for G1

# Modofied from A1 WMP config
class G1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        prop_dim = 69 # proprioception, 3*base_ang_vel  + 3*gravity + 3*command + 29*dof_pos + 29*dof_vel + 2*phase
        action_dim = 29 # num_dof
        num_actions = 29
        privileged_dim = 18 + 3  # (removed)29*kp + 29*kd  + 18*contact_flag(number of joints included in penalize_contacts_on) + (removed for early training)6*DR_param + 3*base_lin_vel privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        height_dim = 187  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        forward_height_dim = 525 # for depth image prediction

        env_name = 'g1'
        use_amp = True
        forward_height_dim = 525 # for depth image prediction
        num_observations = prop_dim + privileged_dim + height_dim + action_dim - 3
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class terrain:
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        # 525 dim, for depth image prediction
        measured_forward_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                                     2.0]  # 1mx1.6m rectangle (without center line)
        measured_forward_points_y = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [wave, rough slope, stairs up, stairs down, discrete, gap, pit, tilt, crawl, rough_flat]
        terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.80]  # x,y,z [m]
        default_joint_angles = {  ### height = 0.7429
                    'left_hip_pitch_joint': -0.2, 
                    'left_hip_roll_joint': -0.0, 
                    'left_hip_yaw_joint': 0.0, 
                    'left_knee_joint': 0.42, 
                    'left_ankle_pitch_joint': -0.23,
                    'left_ankle_roll_joint': 0.0, 
                    'right_hip_pitch_joint': -0.2, 
                    'right_hip_roll_joint': 0.0, 
                    'right_hip_yaw_joint': 0.0, 
                    'right_knee_joint': 0.42, 
                    'right_ankle_pitch_joint': -0.23, 
                    'right_ankle_roll_joint': 0.0, 
                    'waist_yaw_joint': 0.0,
                    'waist_roll_joint': 0.0,
                    'waist_pitch_joint': 0.0,
                    # 'left_shoulder_pitch_joint': 0.25,
                    'left_shoulder_pitch_joint': 0.0,
                    'left_shoulder_roll_joint': 0.2,
                    'left_shoulder_yaw_joint': 0.15,
                    # 'left_elbow_joint': 0.85,
                    'left_elbow_joint': 1.2,
                    'left_wrist_roll_joint': 0.0,
                    'left_wrist_pitch_joint': 0.0,
                    'left_wrist_yaw_joint': 0.0,
                    # 'right_shoulder_pitch_joint': 0.25,
                    'right_shoulder_pitch_joint': 0.0,
                    'right_shoulder_roll_joint': -0.2,
                    'right_shoulder_yaw_joint': -0.15,
                    # 'right_elbow_joint': 0.85,
                    'right_elbow_joint': 1.2,
                    'right_wrist_roll_joint': 0.0,
                    'right_wrist_pitch_joint': 0.0,
                    'right_wrist_yaw_joint': 0.0,
                }

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
                     'hip_pitch': 200,
                     'hip_roll': 150,
                     'hip_yaw': 150,
                     'knee': 200,
                     'ankle': 100, # ??? 20
                     'waist': 200,
                     'shoulder': 20,
                     'elbow': 20,
                     'wrist_roll': 20,
                     'wrist_pitch': 5,
                     'wrist_yaw': 5,
                     }  # [N*m/rad]
        damping = {  
                     'hip_pitch': 5,
                     'hip_roll': 5,
                     'hip_yaw': 5,
                     'knee': 5,
                     'ankle': 5, # ??? 2
                     'waist': 5,
                     'shoulder': 0.5,
                     'elbow': 0.5,
                     'wrist_roll': 0.5,
                     'wrist_pitch': 0.2,
                     'wrist_yaw': 0.2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # ???
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20


    class depth:
        use_camera = True
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.005267, 0.000299, 0.449869]  # Modified for G1
        y_angle = [42, 42]  # positive pitch down  TODO: how to decide the rpy
        z_angle = [0, 0]
        x_angle = [0, 0]

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_zy.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee", "shoulder_yaw", "elbow", 'wrist']
            ### 不能包含 waist，胳膊会碰撞身体，不算reset,只能用肩膀的碰撞， 身体的俯仰角， 身体的高度 判断 reset
        # terminate_after_contacts_on = ["shoulder_pitch"] 
        terminate_after_contacts_on = ["pelvis", "waist" "shoulder_pitch", "knee"]  # original: ["pelvis", "waist" "shoulder_pitch"]

        body_name = "waist"
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

        flip_visual_attachments = False  ### NOTE ??? what for??

    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 2.0]
        randomize_restitution = False
        restitution_range = [0.0, 0.0]

        randomize_base_mass = False
        added_mass_range = [0., 3.]  # kg
        randomize_link_mass = False
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = False
        com_x_pos_range = [-0.05, 0.05]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.05, 0.05]

        push_robots = False
        push_interval_s = 15
        min_push_interval_s = 15
        max_push_vel_xy = 1.0

        randomize_gains = False
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = False
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = False
        latency_range = [0.00, 0.005]

    class normalization:
        class command_scales:
            lin_vel_x = 1.0   # min max [m/s]
            lin_vel_y = 1.0   # min max [m/s]
            ang_vel_yaw = 1.0    # min max [rad/s]
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 2.0
            dof_vel = 0.25
            dof_trq = 0.08
            # privileged
            height_measurements = 1.0
            contact_force = 0.005
            com_pos = 20
            pd_gains = 5


        clip_observations = 20.
        clip_actions = 20.0

        base_height = 0.75 # base height of G1, used to normalize measured height


    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            lin_vel = 0.1
            ang_vel = 0.2
            dof_pos = 0.01
            dof_vel = 1.5
            dof_trq = 1.0
            gravity = 0.05
            height_measurements = 0  # only for critic

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        reward_curriculum = False
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]
        base_pitch_target = -0.1 ### G1
        soft_dof_pos_limit = 0.9
        base_height_target = 0.75   #0.7 #G1
        foot_height_target = 0.15
        tracking_sigma = 0.15  # tracking reward = exp(-error^2/sigma)
        lin_vel_clip = 0.1
        default_gap = 0.22  #0.28, G1
        class scales(LeggedRobotCfg.rewards.scales):
            # G1
            tracking_lin_vel = 20.0
            tracking_ang_vel = 20.0

            alive = 2.0  ### 0.15
            lin_vel_z = -1.0
            ang_vel_xy = 0.0 ## -0.5   ###-0.1
            orientation = -1.0


            stand_normal = -0.01 #-0.05 too high
            base_height = -10.0  ## -1.0
            # dof_acc = -2.5e-7  ### -2.5e-7
            # dof_vel = -1e-5   ### -1e-3
            action_rate = -5e-3  ### -0.005
            smoothness  = -1e-2    # The same as action_smoothness          

            dof_pos_limits = -10.0 ### -5.0
            # dof_vel_limits = -0.1
            torque_limits =  0.0 # -1.0 

            # feet_contact_forces = -5e-4  ### 1e-3
            # contact_no_vel = -0.1  ## -0.2

            torques = -6e-7
            delta_torques = 0.0 # difference between torque and last torque
            #feet_contact_slip = -0.1
            clearance = -0.05
            feet_distance = -0.1
            

            collision = -5.0

            feet_swing_height = 0.0 # -20.0  # -20.0

            # contact = 0.1 
            contact = -1.0 


            hip_pos = -1.0  ### -1.0
            waist_pos = -1.0  ###  -1.0
            arm_pos = -1.0

            # A1-amp
            # tracking_lin_vel = 1.5
            # tracking_ang_vel = 0.5
            # torques = -0.0001
            # dof_acc = -2.5e-7
            # base_height = -0.
            # feet_air_time = 0.5
            # collision = -1.0
            # feet_stumble = -0.1
            # action_rate = -0.03

            # # feet_edge = -1.0  # TODO: implement it
            # dof_error = -0.04

            # lin_vel_z = -1.0
            # cheat = -1
            # stuck = -1


    class commands:
        curriculum = False
        max_lin_vel_forward_x_curriculum = 1.0
        max_lin_vel_backward_x_curriculum = 0.0
        max_lin_vel_y_curriculum = 0.0
        max_ang_vel_yaw_curriculum = 1.0

        max_flat_lin_vel_forward_x_curriculum = 1.0
        max_flat_lin_vel_backward_x_curriculum = 0.0
        max_flat_lin_vel_y_curriculum = 0.0
        max_flat_ang_vel_yaw_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [0.0, 0.8]  # min max [m/s]
            lin_vel_y = [-0., 0.]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-0., 0.]

            flat_lin_vel_x = [-0.0, 0.8]  # min max [m/s]
            flat_lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            flat_ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            flat_heading = [-3.14 / 4, 3.14 / 4]


class G1CfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'WMPRunner'

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        vel_predict_coef = 1.0
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'flat_push1'
        experiment_name = 'g1'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 20000  # number of policy updates
        save_interval = 1000

        amp_reward_coef = 0.5 * 0.02  # set to 0 means not use amp reward
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0, 0, 0] * 9  + [0, 0]     # TODO: which value? [0.05, 0.02, 0.05] * 4

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100

