# 一个改版的g1 locomotion base环境，包含了WMP所需的环境更新过程


from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.math import get_euler_xyz__ as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain import Terrain
import cv2
# Camera position settings (global)
COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.],
    [0.183, -0.047, 0.],
    [-0.183, 0.047, 0.],
    [-0.183, -0.047, 0.]]) + COM_OFFSET


class LeggedRobotG1(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):

        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # self.actions[:,:16] = torch.clip(actions[:,:16], -clip_actions, clip_actions).to(self.device)
        # self.actions[:,16:20] = torch.clip(actions[:,16:20], -clip_actions*5, clip_actions*5).to(self.device)
        # self.actions[:,20:23] = torch.clip(actions[:,20:23], -clip_actions, clip_actions).to(self.device)
        # self.actions[:,23:27] = torch.clip(actions[:,23:27], -clip_actions*5, clip_actions*5).to(self.device)


        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.critic_obs_buf = torch.clip(self.critic_obs_buf, -clip_obs, clip_obs)
        # WMP
        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # depth_buffer:(num_envs, buffer_len, height, width), 取每个环境的倒数第二帧深度图像
            # interpolation = torch.rand((self.cfg.depth.camera_num_envs, 1, 1), device=self.device)
            # self.extras["depth"] = self.depth_buffer[:, -1] * interpolation + self.depth_buffer[:, -2] * (1-interpolation)
        else:
            self.extras["depth"] = None
        # WMP
            
        return self.obs_buf, self.critic_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    # WMP
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image

    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        # depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        # self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)  # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        start_time = time()
        self.gym.start_access_image_tensors(self.sim)
        # for i in range(self.num_envs):
        for i in range(len(self.depth_index)):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                                self.envs[self.depth_index[i]],
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)

            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            # if(i == 0): print(torch.mean(depth_image)) # for debug, sometimes isaacgym will return all -inf depth image if not config properly

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
                                                 dim=0)
        self.gym.end_access_image_tensors(self.sim)
        print('acquiring depth image time:', time()-start_time)

    # attach depth camera into the envs
    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[1]
            camera_props.height = self.cfg.depth.original[0]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_y_angle = np.random.uniform(config.y_angle[0], config.y_angle[1])

            camera_z_angle = np.random.uniform(config.z_angle[0], config.z_angle[1])
            camera_x_angle = np.random.uniform(config.x_angle[0], config.x_angle[1])


            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(np.radians(camera_x_angle),
                                                           np.radians(camera_y_angle), np.radians(camera_z_angle))
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform,
                                           gymapi.FOLLOW_TRANSFORM)
    # WMP

    def get_extra_info(self):
        return self.obs_history_buf, self.base_lin_vel 

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.episode_length_buf += 1

        # self.common_step_counter += 1   ### Scalar
        self.push_interval_counter += 1  ### Scalar
        # self.obs_history_counter += 1 ### Scalar

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        ### in world frame
        self.feet_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,0:3]
        self.feet_vel = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,7:10]
        ### in body frame
        for i in range(self.feet_num):
            self.feet_pos_in_body_frame[:,i,:] = quat_rotate_inverse(self.base_quat, self.feet_pos[:,i,:]- self.base_pos)

        ### dof_pos ,  dof_vel
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        #### 检测到一次接触 就是 接触，  检测到两次没有接触，才是没有接触
        self.contact_flag_last = self.contact_flag
        contact_ = torch.where(self.contact_forces[:, self.feet_indices, 2] >1.0, 1.0, 0.0)
        self.contact_flag = torch.where(torch.logical_or(contact_, self.contact_flag_logical_or), 1.0, 0.0)
        self.contact_flag_logical_or = contact_
        
        self.phase = (self.episode_length_buf * self.dt) % self.phase_period / self.phase_period
        self.phase_left = self.phase
        self.phase_right = (self.phase + self.phase_offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(-1), self.phase_right.unsqueeze(-1)], dim=-1) 

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(reset_env_ids)
        self.update_depth_buffer()

        self._update_commands()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)


        # if self.common_step_counter == 100:
        #     print("begin_force")
        # if self.common_step_counter > 100:
        #     self.apply_body_forces[:,13,0] = -400
        #         ###  NOTE  pos 
        #     self.pos_of_apply_body_forces[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,0:3]
        #     self.pos_of_apply_body_forces[:,13,2] += 0.22
        #     self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, \
        #         gymtorch.unwrap_tensor(self.apply_body_forces), gymtorch.unwrap_tensor(self.pos_of_apply_body_forces), gymapi.ENV_SPACE) #ENV_SPACE / LOCAL_SPACE

        # self.external_forces_counter +=1    ###  num_env
        # self.apply_external_forces()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # WMP
        # for visualizing the depth image
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # self._draw_debug_vis()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
        # WMP


    def apply_external_forces(self):
        '''
            先判断是否需要重新采样 扰动力大小
            重新采样扰动力
            重新采样扰动力持续时间
            清零扰动计时器
            按照 change_per_step 更新扰动力
            施加扰动力
        '''
        env_ids = (self.external_forces_counter % self.external_forces_duration.squeeze(1).int() == 0).nonzero(as_tuple=False).flatten()
        self._resample_external_forces(env_ids)
        self.apply_body_forces[:,13,:] = self.external_forces_sample[:,:]
            ###  NOTE  pos 
        self.pos_of_apply_body_forces[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,0:3]
        self.pos_of_apply_body_forces[:,13,2] += 0.22

        self.apply_body_forces[:,5,:] = self.left_foot_external_forces_sample[:,:]
        self.apply_body_forces[:,11,:] = self.right_foot_external_forces_sample[:,:]

        ### apply external_forces
        self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, \
            gymtorch.unwrap_tensor(self.apply_body_forces), gymtorch.unwrap_tensor(self.pos_of_apply_body_forces), gymapi.ENV_SPACE) #ENV_SPACE / LOCAL_SPACE

    def _resample_external_forces(self, resample_env_ids):
        range_xy = self.cfg.external_forces.max_force_xy
        range_z = self.cfg.external_forces.max_force_z
        self.external_forces_sample[resample_env_ids,:2] = torch_rand_float(range_xy[0],range_xy[1], (len(resample_env_ids), 2), device=self.device)
        self.external_forces_sample[resample_env_ids,2:3] = torch_rand_float(range_z[0],range_z[1], (len(resample_env_ids), 1), device=self.device)
            ###  采样的的 需要施加 扰动力的 环境， 才施加 扰动力
        self.external_forces_sample[resample_env_ids,:] =  self.external_forces_sample[resample_env_ids,:] * self.env_apply_body_forces_flag[resample_env_ids].unsqueeze(-1)
        ### 
        self.external_forces_duration[resample_env_ids] = torch_rand_float(self.external_forces_min_duration, 
                                            self.external_forces_max_duration, (len(resample_env_ids), 1), device=self.device)
        ###                               
        self.external_forces_counter[resample_env_ids] = 0

        ###------------  foot  ----------
        foot_range_xy = self.cfg.external_forces.foot_max_force_xy
        foot_range_z = self.cfg.external_forces.foot_max_force_z

        self.left_foot_external_forces_sample[resample_env_ids,:2] = torch_rand_float(foot_range_xy[0],foot_range_xy[1], (len(resample_env_ids), 2), device=self.device)
        self.left_foot_external_forces_sample[resample_env_ids,2:3] = torch_rand_float(foot_range_z[0],foot_range_z[1], (len(resample_env_ids), 1), device=self.device)
        self.left_foot_external_forces_sample[resample_env_ids,:] =  self.left_foot_external_forces_sample[resample_env_ids,:] * self.env_apply_body_forces_flag[resample_env_ids].unsqueeze(-1)
        
        self.right_foot_external_forces_sample[resample_env_ids,:2] = torch_rand_float(foot_range_xy[0],foot_range_xy[1], (len(resample_env_ids), 2), device=self.device)
        self.right_foot_external_forces_sample[resample_env_ids,2:3] = torch_rand_float(foot_range_z[0],foot_range_z[1], (len(resample_env_ids), 1), device=self.device)
        self.right_foot_external_forces_sample[resample_env_ids,:] =  self.right_foot_external_forces_sample[resample_env_ids,:] * self.env_apply_body_forces_flag[resample_env_ids].unsqueeze(-1)



    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.base_rpy[:,1])>1.0, torch.abs(self.base_rpy[:,0])>0.8)
        self.reset_buf |= ((self.base_pos[:,2] - self.measured_heights[:,93]) < 0.35)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
            # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.phase[env_ids] = 0 

        self.obs_history_buf[env_ids] = 0.
        
        self.contact_flag[env_ids] = 0.
        self.contact_flag_logical_or[env_ids] = 0.
        self.contact_flag_last[env_ids] = 0.


        # # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.commands[env_ids] = 0.0  
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    

    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, ### 27
                                    self.dof_vel * self.obs_scales.dof_vel,### 27
                                    self.actions,  ## 27
                                    
                                    sin_phase,
                                    cos_phase,
                                    ),dim=-1)
        
        if self.cfg.terrain.measure_heights:
            heights_obs = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements

        self.critic_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions, ### 27
                                    sin_phase,
                                    cos_phase,
                                    heights_obs, ### 187
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


        ### get obs_history  ### 正数， 向后移动,  
        self.obs_history_buf = torch.roll(self.obs_history_buf, self.cfg.env.num_obs_step, dims=-1) #TODO: 把[:,:] 删除掉有可能就能训练出来。 [重要]
        self.obs_history_buf[:,:self.cfg.env.num_obs_step].copy_(self.obs_buf) 

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='trimesh':
            self._create_trimesh()

        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1)) ### (num_envs, 1)
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets,1), device='cpu') ###(64,1)
                self.restitution_coeffs = restitution_buckets[bucket_ids] ### (num_env,1)
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props



    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        if self.cfg.domain_rand.randomize_motor_strength:
            rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_id, :] = torch.rand(self.num_dof, dtype=torch.float, device=self.device,requires_grad=False).unsqueeze(0) \
                                                            * (rng[1] - rng[0]) + rng[0]
     
        if self.cfg.domain_rand.randomize_motor_offset:
            rng = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_id, :] = torch.rand(self.num_dof, dtype=torch.float, device=self.device,requires_grad=False).unsqueeze(0) \
                                                            * (rng[1] - rng[0]) + rng[0]
     
        if self.cfg.domain_rand.randomize_Kp_factor:
            rng = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_id, :] = torch.rand(self.num_dof, dtype=torch.float, device=self.device,requires_grad=False).unsqueeze(0) \
                                                            * (rng[1] - rng[0]) + rng[0]

        if self.cfg.domain_rand.randomize_Kd_factor:
            rng = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_id, :] = torch.rand(self.num_dof, dtype=torch.float, device=self.device,requires_grad=False).unsqueeze(0) \
                                                            * (rng[1] - rng[0]) + rng[0]

                
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        if self.cfg.domain_rand.randomize_base_com:
            rng = self.cfg.domain_rand.base_com_range
            com_displacements = torch.rand(3, dtype=torch.float, device=self.device,requires_grad=False)\
                                                                        * (rng[1] - rng[0]) + rng[0]
            com_displacements[2] = com_displacements[2] * 1.5
            props[0].com = gymapi.Vec3(com_displacements[0], com_displacements[1],com_displacements[2])
        # print("props[0].comx:",props[0].com.x)
        # print("props[0].comy:",props[0].com.y)
        # print("props[0].comz:",props[0].com.z)

        if self.cfg.domain_rand.randomize_base_inertia:
            rng = self.cfg.domain_rand.base_inertia_range
            inerita = torch.rand(3, dtype=torch.float, device=self.device,requires_grad=False)\
                                                                        * (rng[1] - rng[0]) + rng[0]
            # print("inerita:",inerita)
            props[0].inertia.x = gymapi.Vec3(inerita[0],5.93180607e-4,7.324662e-6)
            props[0].inertia.y = gymapi.Vec3(5.93180607e-4,inerita[1],2.0969537e-5)
            props[0].inertia.z = gymapi.Vec3(7.324662e-6,2.0969537e-5,inerita[2])
            # print("props[0].inertia:",props[0].inertia.x) ###惯性矩阵的 第一行
            # print("props[0].inertia:",props[0].inertia.y)
            # print("props[0].inertia:",props[0].inertia.z)

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        if self.cfg.domain_rand.push_robots and  (self.push_interval_counter % self.push_interval_length == 0):
            self._push_robots()
            self.push_interval_counter = 0

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.foot_clearance = self._get_foot_clearance()
            # print("self.measured_heights:",self.measured_heights)

    def _resample_commands(self, reset_env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # # set small commands to zero
        # self.commands[reset_env_ids, :2] *= (torch.norm(self.commands[reset_env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        self.commands_sample[reset_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(reset_env_ids), 1), device=self.device).squeeze(1)
        self.commands_sample[reset_env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(reset_env_ids), 1), device=self.device).squeeze(1)
        self.commands_sample[reset_env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(reset_env_ids), 1), device=self.device).squeeze(1)

        self.commands_sample[reset_env_ids, 1] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, :1], dim=-1) > 0.4), \
                torch.clamp(self.commands_sample[reset_env_ids, 1], -0.5, 0.5), self.commands_sample[reset_env_ids, 1])
        self.commands_sample[reset_env_ids, 1] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, :1], dim=-1) > 0.6), \
                torch.clamp(self.commands_sample[reset_env_ids, 1], -0.3, 0.3), self.commands_sample[reset_env_ids, 1])
        self.commands_sample[reset_env_ids, 1] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, :1], dim=-1) > 1.0), \
                torch.clamp(self.commands_sample[reset_env_ids, 1], -0.0, 0.0), self.commands_sample[reset_env_ids, 1])
        
        self.commands_sample[reset_env_ids, 2] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, 1:2], dim=-1) > 0.2), \
                torch.clamp(self.commands_sample[reset_env_ids, 2], -0.3, 0.3), self.commands_sample[reset_env_ids, 2])
        self.commands_sample[reset_env_ids, 2] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, 1:2], dim=-1) > 0.3), \
                torch.clamp(self.commands_sample[reset_env_ids, 2], -0.2, 0.2), self.commands_sample[reset_env_ids, 2])
        self.commands_sample[reset_env_ids, 2] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, 1:2], dim=-1) > 0.5), \
                torch.clamp(self.commands_sample[reset_env_ids, 2], -0.0, 0.0), self.commands_sample[reset_env_ids, 2])
                
        self.commands_sample[reset_env_ids, 2] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, :1], dim=-1) > 0.6), \
                torch.clamp(self.commands_sample[reset_env_ids, 2], -0.5, 0.5), self.commands_sample[reset_env_ids, 2])
        self.commands_sample[reset_env_ids, 2] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, :1], dim=-1) > 1.0), \
                torch.clamp(self.commands_sample[reset_env_ids, 2], -0.35, 0.35), self.commands_sample[reset_env_ids, 2])
        self.commands_sample[reset_env_ids, 2] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, :1], dim=-1) > 1.5), \
                torch.clamp(self.commands_sample[reset_env_ids, 2], -0.2, 0.2), self.commands_sample[reset_env_ids, 2])
        ###  zero vel x,y , zero vel_yaw
        self.commands_sample[reset_env_ids, 2] = \
                torch.where((torch.norm(self.commands_sample[reset_env_ids, :2], dim=-1) < 0.3), \
                torch.clamp(self.commands_sample[reset_env_ids, 2], -0.0, 0.0), self.commands_sample[reset_env_ids, 2])
     

        # set small commands to zero
        self.commands_sample[reset_env_ids, :2] *= (torch.norm(self.commands_sample[reset_env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        ### at  reset function,   need to  set commands is zero

    def _update_commands(self):
        self.commands[:,:1] = torch.where( (self.commands[:, :1]-self.commands_sample[:,:1]) > 0.01,
                                    torch.clamp((self.commands[:, :1]-0.02),min=self.commands_sample[:,:1],max=None),
                                    torch.clamp((self.commands[:, :1]+0.02),min=None, max=self.commands_sample[:,:1]))
        self.commands[:,1:2] = torch.where( (self.commands[:, 1:2]-self.commands_sample[:,1:2]) > 0.01,
                                    torch.clamp((self.commands[:, 1:2]-0.02),min=self.commands_sample[:,1:2],max=None),
                                    torch.clamp((self.commands[:, 1:2]+0.02),min=None, max=self.commands_sample[:,1:2]))
        self.commands[:,2:3] = torch.where( (self.commands[:, 2:3]-self.commands_sample[:,2:3]) > 0.01,
                                    torch.clamp((self.commands[:, 2:3]-0.02),min=self.commands_sample[:,2:3],max=None),
                                    torch.clamp((self.commands[:, 2:3]+0.02),min=None, max=self.commands_sample[:,2:3]))
             

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type=="P":
            if self.cfg.domain_rand.randomize_Kp_factor:
                torques = self.p_gains*self.Kp_factors*(actions_scaled + self.default_dof_pos + self.motor_offsets - self.dof_pos)\
                                                    - self.d_gains*self.Kd_factors*self.dof_vel
            else:
                torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.cfg.domain_rand.randomize_motor_strength:
            torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, reset_env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[reset_env_ids] = self.base_init_state
        self.root_states[reset_env_ids, :3] += self.env_origins[reset_env_ids]
        self.root_states[reset_env_ids, :2] += torch_rand_float(-1., 1., (len(reset_env_ids), 2), device=self.device) # xy position within 1m of the center
        # self.root_states[reset_env_ids, :2] += torch_rand_float(-0., 0., (len(reset_env_ids), 2), device=self.device) # xy position within 1m of the center
        
        # base velocities
        self.root_states[reset_env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(reset_env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        # self.root_states[reset_env_ids, 7:13] = torch_rand_float(-0., 0., (len(reset_env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = reset_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            ####  not use  error
        # self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, 13, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        # self.gym.set_rigid_body_state_tensor(self.sim, gymtorch.unwrap_tensor(self.rigid_body_state))

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length_s) > (0.8 * self.reward_scales["tracking_lin_vel"] / self.dt):
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum_lin_vel_x)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:36] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[36:63] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[63:90] = 0. # previous actions
        noise_vec[90:92] = 0. # sin/cos phase
        
        return noise_vec


    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        ### dof_pos ,  dof_vel
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
            ### used to diff dof_acc
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        ### feet_pos,  feet_vel, in  world frame   pos 足端 平地上 有 正的 0.02cm， 偏置 
        self.feet_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,0:3]
        self.feet_vel = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,7:10]
        self.feet_num = len(self.feet_indices)

        ### base_pos ,  base_quat, base_euler,   lin_vel,  ang_vel   in world frame
        self.base_pos = self.root_states[:,0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_rpy = get_euler_xyz_in_tensor(self.base_quat)

                ### in body frame
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.feet_pos_in_body_frame = torch.zeros_like(self.feet_pos)
        for i in range(self.feet_num):
            self.feet_pos_in_body_frame[:,i,:] = quat_rotate_inverse(self.base_quat, self.feet_pos[:,i,:]- self.base_pos) 

        ###  contact_flag  
        self.contact_flag = torch.zeros(self.num_envs, self.feet_num, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_flag_logical_or = torch.zeros(self.num_envs, self.feet_num, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_flag_last = torch.zeros(self.num_envs, self.feet_num, dtype=torch.float, device=self.device, requires_grad=False)
        
        ### in rough terrain  foot clearance
        self.foot_clearance = torch.zeros_like(self.feet_pos[...,2])

        ### height_map_points
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()

        # reinit
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs_step*self.num_obs_history, device=self.device, dtype=torch.float, requires_grad=False)  #TODO: 本来是有梯度的.


        # initialize some data used later on
        self.push_interval_counter = 0
        self.common_step_counter = 0
        self.obs_history_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.commands_sample = torch.zeros_like(self.commands)
        
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.phase_period = 0.7
        self.phase_offset=0.5
        self.phase = torch.zeros_like(self.episode_length_buf,requires_grad=False)
        self.phase_left = self.phase
        self.phase_right = (self.phase + self.phase_offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(-1), self.phase_right.unsqueeze(-1)], dim=-1) 

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

###-------------   external force disturbance ----------------
        self.apply_body_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float,requires_grad=False)
        self.pos_of_apply_body_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float,requires_grad=False)
        self.external_forces_counter = torch.zeros_like(self.episode_length_buf,requires_grad=False)
        self.external_forces_duration = torch_rand_float(self.external_forces_min_duration, 
                                                      self.external_forces_max_duration, (self.num_envs, 1), device=self.device)
        self.external_forces_sample = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_foot_external_forces_sample = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_foot_external_forces_sample = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)



        ###  以 设定的 添加扰动力的 环境的 占比 进行采用， 选择一些环境施加扰动，  而不是所有环境都施加扰动   ------------------
        self.env_apply_body_forces_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.float,requires_grad=False)

        # WMP
        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.cfg.depth.camera_num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)
        # WMP


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            # if name=="termination":
            #     continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        
        self.num_height_points = grid_x.numel()  ### 187= 17*11
        
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        # print("init_height_point:",points[:, :, 0])
        return points

    def _get_heights(self):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw
            Z axis  is  world base
            在机器人脚下的 对齐 yaw 的 高程图
        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        ### quat_apply_yaw 主要考虑 (x,y) 的变化
        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size  ### 主要考虑 (x,y) 的变化
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_foot_clearance(self):
        if self.cfg.terrain.mesh_type == 'plane':
            raise ValueError("in plane ,  this is error")
            # return torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
   
        # self.feet_pos  ### in world frame  (env_num,2,3)
        points = torch.clone(self.feet_pos)
        # print("self.feet_pos:",self.feet_pos)
        points[...,0:2] += self.terrain.cfg.border_size 
        # points[...,2:3] -= 0.02 
        
        ### (env_num,2,3)
        points[...,0:2] = (points[...,0:2]/self.terrain.cfg.horizontal_scale).long()
        px = points[..., 0].long().view(-1) ### env_num*2,0
        py = points[..., 1].long().view(-1) ### env_num*2,1
        heights1 = self.height_samples[px, py] 
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)  ### env*2

        foot_clearance = points[...,2] - heights.view(self.num_envs, 2)*self.terrain.cfg.vertical_scale
        ###foot_clearance  shape  :   [num_envs,2]
        return foot_clearance


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

            if self.cfg.terrain.play:
                self.rows = torch.randint(0, 1, (self.num_envs,), device=self.device)
                self.cols = torch.randint(0, 1, (self.num_envs,), device=self.device)
                self.env_origins[:] = self.terrain_origins[self.rows, self.cols]
            else:
                self.rand_rows = torch.randint(0, self.cfg.terrain.num_rows, (self.num_envs,), device=self.device)
                self.rand_cols = torch.randint(0, self.cfg.terrain.num_cols, (self.num_envs,), device=self.device)
                self.env_origins[:] = self.terrain_origins[self.rand_rows, self.rand_cols]
        else:
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_length = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        self.external_forces_min_duration = np.ceil(self.cfg.external_forces.min_duration / self.dt)
        self.external_forces_max_duration = np.ceil(self.cfg.external_forces.max_duration / self.dt)

    #WMP
    # 获取高程图，塞给wm里面的深度图predictor    
    def get_forward_map(self):
        return torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_forward_heights, -1,
                             1.) * self.obs_scales.height_measurements
    #WMP


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.measured_heights[:,93] - self.cfg.rewards.base_height_target)
    
    def _reward_stand_normal(self):
        # Penalize base height away from target
        base_pitch = self.base_rpy[:, 1]
        return torch.square(base_pitch - self.cfg.rewards.base_pitch_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        first_contact = (self.feet_air_time > 0.) * self.contact_flag
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.2 #no reward for zero command
        self.feet_air_time *= torch.where(torch.logical_not(self.contact_flag), 1.0, 0.0)
        return rew_airTime
    
    def _reward_feet_clearance(self):
        feet_air = torch.where(self.contact_flag>0., 0.0, 1.0)
        feet_vel_norm = torch.norm(self.feet_vel[...,0:2],dim=-1)
        feet_air_vel_norm = feet_vel_norm * feet_air
        feet_air_z_cost = torch.sum(torch.square(self.foot_clearance - 0.10) * feet_air_vel_norm , dim=-1) 
        return feet_air_z_cost

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.15)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.5
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.03) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)

    def _reward_hip_yaw_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[2,8]]), dim=1)
    
    def _reward_hip_roll_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,7]]), dim=1)
    
    def _reward_feet_distance(self):
        distance_error = torch.abs((self.feet_pos_in_body_frame[:,0,1] - self.feet_pos_in_body_frame[:,1,1]) - self.cfg.rewards.default_gap) ##0.284
        return distance_error
    
    def _reward_not_fly(self):
        return torch.where(torch.logical_or(self.contact_flag[:,0], self.contact_flag[:,1]), 1.0, 0.0)
    
    
    def _reward_feet_contact_slip(self):
        feet_vel_norm = torch.norm(self.feet_vel, dim=-1)
        return torch.sum(feet_vel_norm * self.contact_flag , dim=-1)
    

    def _reward_waist_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,12:13]), dim=-1)

    def _reward_arm_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[13,14,15,16,17,18,19, 20,21,22,23,24,25,26]] - self.default_dof_pos[:,[13,14,15,16,17,18,19, 20,21,22,23,24,25,26]]), dim=-1)
        # return torch.sum(torch.square(self.dof_pos[:,[14,15,16,17,18,19,  21,22,23,24,25,26]] - self.default_dof_pos[:,[14,15,16,17,18,19, 21,22,23,24,25,26]]), dim=-1)

    def _reward_ankle_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[4,5, 10,11]] - self.default_dof_pos[:,[4,5, 10,11]]), dim=-1)
   



    def _reward_arm_joint_power(self):
        power = torch.sum(torch.abs(self.dof_vel[:,[13,14,15,16,17,18,19, 20,21,22,23,24,25,26]] * self.torques[:,[13,14,15,16,17,18,19, 20,21,22,23,24,25,26]]), dim=-1) ### F*v= P (power)
        return power    
    

    def _reward_waist_upper_actions(self):
        return torch.sum(torch.square(self.actions[:,12:]), dim=-1)


    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,:13] - self.actions[:,:13]), dim=1)
    
    def _reward_action_smoothness(self):
        # Penalize changes in actions_smooth
        return torch.sum(torch.square(self.actions - 2*self.last_actions + self.last_last_actions), dim=1)
    

    def _reward_upper_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,13:] - self.actions[:,13:]), dim=1)
    
    def _reward_upper_action_smoothness(self):
        # Penalize changes in actions_smooth
        return torch.sum(torch.square(self.actions[:,13:] - 2*self.last_actions[:,13:] + self.last_last_actions[:,13:]), dim=1)
    

