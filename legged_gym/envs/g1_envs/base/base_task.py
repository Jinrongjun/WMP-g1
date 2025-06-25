import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1


        self.num_envs = cfg.env.num_envs
        self.num_obs_step = cfg.env.num_obs_step
        self.num_critic_obs = cfg.env.num_critic_obs
        self.num_actions = cfg.env.num_actions
        self.num_obs_history = cfg.env.num_obs_history


        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs_step*self.num_obs_history, device=self.device, dtype=torch.float)
        
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs_step, device=self.device, dtype=torch.float)
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_critic_obs, device=self.device, dtype=torch.float)
        
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.extras = {}


        self.motor_strengths = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                     requires_grad=False)

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            ######## -----------------------------------
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "w")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "a")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "s")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "d")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "e")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "q")
            self.theta = 0.0
            self.camera_direction = np.array([np.cos(self.theta), np.sin(self.theta), 0.])  
            self.camera_direction2 = np.array([np.cos(self.theta + 0.5 * np.pi), np.sin(self.theta + 0.5 * np.pi), 0.])
            self.camera_pos = np.array(cfg.viewer.pos)
            self.camera_lookat = np.array(cfg.viewer.lookat)



    def get_observations(self):
        return self.obs_buf, self.critic_obs_buf
    
    ###  继承类 重构
    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # self.obs_buf, self.critic_obs_buf, self.rew_buf, self.reset_buf, self.extras = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False),
        #                                                                                         torch.zeros(self.num_envs, 10, device=self.device, requires_grad=False))
        
        self.obs_buf, self.critic_obs_buf, self.rew_buf, self.reset_buf, self.extras = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))

        return self.obs_buf, self.critic_obs_buf
    
    
    ###  继承类 重构
    def step(self, actions):
        raise NotImplementedError



    def _set_camera(self,pos,lookat):
        cam_pos = gymapi.Vec3(*pos)
        cam_target = gymapi.Vec3(*lookat)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

                if evt.action == 'w' and evt.value > 0:
                    self.camera_pos = self.camera_pos +0.5 * self.camera_direction
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 's' and evt.value > 0:
                    self.camera_pos = self.camera_pos - 0.5 *  self.camera_direction
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 'q' and evt.value > 0:
                    self.theta = self.theta + 0.1
                    self.camera_direction = np.array([np.cos(self.theta), np.sin(self.theta), 0.])  
                    self.camera_direction2 = np.array([np.cos(self.theta + 0.5 * np.pi), np.sin(self.theta + 0.5 * np.pi), 0.])
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 'e' and evt.value > 0:
                    self.theta = self.theta -0.1
                    self.camera_direction = np.array([np.cos(self.theta), np.sin(self.theta), 0.])  
                    self.camera_direction2 = np.array([np.cos(self.theta + 0.5 * np.pi), np.sin(self.theta + 0.5 * np.pi), 0.])
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 'a' and evt.value > 0:
                    self.camera_pos = self.camera_pos + 0.5 * self.camera_direction2
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)
                if evt.action == 'd' and evt.value > 0:
                    self.camera_pos = self.camera_pos - 0.5 *  self.camera_direction2
                    self._set_camera(self.camera_pos, self.camera_direction + self.camera_pos)


            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)