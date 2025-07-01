from ..base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        env_spacing = 3.  # 如果是 plane 地形， 每个机器人间隔 3m
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        
        num_envs = 4096
        num_obs_step = 92    # actor 使用的obs的维度 (单个时刻，没有考虑到多步的历史)
        num_critic_obs = 282 # critic 使用的obs的维度
        num_obs_history = 5  # history 的长度
        num_actions = 27     # action 的数量


    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh

        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 50 # [m] ### max_vel * episode_lens
        static_friction = 0.95
        dynamic_friction = 0.1
        restitution = 0.0  

        play = False

### ---------------    plane ----------------
        slope_max = 0.12 ## 30+ degree
        discrete_height = 0.025
        uniform_terrain_height_max = 0.03
        step_height_max = 0.1  ##0.1
        wave_terrain_amplitude_max = 0.04 ### 0.1 base + 0.15


### ---------------   terrain  ----------------
        # slope_max = 0.2 ## 30+ degree
        # discrete_height = 0.04
        # uniform_terrain_height_max = 0.04
        # step_height_max = 0.06  ##0.1
        # wave_terrain_amplitude_max = 0.04 ### 0.1 base + 0.15

        # slope_max = 0.25 ## 30+ degree
        # discrete_height = 0.05
        # uniform_terrain_height_max = 0.04
        # step_height_max = 0.08  ##0.1
        # wave_terrain_amplitude_max = 0.04 ### 0.1 base + 0.15


        # slope_max = 0.3 ## 30+ degree
        # discrete_height = 0.06
        # uniform_terrain_height_max = 0.04
        # step_height_max = 0.10  ##0.1
        # wave_terrain_amplitude_max = 0.04 ### 0.1 base + 0.15



        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # mun=17 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] ### num=11
        
        curriculum = True
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.

        num_rows= 20 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)

        num_sub_terrains = 0 ##  num_rows * num_clos
        terrain_proportions = [0.0, 0.0, 0.0, 1.0]  #[0.2, 0.3, 0.2, 0.3]

        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces


    class commands( LeggedRobotCfg.commands ):
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        curriculum = True

### -------   plane  -------------------
        max_curriculum_lin_vel_x = 1.0
        max_curriculum_lin_vel_y = 0.5
        max_curriculum_ang_vel_yaw = 1.0
        resampling_time =10.0
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]

### -------   terrain  -------------------
        # max_curriculum_lin_vel_x = 1.0
        # max_curriculum_lin_vel_y = 0.0
        # max_curriculum_ang_vel_yaw = 1.0
        # resampling_time =10.0
        # class ranges:
        #     lin_vel_x = [-0.0, 1.0] # min max [m/s]
        #     lin_vel_y = [-0.0, 0.0]   # min max [m/s]
        #     ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.80] # x,y,z [m]  
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
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
                     'hip_pitch': 200,
                     'hip_roll': 150,
                     'hip_yaw': 150,
                     'knee': 200,
                     'ankle': 20,
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
                     'ankle': 2,
                     'waist': 5,
                     'shoulder': 0.5,
                     'elbow': 0.5,
                     'wrist_roll': 0.5,
                     'wrist_pitch': 0.2,
                     'wrist_yaw': 0.2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5  ### arm * 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        # decimation = 4
        decimation = 20


    class depth:
        use_camera = False
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        y_angle = [-5, 5]  # positive pitch down
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


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_27dof-zy.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee", "shoulder_yaw", "elbow", 'wrist']
            ### 不能包含 waist，胳膊会碰撞身体，不算reset,只能用肩膀的碰撞， 身体的俯仰角， 身体的高度 判断 reset
        # terminate_after_contacts_on = ["shoulder_pitch"] 
        terminate_after_contacts_on = ["pelvis", "waist" "shoulder_pitch"] 

        body_name = "waist"

        # self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

        flip_visual_attachments = False  ### NOTE
  
    class normalization:
        class command_scales:
            lin_vel_x = 1.0   # min max [m/s]
            lin_vel_y = 1.0   # min max [m/s]
            ang_vel_yaw = 1.0    # min max [rad/s]
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_trq = 0.08
            height_measurements = 1.0
        
        clip_observations = 30.0
        clip_actions = 5.0


        
    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:  ### zhen dui de shi  obs de noise , zhe ge zhi cheng yi obs scale tian jia dao obs zhong 
            lin_vel = 0.1
            ang_vel = 0.2
            dof_pos = 0.01
            dof_vel = 1.5
            dof_trq = 1.0
            gravity = 0.05
  

    class domain_rand(LeggedRobotCfg.domain_rand):
        ### env contact porperty
        randomize_friction = False
        friction_range = [0.7, 1.0]
        randomize_restitution =False
        restitution_range = [0.0, 0.05]
       
        ### body porperty
        randomize_base_mass = False
        added_mass_range = [-5.0, 5.0]
        randomize_base_com = False
        base_com_range = [-0.015, 0.015]
        randomize_base_inertia =False
        base_inertia_range = [-0.0005, 0.0005] #### 

        ### motor
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = False
        motor_offset_range = [-0.05, 0.05]
        randomize_Kp_factor = False
        Kp_factor_range = [0.9, 1.1]
        randomize_Kd_factor = False
        Kd_factor_range = [0.9, 1.1]
        randomize_motor_delay = False
        motor_delay_range = 2 ### 4 torque_calc_step 相当于时延0.02s


        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 1.0
        # max_push_vel_xy = 1.0
        # max_push_vel_z = 0.5


    class external_forces:
        enable_external_force = True

        foot_max_force_xy = [-100, 100]  ### N
        foot_max_force_z = [-0.0, 0.0]

        max_force_xy = [-200, 200]  ### N
        max_force_z = [-200, 0.0]

        # max_force_xy = [-200, 200]  ### N
        # max_force_z = [-200, 0.0]

        # max_force_xy = [-50, 50]  ### N
        # max_force_z = [-50, 0.0]

        min_duration = 2.0   ### s
        max_duration = 10.0   ### s

        # max_change_per_step = 200 ## N per step , dt = 0.02


    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True
        # only_positive_rewards = False  ### TODO
        base_height_target = 0.70 ### TODO
        base_pitch_target = -0.1 ### TODO
        tracking_sigma = 0.25
        max_contact_force = 400. # forces above this value are penalized
        default_gap = 0.22  #0.28
        class scales( LeggedRobotCfg.rewards.scales ):

            tracking_lin_vel = 2.0
            tracking_ang_vel = 2.0

            alive = 2.0  ### 0.15
            lin_vel_z = -1.0
            ang_vel_xy = -0.5   ###-0.1
            orientation = -1.0


            stand_normal = -0.01 #-0.05 too high
            base_height = -1.0  ## -10.0
            # dof_acc = -2.5e-7  ### -2.5e-7
            # dof_vel = -1e-5   ### -1e-3
            action_rate = -5e-3  ### -0.005
            action_smoothness  = -1e-2              

            dof_pos_limits = -10.0 ### -5.0
            # dof_vel_limits = -0.1
            torque_limits = -1.0 

            # feet_contact_forces = -5e-4  ### 1e-3
            # contact_no_vel = -0.1  ## -0.2

            torques = -6e-7
            #feet_contact_slip = -0.1
            feet_distance = -0.1
            

            # collision = -5.0

            feet_swing_height = -20.0  # -20.0

            # contact = 0.1 
            contact = 1.0 


            hip_pos = -1.0  ### -1.0
            waist_pos = -1.0  ###  -1.0
            arm_pos = -1.0

            # hip_pos = -2.0  ### -1.0
            # waist_pos = -2.0  ###  -1.0
            # arm_pos = -2.0

            
            # stand_still = -1.0
            


### ----------  symmetry  ----------------------------
# obs_symmetry = [    -0.0001, 1, -2,\
#                     3, -4, 5,\
#                     6, -7, -8,\
                    
#                     15, -16, -17, 18, 19, -20,\
#                     9, -10, -11, 12, 13, -14,\
#                     -21,\
#                     29, -30, -31, 32, -33, 34, -35,\
#                     22, -23, -24, 25, -26, 27, -28,\
                    
#                     42, -43, -44, 45, 46, -47,\
#                     36, -37, -38, 39, 40, -41,\
#                     -48,\
#                     56, -57, -58, 59, -60, 61, -62,\
#                     49, -50, -51, 52, -53, 54, -55,\
                    
#                     69, -70, -71, 72, 73, -74,\
#                     63, -64, -65, 66, 67, -68,\
#                     -75,\
#                     83, -84, -85, 86, -87, 88, -89,\
#                     76, -77, -78, 79, -80, 81, -82,
                    
#                     -90, -91
#                     ]

act_symmetry = [    6, -7, -8, 9, 10, -11, \
                    0.0001, -1, -2, 3, 4, -5, \
                    -12, \
                    20, -21, -22, 23, -24, 25, -26, \
                    13, -14, -15, 16, -17, 18, -19  ]

# # sym_coef = 0.1
# sym_coef = 1.0


class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    seed = -1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2 ### normal
        entropy_coef = 0.001 ### normal

        num_learning_epochs = 5 ## normal
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99 ### normal ,for g1 0.99 work
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        save_interval = 50 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'G1_PPO'
        max_iterations = 10000
        # num_steps_per_env = 24
        num_steps_per_env = 32
        # num_steps_per_env = 64  ### 这个数字对于加了 历史信息的  需要调整 with gamma
        # retrain_load_run = 'reload'
        # retrain_checkpoint = 1000

        resume = True
        load_run = 'play'
        checkpoint = 10000  #3501


class G1RoughCfgPPOEMLP( LeggedRobotCfgPPO ):
    seed = -1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2 ### normal
        entropy_coef = 0.001 ### normal

        num_learning_epochs = 5 ## normal
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99 ### normal ,for g1 0.99 work
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticSymmetry'
        algorithm_class_name = 'PPO'
        save_interval = 50 # check for potential saves every this many iterations # TODO:
        run_name = ''
        experiment_name = 'G1_PPO_EMLP'
        max_iterations = 10000
        # num_steps_per_env = 24
        num_steps_per_env = 32
        # num_steps_per_env = 64  ### 这个数字对于加了 历史信息的  需要调整 with gamma
        # retrain_load_run = 'reload'
        # retrain_checkpoint = 1000

        resume = True
        load_run = 'play'
        checkpoint = 9950    #4300



class G1RoughCfgPPOHalf_symmetry( LeggedRobotCfgPPO ):
    seed = -1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2 ### normal
        entropy_coef = 0.001 ### normal

        num_learning_epochs = 5 ## normal
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99 ### normal ,for g1 0.99 work
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticHalfSymmetry'
        algorithm_class_name = 'PPO'
        save_interval = 50 # check for potential saves every this many iterations # TODO:
        run_name = ''
        experiment_name = 'G1_PPO_Half_symmetry'
        max_iterations = 10000
        # num_steps_per_env = 24
        num_steps_per_env = 32
        # num_steps_per_env = 64  ### 这个数字对于加了 历史信息的  需要调整 with gamma
        # retrain_load_run = 'reload'
        # retrain_checkpoint = 1000

        resume = False
        load_run = 'play'
        checkpoint = 4600
 


        
 