from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        env_spacing = 3.  # 如果是 plane 地形， 每个机器人间隔 3m
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        
        num_envs =4096
        num_obs_step = 332
        num_critic_obs = 695
        num_obs_history = 5
        num_actions = 29


    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh

        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 50 # [m] ### max_vel * episode_lens
        static_friction = 0.95
        dynamic_friction = 0.1
        restitution = 0.0  

        # play = True
        play = False

### ---------------    plane ----------------
        slope_max = 0.1 ## 30+ degree
        discrete_height = 0.02
        uniform_terrain_height_max = 0.02
        step_height_max = 0.02  ##0.1
        wave_terrain_amplitude_max = 0.04 ### 0.1 base + 0.15

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # mun=17 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] ### num=11
        
        curriculum = False
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.

        num_rows= 20 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)

        num_sub_terrains = 0 ##  num_rows * num_clos
        terrain_proportions = [0.2, 0.3, 0.2, 0.3]  

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
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
                     'hip_pitch': 200,
                     'hip_roll': 150,
                     'hip_yaw': 150,
                     'knee': 200,
                     'ankle': 100,
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
                     'ankle': 5,
                     'waist': 5,
                     'shoulder': 0.5,
                     'elbow': 0.5,
                     'wrist_roll': 0.5,
                     'wrist_pitch': 0.2,
                     'wrist_yaw': 0.2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_zy.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee", "shoulder_yaw", "elbow", 'torso']
            ### 不能包含 waist，胳膊会碰撞身体，不算reset,只能用肩膀的碰撞， 身体的俯仰角， 身体的高度 判断 reset
        # terminate_after_contacts_on = ["shoulder_pitch"] 
        # terminate_after_contacts_on = ["pelvis", "torso" "shoulder_pitch"]
        terminate_after_contacts_on = ["pelvis", "shoulder_pitch"] 
        # body_name = "waist"
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False  ### NOTE
  
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
            height_measurements = 1.0
        
        clip_observations = 20.0
        clip_actions = 20.0

        
    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:  ### zhen dui de shi  obs de noise , zhe ge zhi cheng yi obs scale tian jia dao obs zhong 
            lin_vel = 0.1
            ang_vel = 0.2
            dof_pos = 0.02
            dof_vel = 0.5
            dof_trq = 1.0
            gravity = 0.05
  

    class domain_rand(LeggedRobotCfg.domain_rand):
        ### env contact porperty
        randomize_friction = True
        friction_range = [1.0, 1.0]
        # friction_range = [0.7, 1.0]

        randomize_base_mass = True
        added_mass_range = [-0.0, 0.0]
        # added_mass_range = [-5.0, 5.0]

        randomize_base_com = True
        base_com_range = [-0.0, 0.0]
        # base_com_range = [-0.015, 0.015]
        
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = False
        motor_offset_range = [-0.05, 0.05]
        randomize_Kp_factor = False
        Kp_factor_range = [0.9, 1.1]
        randomize_Kd_factor = False
        Kd_factor_range = [0.9, 1.1]

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.3


    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True
        # only_positive_rewards = False  ### TODO
        base_height_target = 0.69 ### TODO
        tracking_sigma = 0.25
        max_contact_force = 400. # forces above this value are penalized
        class scales( LeggedRobotCfg.rewards.scales ):

            tracking_base_pos_delta = 10.0
            tracking_base_quat_delta = 5.0
            tracking_dof_pos = 20.0  
            tracking_keypoints_delta = 10.0 
            
            alive = 0.5
            # feet_air_time = 1.0
            # feet_swing_height = -20.0
            # lin_vel_z = -1.0
            # ang_vel_xy = -0.01  ## -0.1 
            # orientation = -1.0
            dof_acc = -2.5e-7  ### -2.5e-7
            dof_vel = -1e-5   ### -1e-3
            action_rate = -1e-2  ### -0.005
            # action_smoothness  = -1e-2              
            dof_pos_limits = -10.0 ### -5.0
            dof_vel_limits = -0.1
            torque_limits = -1.0 
            # feet_contact_forces = -5e-4  ### 1e-3
            # torques = -6e-7
            feet_contact_slip = -0.1  ### -0.1
            # collision = -5.0

            # # hip_pos = -1.0  ### -1.0
            # # waist_pos = -1.0  ###  -1.0
            # # arm_pos = -1.0
            # hip_pos = -2.0  ### -1.0
            # waist_pos = -2.0  ###  -1.0
            # arm_pos = -2.0
            # # stand_still = -1.0




class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    seed = -1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512, 256, 128]
        # actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [1024, 512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2 ### normal
        # clip_param = 0.05 ### normal
        entropy_coef = 0.001 ### normal
        # entropy_coef = 0.002 ### normal
        
        num_learning_epochs = 5 ## normal
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99 ### normal ,for g1 0.99 work
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
        
### ----------  symmetry  ----------------------------
        obs_symmetry = [    -0.0001, 1, -2,\
                            3, -4, 5,\
                            
                            12, -13, -14, 15, 16, -17,\
                            6, -7, -8, 9, 10, -11,\
                            -18, -19, 20,\
                            28, -29, -30, 31, -32, 33, -34,\
                            21, -22, -23, 24, -25, 26, -27,\
                            
                            41, -42, -43, 44, 45, -46,\
                            35, -36, -37, 38, 39, -40,\
                            -47, -48, 49,\
                            57, -58, -59, 60, -61, 62, -63,\
                            50, -51, -52, 53, -54, 55, -56,\
                            
                            70, -71, -72, 73, 74, -75,\
                            64, -65, -66, 67, 68, -69,\
                            -76, -77, 78,\
                            86, -87, -88, 89, -90, 91, -92,\
                            79, -80, -81, 82, -83, 84, -85,\
                                
                            93, -94, 95,\
                            
                            -96, 97, -98, 99,\
                                
                            106, -107, -108, 109, 110, -111,\
                            100, -101, -102, 103, 104, -105,\
                            -112, -113, 114,\
                            122, -123, -124, 125, -126, 127, -128,\
                            115, -116, -117, 118, -119, 120, -121,  
                            ]
        
        ### 按顺序应该是先左腿、右腿、 腰部、 左胳膊、右胳膊,  p_x,p_y, p_z, xyzw
        keypoints_symmetry = [
                                42, -43, 44, -45, 46, -47, 48,
                                49, -50, 51, -52, 53, -54, 55,
                                56, -57, 58, -59, 60, -61, 62,
                                63, -64, 65, -66, 67, -68, 69,
                                70, -71, 72, -73, 74, -75, 76,
                                77, -78, 79, -80, 81, -82, 83,

                                0.0001, -1, 2, -3, 4, -5, 6,
                                7, -8, 9, -10, 11, -12, 13,
                                14, -15, 16, -17, 18, -19, 20,
                                21, -22, 23, -24, 25, -26, 27,
                                28, -29, 30, -31, 32, -33, 34,
                                35, -36, 37, -38, 39, -40, 41,
                                
                                84,  -85,  86,  -87,  88,  -89,  90,
                                91,  -92,  93,  -94,  95,  -96,  97,
                                98,  -99,  100, -101, 102, -103, 104,
                                
                                154, -155, 156, -157, 158, -159, 160,
                                161, -162, 163, -164, 165, -166, 167,
                                168, -169, 170, -171, 172, -173, 174,
                                175, -176, 177, -178, 179, -180, 181,
                                182, -183, 184, -185, 186, -187, 188,
                                189, -190, 191, -192, 193, -194, 195,
                                196, -197, 198, -199, 200, -201, 202,
                                
                                105, -106, 107, -108, 109, -110, 111,
                                112, -113, 114, -115, 116, -117, 118,
                                119, -120, 121, -122, 123, -124, 125,
                                126, -127, 128, -129, 130, -131, 132,
                                133, -134, 135, -136, 137, -138, 139,
                                140, -141, 142, -143, 144, -145, 146,
                                147, -148, 149, -150, 151, -152, 153,
                            ]
        
        
        
        
        traj_symmetry = [   0, -1, 2,\
                            -3, 4, -5, 6,\
                            13, -14, -15, 16, 17, -18,\
                            7, -8, -9, 10, 11, -12,\
                            -19, -20, 21,
                            29, -30, -31, 32, -33, 34, -35,\
                            22, -23, -24, 25, -26, 27, -28,\
                        ]
        
  
        act_symmetry = [    6, -7, -8, 9, 10, -11, \
                            0.0001, -1, -2, 3, 4, -5, \
                            -12, -13, 14,\
                            22, -23, -24, 25, -26, 27, -28, \
                            15, -16, -17, 18, -19, 20, -21  ]
        
        # sym_coef = 1.0
        sym_coef = 5.0
        

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        save_interval = 50 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'G1_PPO'
        max_iterations = 100000
        # num_steps_per_env = 24
        num_steps_per_env = 32
        # num_steps_per_env = 64  ### 这个数字对于加了 历史信息的  需要调整 with gamma
        retrain_load_run = 'reload'
        retrain_checkpoint = 2250

        resume = False
        load_run = 'play'
        checkpoint = 1650



class G1RoughCfgPPOEMLP( LeggedRobotCfgPPO ):
    seed = -1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512, 256, 128]
        # actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [1024, 512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2 ### normal
        # clip_param = 0.05 ### normal
        entropy_coef = 0.001 ### normal
        # entropy_coef = 0.002 ### normal
        
        num_learning_epochs = 5 ## normal
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99 ### normal ,for g1 0.99 work
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticSymmetry' # 使用这个具有对称等变性的actor critic的实现
        algorithm_class_name = 'PPO'
        save_interval = 50 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'G1_EMLP_actor_symm_only'
        max_iterations = 100000
        # num_steps_per_env = 24
        num_steps_per_env = 32
        # num_steps_per_env = 64  ### 这个数字对于加了 历史信息的  需要调整 with gamma
        retrain_load_run = 'reload'
        retrain_checkpoint = 2250

        resume = False
        load_run = 'play'
        checkpoint = 1650
