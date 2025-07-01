from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
    
    class commands:
        curriculum = True
        max_curriculum_lin_vel_x = 1.
        max_curriculum_lin_vel_y = 1.
        max_curriculum_ang_vel_yaw = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            
            
    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.

        # armature = 0.
        armature = 0.01

        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        push_interval = None
        max_push_vel_xy = 0.5

    class rewards:
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized 缩放 urdf的 
        soft_dof_vel_limit = 1. ### _reward_dof_vel_limits 用到
        soft_torque_limit = 1. ### 没用到
        base_height_target = 1. ### 惩罚 高度
        
        max_contact_force = 100. # forces above this value are penalized
        
        class scales:
            termination = -0.0 ### # Terminal reward / penalty
   


    class normalization:
        
        class command_scales:
            lin_vel_x = 1.0   # min max [m/s]
            lin_vel_y = 1.0   # min max [m/s]
            ang_vel_yaw = 1.0    # min max [rad/s]
        
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 2.0
            dof_vel = 0.2
        
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        
        noise_level = 1.0 # scales other values
        class noise_scales:  ### zhen dui de shi  obs de noise , zhe ge zhi cheng yi obs scale tian jia dao obs zhong 
            lin_vel = 0.2
            ang_vel = 0.2
            dof_pos = 0.02
            dof_vel = 0.5
            gravity = 0.05

    
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [-1, -1, 0.5]  # [m]
        lookat = [1., 1, 0.]  # [m]

    class sim:
        # dt =  0.005
        dt =  0.001

        substeps = 1  ### zy change , orign=1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10   ### CPU core number  , zy change , orign=10
            solver_type = 1  # 0: pgs, 1: tgs ,  1 better
            num_position_iterations = 4  # large better
            num_velocity_iterations = 0 # 0 better
            contact_offset = 0.005  # [m] ###  zy change  orign=0.01
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            friction_offset_threshold = 0.005  ## zy add


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt