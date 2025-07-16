import os
import pickle
from scripts.franka_env import FrankaEnv
from nodes.ps4policy import ExpertPolicyPS4

SIM = True  # Set to False if you want to use the real robot
CONFIG = 'config/global_config.yaml'
DATA_SAVE_PATH = 'data/'  # Path to save the collected data

if not os.path.exists(DATA_SAVE_PATH):
    os.makedirs(DATA_SAVE_PATH)

# Set the number of trajectories you want to collect
num_traj = 2
# Initialize the Franka environment
env = FrankaEnv(sim=SIM, config=CONFIG)
# Initialize PS4 controller policy
ps4policy = ExpertPolicyPS4(node_name="ps4policy", global_config=CONFIG) 

for i in range(num_traj):
    print(f"Collecting trajectory {i}...")
    trajectory = []
    obs, _ = env.reset()
    ps4policy.reset()
    start_controller = False
    trajectory.append({'joint_states': obs['joint_states'][2],
                       'eef_position': ps4policy.initial_ee_position,
                       'eef_orientation': ps4policy.initial_ee_orientation, 
                       'camera_rgb': obs['camera_rgb'],
                       'camera_depth': obs['camera_depth'],
                       'gripper_camera_rgb': obs['gripper_camera_rgb'],
                       'gripper_camera_depth': obs['gripper_camera_depth']
                       }
                      )
    
    while start_controller is False:
        print(f"Press Cross ❌ on the PS4 controller to start...")
        start_controller = ps4policy.button_states['Cross (X)']
        ps4policy.unlocked = True

    # After pressing Cross (X), the controller is unlocked and ready to collect data
    print("Controller started, collecting data...")
    finished_rollout = False

    # Get the intial state from the environment
    q = obs['joint_states'][2].tolist() ## joint 1-7, finger 1-2
    q  = q[:7] + [0] + q[-2:] + [0] * 2 ## joint 1-7, hand joint, finger 1-2, camera joint, tcp joint

    while finished_rollout is False:
        # Get the action from the PS4 controller
        new_q, gripper_position, save_transition = ps4policy.get_action(q)
        # Execute the action in the environment, defined in the franka.yaml
        action = new_q[:7] + [gripper_position] 
        obs, _, _, _, _ = env.step(action)
        ## Get new end-effector position and orientation
        q = obs['joint_states'][2].tolist()
        q  = q[:7] + [0] + q[-2:] + [0] * 2
        new_ee_position, new_ee_orientation = ps4policy.forward_kinematics(q)
        if save_transition:
            trajectory.append({'joint_states': obs['joint_states'][2],
                       'eef_position': new_ee_position,
                       'eef_orientation': new_ee_orientation, 
                       'camera_rgb': obs['camera_rgb'],
                       'camera_depth': obs['camera_depth'],
                       'gripper_camera_rgb': obs['gripper_camera_rgb'],
                       'gripper_camera_depth': obs['gripper_camera_depth']
                       }
                      )

        if ps4policy.button_states["PS Button"]:
            print(f"PS 🎮 Button pressed, saving trajectory.")
            finished_rollout = True
            with open(os.path.join(DATA_SAVE_PATH, f'trajectory_{i}.pkl'), 'wb') as f:
                pickle.dump(trajectory, f)
            