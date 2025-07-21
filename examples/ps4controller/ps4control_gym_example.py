import os
import pickle
from scripts.franka_env import FrankaEnv
from nodes.ps4policy import ExpertPolicyPS4
from pathlib import Path

# Indicates whether to run in simulation mode or on the real robot
SIM = True  # Set to False if you want to use the real robot

# Path to the global configuration YAML file
CONFIG_PATH = Path(__file__).parent / "config/global_config.yaml"

# Directory where collected trajectory data will be saved
DATA_SAVE_PATH = Path(__file__).parent / "data/"

# Number of trajectories to collect
NUM_TRAJ = 50


def main():
    """
    Main function to collect robot demonstration trajectories using a PS4 controller.

    This script sets up a Franka environment (either simulation or real) and a PS4 controller-based
    expert policy interface. It collects a predefined number of trajectories based on user interaction
    through the controller and saves each trajectory as a pickle file.

    Each trajectory consists of joint states, end-effector positions and orientations, RGB and depth images
    from both a fixed and gripper-mounted camera.

    Key steps:
    - Create data directory if it doesn't exist
    - Initialize environment and PS4 controller policy
    - Wait for user to press the "Cross (X)" button to start each trajectory
    - Collect data while the rollout is ongoing
    - Press the "PS Button" to end the rollout and save the trajectory
    """
    if not os.path.exists(DATA_SAVE_PATH):
        os.makedirs(DATA_SAVE_PATH)

    # Initialize the Franka environment
    env = FrankaEnv(sim=SIM, config=CONFIG_PATH)

    # Initialize PS4 controller policy
    ps4policy = ExpertPolicyPS4(node_name="ps4policy", global_config=CONFIG_PATH)

    for i in range(NUM_TRAJ):
        print(f"Collecting trajectory {i}...")
        trajectory = []

        # Reset environment and policy
        obs, _ = env.reset()
        ps4policy.reset()

        start_controller = False

        # Save initial observation
        trajectory.append(
            {
                "joint_states": obs["joint_states"][2],
                "eef_position": ps4policy.initial_ee_position,
                "eef_orientation": ps4policy.initial_ee_orientation,
                "camera_rgb": obs["camera_rgb"],
                "camera_depth": obs["camera_depth"],
                "gripper_camera_rgb": obs["gripper_camera_rgb"],
                "gripper_camera_depth": obs["gripper_camera_depth"],
            }
        )

        # Wait for user to press Cross (X) button
        while start_controller is False:
            print(f"Press Cross on the PS4 controller to start...")
            start_controller = ps4policy.button_states["Cross (X)"]
            ps4policy.unlocked = True

        print("Controller started, collecting data...")
        finished_rollout = False

        # Get initial joint configuration and format it
        q = obs["joint_states"][2].tolist()  # joint 1-7, finger 1-2
        q = q[:7] + [0] + q[-2:] + [0] * 2  # Add placeholders for extra joints

        while not finished_rollout:
            # Get next action from PS4 controller
            new_q, gripper_position, save_transition = ps4policy.get_action(q)
            action = new_q[:7] + [gripper_position]  # Format for env.step

            # Step the environment
            obs, _, _, _, _ = env.step(action)

            # Update joint configuration
            q = obs["joint_states"][2].tolist()
            q = q[:7] + [0] + q[-2:] + [0] * 2

            # Compute new end-effector pose
            new_ee_position, new_ee_orientation = ps4policy.forward_kinematics(q)

            # Save transition if flagged
            if save_transition:
                trajectory.append(
                    {
                        "joint_states": obs["joint_states"][2],
                        "eef_position": new_ee_position,
                        "eef_orientation": new_ee_orientation,
                        "camera_rgb": obs["camera_rgb"],
                        "camera_depth": obs["camera_depth"],
                        "gripper_camera_rgb": obs["gripper_camera_rgb"],
                        "gripper_camera_depth": obs["gripper_camera_depth"],
                    }
                )

            # If PS Button is pressed, end and save trajectory
            if ps4policy.button_states["PS Button"]:
                print(f"PS Button pressed, saving trajectory.")
                finished_rollout = True
                with open(
                    os.path.join(DATA_SAVE_PATH, f"trajectory_{i}.pkl"), "wb"
                ) as f:
                    pickle.dump(trajectory, f)


if __name__ == "__main__":
    main()
