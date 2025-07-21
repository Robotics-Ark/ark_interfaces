from ark.env.ark_env import ArkEnv
from arktypes import joint_group_command_t, joint_state_t, rgbd_t
from arktypes.utils import pack, unpack
import random


class FrankaEnv(ArkEnv):
    """
    FrankaEnv defines an environment interface for the Franka robot using the ARK framework.

    It supports simulation or real-world operation and provides observation and action channel
    packing/unpacking, reward logic, episode termination logic, and object reset functionality.
    """

    def __init__(self, config=None, sim=True):
        """
        Initializes the Franka environment.

        Args:
            config (dict, optional): Global configuration dictionary for environment setup.
            sim (bool, optional): If True, runs in simulation mode. If False, runs with a real robot.
        """
        environment_name = "Franka_Enviroment"
        action_space = {
            "Franka/joint_group_command/sim": joint_group_command_t,
        }
        observation_space = {
            "Franka/joint_states/sim": joint_state_t,
            "camera/rgbd/sim": rgbd_t,
            "gripper_camera/rgbd/sim": rgbd_t,
        }

        super().__init__(
            environment_name=environment_name,
            action_channels=action_space,
            observation_channels=observation_space,
            global_config=config,
            sim=sim,
        )

    def action_packing(self, action):
        """
        Converts a raw action into a format suitable for communication via LCM.

        Args:
            action (list or array): The raw joint command.

        Returns:
            dict: Dictionary containing the packed joint command message.
        """
        return {
            "Franka/joint_group_command/sim": pack.joint_group_command(
                name="all",
                cmd=action,
            ),
        }

    def observation_unpacking(self, observation):
        """
        Unpacks the observation data from incoming messages.

        Args:
            observation (dict): Raw observation data received from the environment.

        Returns:
            dict: Dictionary with parsed joint states and RGB-D data from both fixed and gripper cameras.
        """
        joint_states = unpack.joint_state(observation["Franka/joint_states/sim"])
        camera_rgb, camera_depth = unpack.rgbd(observation["camera/rgbd/sim"])
        gripper_camera_rgb, gripper_camera_depth = unpack.rgbd(
            observation["gripper_camera/rgbd/sim"]
        )

        return {
            "joint_states": joint_states,
            "camera_rgb": camera_rgb,
            "camera_depth": camera_depth,
            "gripper_camera_rgb": gripper_camera_rgb,
            "gripper_camera_depth": gripper_camera_depth,
        }

    def terminated_truncated_info(self, state, action, next_state):
        """
        Determines if the current episode should terminate or be truncated.

        Args:
            state (dict): Current state of the environment.
            action (list): Action taken at the current step.
            next_state (dict): Resulting state after taking the action.

        Returns:
            tuple: (terminated (bool), truncated (bool), info (dict or None))
        """
        # Logic can be customized based on task-specific conditions
        return False, False, None

    def reward(self, state, action, next_state):
        """
        Computes the reward for a transition from state to next_state using the given action.

        Args:
            state (dict): Current state of the environment.
            action (list): Action taken.
            next_state (dict): Resulting state after taking the action.

        Returns:
            float: Reward value.
        """
        # Placeholder reward logic; override in subclasses or extensions
        return 0.0

    def reset_objects(self):
        """
        Resets objects in the environment to randomized starting positions.

        This is typically called at the beginning of each episode.
        """
        print("Resetting objects in the Franka environment.")
        self.reset_component("Franka")

        self.reset_component(
            "BluePlate",
            base_position=[random.uniform(-0.2, -0.15), random.uniform(0.1, 0.2), 0.7],
        )
        self.reset_component(
            "Mug",
            base_position=[
                random.uniform(-0.2, -0.15),
                random.uniform(-0.2, -0.1),
                0.7,
            ],
        )
