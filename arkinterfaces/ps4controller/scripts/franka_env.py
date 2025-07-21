from ark.env.ark_env import ArkEnv
from arktypes import joint_group_command_t, joint_state_t, rgbd_t
from arktypes.utils import pack, unpack
import random


class FrankaEnv(ArkEnv):
    """
    Franka Environment for the Franka robot.
    This environment is designed to work with the Franka robot in a simulation or real-world setting.
    It inherits from ArkEnv, which provides the basic structure and functionality for ARK environments.
    """

    def __init__(self, config=None, sim=True):
        """
        Initialize the FrankaEnv environment.

        Args:
            config (dict, optional): Configuration dictionary for the environment. 
                This may include parameters such as control frequency, robot setup, etc.
            sim (bool, optional): Whether the environment is running in simulation. 
                If False, assumes real-world deployment. Defaults to True.

        """
        environment_name = "Franka_Enviroment"
        action_space = {
            "Franka/joint_group_command/sim": joint_group_command_t,
        }
        observation_space = {
            "Franka/joint_states/sim": joint_state_t,  # Assuming joint states are also packed in a similar way
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
        Pack the action into a message format suitable for LCM.

        :param action: The action to be packed.
        :return: A dictionary containing the packed action.
        """
        return {
            "Franka/joint_group_command/sim": pack.joint_group_command(
                name="all",
                cmd=action,
            ),
        }

    def observation_unpacking(self, observation):
        """
        Unpack the observation from the message format.

        :param observation: The observation to be unpacked.
        :return: The unpacked observation.
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
        Check if the episode is terminated or truncated.

        :param observation: The current observation.
        :return: A tuple containing termination status, truncation status, and additional info.
        """
        # Implement your logic to determine if the episode is terminated or truncated
        return False, False, None

    def reward(self, state, action, next_state):
        """
        Calculate the reward based on the current state, action, and next state.

        :param state: The current state.
        :param action: The action taken.
        :param next_state: The next state after taking the action.
        :return: The calculated reward.
        """
        # Implement your reward calculation logic here
        return 0.0

    def reset_objects(self):
        """
        Reset the objects in the environment.
        This method is called when the environment is reset.
        """
        # Implement your logic to reset objects in the environment
        print("Resetting objects in the Franka environment.")
        self.reset_component("Franka")
        # Randomly reset the positions of the objects
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
