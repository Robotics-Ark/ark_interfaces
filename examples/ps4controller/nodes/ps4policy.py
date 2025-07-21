import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from arktypes import ps4_controller_state_t
from ark.tools.log import log
from ark.client.comm_infrastructure.instance_node import InstanceNode

URDF_PATH = "../../../ark_robots/ark_franka/franka_panda/panda_with_gripper.urdf"
EE_IDX = 11


class ExpertPolicyPS4(InstanceNode):
    """
    ExpertPolicyPS4 translates PS4 controller inputs into motion commands for a Franka Panda robot
    using inverse kinematics and a PyBullet simulation. It is primarily used for data collection
    and teleoperation.

    Attributes:
        client (BulletClient): PyBullet client instance.
        robot_id (int): Identifier for the loaded robot in the PyBullet environment.
        last_q (list): Most recent joint configuration.
        home_position (list): Default joint configuration.
        damping (list): Damping coefficients for inverse kinematics.
        unlocked (bool): Whether controller input is currently allowed to affect the robot.
        button_states (dict): Mapping of PS4 buttons to their current pressed state.
        axis_states (dict): Mapping of PS4 joystick/trigger axes to their values.
        lower_limits, upper_limits, joint_ranges (list): Joint limits and ranges for IK solving.
    """

    def __init__(self, node_name: str, global_config=None):
        """
        Initializes the ExpertPolicyPS4 node.

        Args:
            node_name (str): Name of the node.
            global_config (dict or None): Optional global configuration dictionary.
        """
        super().__init__(node_name, global_config)
        self.client = self._connect_pybullet()

        self.robot_id = self.client.loadURDF(
            URDF_PATH,
            basePosition=[-0.55, 0, 0.6],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
        )

        self.get_joint_limits()

        self.home_position = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0, 0, 0, 0]
        self.damping = [10, 10, 10, 10, 10, 10, 10, 0.1, 0.1, 0.1, 0.1, 10]
        self.last_q = self.home_position

        self.initial_ee_position, self.initial_ee_orientation = self.forward_kinematics(
            self.home_position
        )

        self.unlocked = False

        self.button_states = {
            "Cross (X)": False,
            "Circle (O)": False,
            "Triangle": False,
            "Square": False,
            "L1": False,
            "R1": False,
            "Share": False,
            "Options": False,
            "L3 (Left Stick Click)": False,
            "R3 (Right Stick Click)": False,
            "PS Button": False,
            "Touchpad Click": False,
        }

        self.axis_states = {
            "Left Stick - Horizontal": 0,
            "Left Stick - Vertical": 0,
            "Right Stick - Horizontal": 0,
            "Right Stick - Vertical": 0,
            "L2 Trigger": 0,
            "R2 Trigger": 0,
            "D-Pad Horizontal": 0,
            "D-Pad Vertical": 0,
        }

        self.create_subscriber(
            "ps4_controller", ps4_controller_state_t, self.callback_controller
        )

    def _connect_pybullet(self):
        """
        Establishes a PyBullet client connection in DIRECT mode (no GUI).

        Returns:
            BulletClient: The connected PyBullet client.
        """
        connection_mode_str = "DIRECT"
        connection_mode = getattr(p, connection_mode_str)
        return BulletClient(connection_mode)

    def get_joint_limits(self):
        """
        Retrieves and stores the lower/upper joint limits and joint ranges from the robot model.
        """
        num_joints = p.getNumJoints(self.robot_id)
        self.lower_limits = []
        self.upper_limits = []
        self.joint_ranges = []

        for joint_index in range(num_joints):
            joint_info = self.client.getJointInfo(self.robot_id, joint_index)
            self.lower_limits.append(joint_info[8])
            self.upper_limits.append(joint_info[9])
            self.joint_ranges.append(joint_info[9] - joint_info[8])

    def callback_controller(self, t, channel_name, msg):
        """
        Callback function to update internal button and axis states from the PS4 controller input.

        Args:
            t (float): Timestamp of the message.
            channel_name (str): Channel name.
            msg (ps4_controller_state_t): PS4 controller message.
        """
        self.button_states["Cross (X)"] = msg.cross_x == 1
        self.button_states["Circle (O)"] = msg.circle_o == 1
        self.button_states["Triangle"] = msg.triangle == 1
        self.button_states["Square"] = msg.square == 1
        self.button_states["L1"] = msg.l1 == 1
        self.button_states["R1"] = msg.r1 == 1
        self.button_states["Share"] = msg.share == 1
        self.button_states["Options"] = msg.options == 1
        self.button_states["L3 (Left Stick Click)"] = msg.l3_left_stick_click == 1
        self.button_states["R3 (Right Stick Click)"] = msg.r3_right_stick_click == 1
        self.button_states["PS Button"] = msg.ps_button == 1
        self.button_states["Touchpad Click"] = msg.touchpad_click == 1

        self.axis_states["Left Stick - Horizontal"] = msg.left_stick_horizontal
        self.axis_states["Left Stick - Vertical"] = msg.left_stick_vertical
        self.axis_states["Right Stick - Horizontal"] = msg.right_stick_horizontal
        self.axis_states["Right Stick - Vertical"] = msg.right_stick_vertical
        self.axis_states["L2 Trigger"] = msg.l2_trigger
        self.axis_states["R2 Trigger"] = msg.r2_trigger
        self.axis_states["D-Pad Horizontal"] = msg.dpad_horizontal
        self.axis_states["D-Pad Vertical"] = msg.dpad_vertical

    def forward_kinematics(self, q):
        """
        Computes forward kinematics to determine the end-effector position and orientation.

        Args:
            q (list): Joint configuration.

        Returns:
            tuple: End-effector position (np.array) and orientation as quaternion (np.array).
        """
        for joint_index, pos in enumerate(q):
            self.client.resetJointState(self.robot_id, joint_index, pos)

        link_states = self.client.getLinkState(self.robot_id, EE_IDX)
        return np.array(link_states[4]), np.array(link_states[5])

    def inverse_kinematics(self, ee_pos, ee_orn):
        """
        Solves inverse kinematics to obtain joint positions that reach a target pose.

        Args:
            ee_pos (list or np.ndarray): Target end-effector position.
            ee_orn (list or np.ndarray): Target orientation as quaternion.

        Returns:
            list: Joint angles computed via IK.
        """
        if not isinstance(ee_pos, list):
            ee_pos = ee_pos.tolist()
        if not isinstance(ee_orn, list):
            ee_orn = ee_orn.tolist()

        return p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=EE_IDX,
            targetPosition=ee_pos,
            targetOrientation=ee_orn,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.last_q,
            solver=p.IK_DLS,
            jointDamping=self.damping,
        )

    def get_action(self, q):
        """
        Computes the next joint configuration based on current controller input.

        Args:
            q (list): Current joint configuration.

        Returns:
            tuple:
                new_q (list): Updated joint configuration.
                q_gripper (float): Gripper position from trigger input.
                save_transition (bool): Whether the action should be logged as a data point.
        """
        v_x = -self.axis_states["D-Pad Vertical"]
        v_y = -self.axis_states["D-Pad Horizontal"]

        v_z = (
            1.0
            if self.button_states["L1"]
            else -1.0 if self.button_states["R1"] else 0.0
        )

        v_roll = -(self.axis_states["Right Stick - Vertical"] - 128) / 128
        v_pitch = -(self.axis_states["Right Stick - Horizontal"] - 128) / 128
        v_yaw = -(self.axis_states["Left Stick - Horizontal"] - 128) / 128

        v_roll = 0 if abs(v_roll) < 0.03 else v_roll
        v_pitch = 0 if abs(v_pitch) < 0.03 else v_pitch
        v_yaw = 0 if abs(v_yaw) < 0.03 else v_yaw

        q_gripper = self.axis_states["R2 Trigger"] / 255

        is_moving = any([v_x, v_y, v_z, v_roll, v_pitch, v_yaw, q_gripper])

        if is_moving:
            save_transition = self.unlocked
            ee_pos, ee_orn = self.forward_kinematics(q)
            update_ee_position = np.array(ee_pos) + np.array([v_x, v_y, v_z]) * 0.2

            current_rot = R.from_quat(ee_orn).as_matrix()
            delta_rot = R.from_euler(
                "xyz", [v_roll * 0.1, v_pitch * 0.1, v_yaw * 0.1]
            ).as_matrix()
            update_rot = current_rot @ delta_rot
            update_ee_orientation = R.from_matrix(update_rot).as_quat()

            new_q = list(
                self.inverse_kinematics(
                    update_ee_position.tolist(), update_ee_orientation
                )
            )
            self.last_q = new_q
        else:
            save_transition = False
            new_q = self.last_q

        return new_q, q_gripper, save_transition

    def reset(self):
        """
        Resets internal state to prepare for a new rollout.
        """
        self.unlocked = False
        self.initial_ee_position, self.initial_ee_orientation = self.forward_kinematics(
            self.home_position
        )
        self.last_q = self.home_position
