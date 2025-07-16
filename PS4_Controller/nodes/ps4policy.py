import numpy as np  
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation as R

from arktypes import ps4_controller_state_t
from ark.tools.log import log
from ark.client.comm_infrastructure.instance_node import InstanceNode

URDF_PATH = "../../ark_robots/ark_franka/franka_panda/panda_with_gripper.urdf"
EE_IDX = 11

class ExpertPolicyPS4(InstanceNode):
    
    def __init__(self, node_name: str, global_config=None):
        
        super().__init__(node_name, global_config)
        self.client = self._connect_pybullet()
        ## check the baseposition here
        self.robot_id = self.client.loadURDF(URDF_PATH, basePosition=[-0.55, 0, 0.6], baseOrientation=[0,0,0,1], useFixedBase=True) ## remove merge links
        self.get_joint_limits()        
        self.home_position = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0, 0, 0, 0]
        self.damping = [10, 10, 10, 10, 10, 10, 10, 0.1, 0.1, 0.1, 0.1, 10]
        self.last_q = self.home_position
        self.initial_ee_position, self.initial_ee_orientation = self.forward_kinematics(self.home_position) # orientation returns quaternion x y z w

        self.unlocked = False
        
        self.button_states = {
            'Cross (X)': False,
            'Circle (O)': False,
            'Triangle': False,
            'Square': False,
            'L1': False,
            'R1': False,
            'Share': False,
            'Options': False,
            'L3 (Left Stick Click)': False,
            'R3 (Right Stick Click)': False,
            'PS Button': False,
            'Touchpad Click': False
        }
        self.axis_states = {
            'Left Stick - Horizontal': 0,   # left [0, 255] right, neutral at 128
            'Left Stick - Vertical': 0,     # up [0, 255] down, neutral at 128
            'Right Stick - Horizontal': 0,  # left [0, 255] right, neutral at 128
            'Right Stick - Vertical': 0,    # up [0, 255] down, neutral at 128
            'L2 Trigger': 0,
            'R2 Trigger': 0,
            'D-Pad Horizontal': 0,
            'D-Pad Vertical': 0
        }
        
        # listen to controller commands
        self.create_subscriber("ps4_controller", ps4_controller_state_t, self.callback_controller)


    def _connect_pybullet(self):   
        connection_mode_str = "DIRECT"
        connection_mode = getattr(p, connection_mode_str)
        return BulletClient(connection_mode) 
        

    def get_joint_limits(self):
        num_joints = p.getNumJoints(self.robot_id)
        self.lower_limits = []
        self.upper_limits = []
        self.joint_ranges = []
        # Loop over each joint and print its limits
        for joint_index in range(num_joints):
            joint_info = self.client.getJointInfo(self.robot_id, joint_index)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            self.lower_limits.append(lower_limit)
            self.upper_limits.append(upper_limit)
            self.joint_ranges.append(upper_limit - lower_limit)

    def get_data_store_decision(self) -> bool:
        while (self.button_states['Square'] is False) and (self.button_states['Circle (O)'] is False):
            log.warn("Do you want to store the last rollout? Press Square to store or O to dismiss.")
        
        if self.button_states['Square']:
            return True
        elif self.button_states['Circle (O)']:
            return False
        
        
    def callback_controller(self, t, channel_name, msg):
        
        self.button_states['Cross (X)'] = msg.cross_x == 1
        self.button_states['Circle (O)'] = msg.circle_o == 1
        self.button_states['Triangle'] = msg.triangle == 1
        self.button_states['Square'] = msg.square == 1
        self.button_states['L1'] = msg.l1 == 1
        self.button_states['R1'] = msg.r1 == 1
        self.button_states['Share'] = msg.share == 1
        self.button_states['Options'] = msg.options == 1
        self.button_states['L3 (Left Stick Click)'] = msg.l3_left_stick_click == 1
        self.button_states['R3 (Right Stick Click)'] = msg.r3_right_stick_click == 1
        self.button_states['PS Button'] = msg.ps_button == 1
        self.button_states['Touchpad Click'] = msg.touchpad_click == 1

        self.axis_states['Left Stick - Horizontal'] = msg.left_stick_horizontal
        self.axis_states['Left Stick - Vertical'] = msg.left_stick_vertical
        self.axis_states['Right Stick - Horizontal'] = msg.right_stick_horizontal
        self.axis_states['Right Stick - Vertical'] = msg.right_stick_vertical
        self.axis_states['L2 Trigger'] = msg.l2_trigger
        self.axis_states['R2 Trigger'] = msg.r2_trigger
        self.axis_states['D-Pad Horizontal'] = msg.dpad_horizontal
        self.axis_states['D-Pad Vertical'] = msg.dpad_vertical
        
    
    def forward_kinematics(self, q):
        for joint_index, pos in enumerate(q):
            self.client.resetJointState(self.robot_id, joint_index, pos)

        link_states = self.client.getLinkState(self.robot_id, EE_IDX)
        end_effector_position = np.array(link_states[4])
        end_effector_orientation = np.array(link_states[5])

        return end_effector_position, end_effector_orientation
    
    
    def inverse_kinematics(self, ee_pos, ee_orn):
        # check if lists
        if not isinstance(ee_pos, list):
            ee_pos = ee_pos.tolist()
        if not isinstance(ee_orn, list):
            ee_orn = ee_orn.tolist()
        q_new = p.calculateInverseKinematics(bodyUniqueId=self.robot_id, 
                                             endEffectorLinkIndex=EE_IDX, 
                                             targetPosition=ee_pos, 
                                             targetOrientation=ee_orn, 
                                             lowerLimits=self.lower_limits,
                                             upperLimits=self.upper_limits,
                                             jointRanges=self.joint_ranges,
                                             restPoses=self.last_q,   
                                             solver=p.IK_DLS,
                                             jointDamping=self.damping)
        return q_new
    
    
    def get_action(self, q):
        """
        You can custiomize the action based on the PS4 controller input here for the best performance.
        The corresponding buttons are defined in the ps4controller.py file.
        """   
        # linear velocity
        v_x = -self.axis_states['D-Pad Vertical']
        v_y = -self.axis_states['D-Pad Horizontal']

        if self.button_states['L1']:
            v_z = 1.0
        elif self.button_states['R1']:
            v_z = -1.0
        else:
            v_z = 0.0
        
        # angular velocity
        v_roll = -(self.axis_states['Right Stick - Vertical'] - 128) / 128 
        if np.abs(v_roll) < 0.03:
            v_roll = 0
        
        v_pitch = -(self.axis_states['Right Stick - Horizontal'] - 128) / 128 
        if np.abs(v_pitch) < 0.03:
            v_pitch = 0
        
        v_yaw = -(self.axis_states['Left Stick - Horizontal'] - 128) / 128 
        if np.abs(v_yaw) < 0.03:
            v_yaw = 0
        
        # gripper position
        q_gripper = self.axis_states['R2 Trigger'] / 255

        # if any button or joy sticker was touched
        if (v_x != 0.0 or v_y != 0.0) or (v_z != 0.0) or (v_roll != 0.0 and v_pitch != 0.0) or (v_yaw != 0.0) or (q_gripper != 0.0):
            if self.unlocked:
                save_transition = True
            else:
                save_transition = False
        
            end_effector_position, end_effector_orientation = self.forward_kinematics(q)
            ## Relative position with respect to the world frame
            update_ee_position = np.array(end_effector_position) + np.array([v_x * 0.2, v_y * 0.2, v_z * 0.2])
            ## Relative rotation with respect to the end effector orientation
            current_ee_orientation_matrix = R.from_quat(end_effector_orientation).as_matrix()
            delta_ee_rotation = R.from_euler('xyz', [v_roll * 0.05, v_pitch * 0.05, v_yaw * 0.05]).as_matrix()
            update_ee_orientation_matrix = current_ee_orientation_matrix @ delta_ee_rotation
            update_ee_orientation = R.from_matrix(update_ee_orientation_matrix).as_quat()
            ## Conver tuple to list
            new_q = list(self.inverse_kinematics(update_ee_position.tolist(), update_ee_orientation))
            self.last_q = new_q

        else:
            save_transition = False # to avoid idle statees
            new_q = self.last_q

        return new_q, q_gripper, save_transition

    def reset(self):
        self.unlocked = False
        self.initial_ee_position, self.initial_ee_orientation = self.forward_kinematics(self.home_position) # orientation returns quaternion x y z w
        self.last_q = self.home_position