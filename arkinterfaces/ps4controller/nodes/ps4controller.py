import pprint

import numpy as np
from evdev import InputDevice, categorize, ecodes, list_devices

from ark.client.comm_infrastructure.base_node import BaseNode, main
from arktypes import string_t, ps4_controller_state_t, float_t, flag_t
from ark.tools.log import log
from typing import Dict, Any, Optional


class PS4Controller(BaseNode):

    def __init__(self, node_name: str, global_config=None):
        super().__init__(node_name, global_config)

        # Find the correct controller device
        self.controller = self.find_controller()
        if self.controller is None:
            log.error(
                "No controller detected! Make sure it is connected via Bluetooth or USB."
            )
            exit()

        log.ok(f"🎮 Connected to: {self.controller.name} at {self.controller.path}")

        self.button_map = {
            ecodes.BTN_SOUTH: "Cross (X)",
            ecodes.BTN_EAST: "Circle (O)",
            ecodes.BTN_NORTH: "Triangle",
            ecodes.BTN_WEST: "Square",
            ecodes.BTN_TL: "L1",
            ecodes.BTN_TR: "R1",
            ecodes.BTN_SELECT: "Share",
            ecodes.BTN_START: "Options",
            ecodes.BTN_THUMBL: "L3 (Left Stick Click)",
            ecodes.BTN_THUMBR: "R3 (Right Stick Click)",
            ecodes.BTN_MODE: "PS Button",
            0x14A: "Touchpad Click",  # Custom mapping for touchpad
        }
        self.axis_map = {
            ecodes.ABS_X: "Left Stick - Horizontal",  #  [0, 255], neutral at 128
            ecodes.ABS_Y: "Left Stick - Vertical",
            ecodes.ABS_RX: "Right Stick - Horizontal",
            ecodes.ABS_RY: "Right Stick - Vertical",
            ecodes.ABS_Z: "L2 Trigger",
            ecodes.ABS_RZ: "R2 Trigger",
            ecodes.ABS_HAT0X: "D-Pad Horizontal",
            ecodes.ABS_HAT0Y: "D-Pad Vertical",
        }
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
            "Left Stick - Horizontal": 0,  # left [0, 255] right, neutral at 128
            "Left Stick - Vertical": 0,  # up [0, 255] down, neutral at 128
            "Right Stick - Horizontal": 0,  # left [0, 255] right, neutral at 128
            "Right Stick - Vertical": 0,  # up [0, 255] down, neutral at 128
            "L2 Trigger": 0,
            "R2 Trigger": 0,
            "D-Pad Horizontal": 0,
            "D-Pad Vertical": 0,
        }

        # ! IMPORTANT !
        # in some cases, the controller would initially publish
        # high button activations for some reason.
        # In this implementation, we ignore any input until we observed a state
        # where all the activations are low.
        self.lock = True

        self.pub = self.create_publisher("ps4_controller", ps4_controller_state_t)

        self.create_stepper(120, self.event_callback)  # read controller events
        self.create_stepper(60, self.step)  # publish controller state

    def find_controller(self):
        """Find the PS4 controller dynamically."""
        devices = [InputDevice(path) for path in list_devices()]
        print("🔍 Scanning available input devices...")

        for device in devices:
            print(f" {device.path} - {device.name}")
            if "Wireless Controller" in device.name or "Sony" in device.name:
                return device
        return None

    def pack(self):
        # Initialize the LCM message
        msg = ps4_controller_state_t()

        # Pack button states into the LCM message (booleans will be 0 for False, 1 for True)
        msg.cross_x = 1 if self.button_states["Cross (X)"] else 0
        msg.circle_o = 1 if self.button_states["Circle (O)"] else 0
        msg.triangle = 1 if self.button_states["Triangle"] else 0
        msg.square = 1 if self.button_states["Square"] else 0
        msg.l1 = 1 if self.button_states["L1"] else 0
        msg.r1 = 1 if self.button_states["R1"] else 0
        msg.share = 1 if self.button_states["Share"] else 0
        msg.options = 1 if self.button_states["Options"] else 0
        msg.l3_left_stick_click = (
            1 if self.button_states["L3 (Left Stick Click)"] else 0
        )
        msg.r3_right_stick_click = (
            1 if self.button_states["R3 (Right Stick Click)"] else 0
        )
        msg.ps_button = 1 if self.button_states["PS Button"] else 0
        msg.touchpad_click = 1 if self.button_states["Touchpad Click"] else 0

        # Pack axis states (int16, range from 0 to 255)
        msg.left_stick_horizontal = self.axis_states["Left Stick - Horizontal"]
        msg.left_stick_vertical = self.axis_states["Left Stick - Vertical"]
        msg.right_stick_horizontal = self.axis_states["Right Stick - Horizontal"]
        msg.right_stick_vertical = self.axis_states["Right Stick - Vertical"]
        msg.l2_trigger = self.axis_states["L2 Trigger"]
        msg.r2_trigger = self.axis_states["R2 Trigger"]
        msg.dpad_horizontal = self.axis_states["D-Pad Horizontal"]
        msg.dpad_vertical = self.axis_states["D-Pad Vertical"]

        return msg

    def step(self):

        if self.lock:
            # check if activations are low
            activations = np.array(
                [
                    self.axis_states["Left Stick - Horizontal"] - 128,
                    self.axis_states["Left Stick - Vertical"] - 128,
                    self.axis_states["Right Stick - Horizontal"] - 128,
                    self.axis_states["Right Stick - Vertical"] - 128,
                    self.axis_states["L2 Trigger"],
                    self.axis_states["R2 Trigger"],
                ]
            )
            print(activations)
            activations = np.abs(activations)
            if np.max(np.abs(activations)) >= 10:
                log.warn(
                    "High initial activations detected, controller is locked. Please move the joysticks and triggers."
                )
                return
            # if activations low, unlock the controller
            else:
                self.lock = False
                return
        # Pack the controller state into the LCM message
        msg = self.pack()
        # Publish the message
        self.pub.publish(msg)
        log.info(f"Published controller state")

    def event_callback(self):
        for event in self.controller.read_loop():
            if event.type == ecodes.EV_KEY:  # Button Presses
                self.handle_button(event)
            elif event.type == ecodes.EV_ABS:  # Analog Sticks & Triggers
                self.handle_axis(event)
            elif event.type == ecodes.EV_SYN:
                continue  # Ignore synchronization events

    def handle_button(self, event):
        """Handle button press and release events."""
        self.button_states[self.button_map.get(event.code, event.code)] = bool(
            event.value
        )

    def handle_axis(self, event):
        """Handle analog stick and trigger movements."""
        self.axis_states[self.axis_map.get(event.code, event.code)] = event.value


if __name__ == "__main__":
    main(PS4Controller, "ps4_controller", "./config/global_config.yaml")
