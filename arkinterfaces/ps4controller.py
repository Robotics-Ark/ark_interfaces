import pprint

import numpy as np
from evdev import InputDevice, categorize, ecodes, list_devices

from ark.client.comm_infrastructure.base_node import BaseNode, main
from arktypes import string_t, ps4_controller_state_t, float_t, flag_t
from ark.tools.log import log
from typing import Dict, Any, Optional
from pathlib import Path


class PS4Controller(BaseNode):
    """
    PS4Controller reads events from a connected PS4 controller (via USB or Bluetooth) and publishes
    its state using LCM messages at a fixed frequency.

    It handles button states, axis positions, and ensures that noisy initial activations are filtered
    out using a lock mechanism before data is broadcast.

    Attributes:
        controller (InputDevice): The detected PS4 controller device.
        button_map (dict): Mapping from evdev key codes to button labels.
        axis_map (dict): Mapping from evdev axis codes to axis labels.
        button_states (dict): Current state of each PS4 button (pressed or not).
        axis_states (dict): Current value of each joystick and trigger axis.
        lock (bool): Whether controller input is currently ignored due to high initial activations.
        pub (Publisher): Publisher for sending controller state as `ps4_controller_state_t`.
    """

    def __init__(self, node_name: str, global_config=None):
        """
        Initializes the PS4Controller node, detects the connected controller,
        maps inputs, and sets up publishers and steppers.

        Args:
            node_name (str): Name of the node instance.
            global_config (dict or None): Optional configuration parameters.
        """
        super().__init__(node_name, global_config)

        self.controller = self.find_controller()
        if self.controller is None:
            log.error(
                "No controller detected! Make sure it is connected via Bluetooth or USB."
            )
            exit()

        log.ok(f"Connected to: {self.controller.name} at {self.controller.path}")

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
            0x14A: "Touchpad Click",
        }

        self.axis_map = {
            ecodes.ABS_X: "Left Stick - Horizontal",
            ecodes.ABS_Y: "Left Stick - Vertical",
            ecodes.ABS_RX: "Right Stick - Horizontal",
            ecodes.ABS_RY: "Right Stick - Vertical",
            ecodes.ABS_Z: "L2 Trigger",
            ecodes.ABS_RZ: "R2 Trigger",
            ecodes.ABS_HAT0X: "D-Pad Horizontal",
            ecodes.ABS_HAT0Y: "D-Pad Vertical",
        }

        self.button_states = {name: False for name in self.button_map.values()}
        self.axis_states = {name: 0 for name in self.axis_map.values()}

        self.lock = True  # Ignore events until all activations are low

        self.pub = self.create_publisher("ps4_controller", ps4_controller_state_t)

        self.create_stepper(120, self.event_callback)  # Event reader
        self.create_stepper(60, self.step)  # State publisher

    def find_controller(self):
        """
        Scans for connected input devices and returns the PS4 controller if found.

        Returns:
            InputDevice or None: The connected PS4 controller, or None if not found.
        """
        devices = [InputDevice(path) for path in list_devices()]
        print("Scanning available input devices...")
        for device in devices:
            print(f" {device.path} - {device.name}")
            if "Wireless Controller" in device.name or "Sony" in device.name:
                return device
        return None

    def pack(self):
        """
        Packs the current button and axis states into an `ps4_controller_state_t` LCM message.

        Returns:
            ps4_controller_state_t: The packed controller state message.
        """
        msg = ps4_controller_state_t()

        msg.cross_x = int(self.button_states["Cross (X)"])
        msg.circle_o = int(self.button_states["Circle (O)"])
        msg.triangle = int(self.button_states["Triangle"])
        msg.square = int(self.button_states["Square"])
        msg.l1 = int(self.button_states["L1"])
        msg.r1 = int(self.button_states["R1"])
        msg.share = int(self.button_states["Share"])
        msg.options = int(self.button_states["Options"])
        msg.l3_left_stick_click = int(self.button_states["L3 (Left Stick Click)"])
        msg.r3_right_stick_click = int(self.button_states["R3 (Right Stick Click)"])
        msg.ps_button = int(self.button_states["PS Button"])
        msg.touchpad_click = int(self.button_states["Touchpad Click"])

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
        """
        Periodically publishes the controller state as an LCM message.

        It ensures all axes are at rest (low activations) before unlocking the device.
        """
        if self.lock:
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
            if np.max(np.abs(activations)) >= 10:
                log.warn(
                    "High initial activations detected, controller is locked. Please move the joysticks and triggers."
                )
                return
            else:
                self.lock = False
                return

        msg = self.pack()
        self.pub.publish(msg)
        log.info(f"Published controller state")

    def event_callback(self):
        """
        Reads raw events from the controller in a non-blocking loop.
        Categorizes each event and dispatches to the appropriate handler.
        """
        for event in self.controller.read_loop():
            if event.type == ecodes.EV_KEY:
                self.handle_button(event)
            elif event.type == ecodes.EV_ABS:
                self.handle_axis(event)
            elif event.type == ecodes.EV_SYN:
                continue  # Ignore synchronization events

    def handle_button(self, event):
        """
        Updates button state based on key press/release event.

        Args:
            event (InputEvent): Event object containing button press info.
        """
        button_name = self.button_map.get(event.code, event.code)
        self.button_states[button_name] = bool(event.value)

    def handle_axis(self, event):
        """
        Updates axis state based on joystick or trigger movement.

        Args:
            event (InputEvent): Event object containing axis value.
        """
        axis_name = self.axis_map.get(event.code, event.code)
        self.axis_states[axis_name] = event.value


if __name__ == "__main__":
    CONFIG_PATH = (
        Path(__file__).parent.parent
        / "examples/ps4controller/config/global_config.yaml"
    )
    main(PS4Controller, "ps4_controller", CONFIG_PATH)
