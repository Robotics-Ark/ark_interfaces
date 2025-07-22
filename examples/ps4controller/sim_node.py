from ark.client.comm_infrastructure.base_node import main
from ark.system.simulation.simulator_node import SimulatorNode
from ark.tools.log import log
import os
from pathlib import Path

# Path to the global configuration file used to initialize the simulation environment
CONFIG_PATH = Path(__file__).parent / "config/global_config.yaml"


class PyBulletNode(SimulatorNode):
    """
    PyBulletNode is a simulator node implementation that interfaces with the PyBullet physics engine.
    Inherits from SimulatorNode to provide structure for initializing and stepping through a simulated scene.
    """

    def initialize_scene(self):
        """
        Set up the initial scene in the PyBullet simulation environment.

        This method should load models, configure the environment, and prepare
        all necessary components required before simulation steps begin.
        """
        pass

    def step(self):
        """
        Perform one step of the simulation.

        This method should advance the simulation by one timestep, updating the state
        of all relevant entities and handling any physics or logic updates.
        """
        pass


if __name__ == "__main__":
    main(PyBulletNode, CONFIG_PATH)
