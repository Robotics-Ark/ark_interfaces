from ark.client.comm_infrastructure.base_node import main
from ark.system.simulation.simulator_node import SimulatorNode
from ark.tools.log import log
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / "config/global_config.yaml"
print(CONFIG_PATH)

class PyBulletNode(SimulatorNode):

    def initialize_scene(self):
        pass

    def step(self):
        pass


if __name__ == "__main__":
    main(PyBulletNode, CONFIG_PATH)
