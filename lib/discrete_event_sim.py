import logging

from simpy import Environment as SimpyEnvironment

from lib.common import setup_asymmetric_links
from lib.config import Config
from lib.discrete_event import BroadcastPipe
from lib.gui import Graph, run_graph_updates
from lib.node import MeshNode

logger = logging.getLogger(__name__)

class SimulationResults:
    """Class to hold simulation result data. Any interesting or relevant
    statistic/data from a simulation should wind up in here. Reporting
    functions can take this object and present a report to the user however
    makes sense.

    Just a glorified dictionary honestly.
    """
    def __init__(self, results: dict):
        self.results = results.copy() # only a shallow copy

class DiscreteEventSim:
    """Class for a full Discrete Event Simulation. Contains
    simulation config, all necessary state, and sim plumbing.
    """

    # TODO: once our PR for #48 is merged we'll have a NodeConfig class we can accept, rather than obscuring all node placement/creation in this class
    def __init__(self, config: Config, node_configs: [] = [], graph: Graph | None = None):
        """Constructor.

        Arguments:
        config -- Config object defining global constants for simulation.
        node_configs -- Output of parse_params. List of node configurations. Default [].
        graph -- Optional Graph object for GUI. If provided GUI will be used. Default None, for no GUI.
        """
        # InteractiveSim takes argparse output as the single parameter here, tightly
        # coupling CLI arguments to this class. This works but try with a looser
        # coupling first, so that parameters can be translated/normalized first,
        # and then only relevant ones passed to the constructor.

        # set state from parameters
        self.env = SimpyEnvironment()
        self.conf = config
        self.node_configs = node_configs
        self.nodes = []

        # internal state
        self.bc_pipe = BroadcastPipe(self.env)
        self.packets = []
        self.packetsAtN = [[] for _ in range(self.conf.NR_NODES)]
        self.messageSeq = {"val": 0}

        # stats & data tracking
        self.messages = []
        self.delays = []
        self.totalPairs = 0
        self.symmetricLinks = 0
        self.asymmetricLinks = 0
        self.noLinks = 0

        # note: we allow user to specify if graphing will happen or not
        self.graph = graph

        # create nodes once we have the various things they have to be wired into
        for i in range(self.conf.NR_NODES):
            node = MeshNode(self.conf, self.nodes, self.env, self.bc_pipe, i, self.conf.PERIOD, self.messages, self.packetsAtN, self.packets, self.delays, self.node_configs[i], self.messageSeq)
            self.nodes.append(node)
            # fun trick, this works too: graph.add_node(node) if graph else None
            if self.graph is not None:
                self.graph.add_node(node)

        # setup that requires having nodes
        self.totalPairs, self.symmetricLinks, self.asymmetricLinks, self.noLinks = setup_asymmetric_links(self.conf, self.nodes)

        if self.graph is not None and self.conf.MOVEMENT_ENABLED:
            # NOTE: this does not run under test, since we skip creating a GUI
            # TODO: batchSim does this, but without the 4th parameter
            self.env.process(run_graph_updates(self.env, self.graph, self.nodes, self.conf.ONE_MIN_INTERVAL))
        self.conf.update_router_dependencies()

    def run_simulation(self):
        self.env.run(until=self.conf.SIMTIME)

    def get_env(self) -> SimpyEnvironment:
        """get a reference to the Sim's SimPy Environment.
        Useful for adding your own processes to the environment.
        Originally a hack to support batchSim.py, which has a progress
        tracking process
        """
        return self.env

    # just return a dictionary for now, refactor into an object later
    def get_results(self) -> {}:
        results = {
            "packets": self.packets,
            "packetsAtN": self.packetsAtN,
            "messageSeq": self.messageSeq,
            "messages": self.messages,
            "delays": self.delays,
            "totalPairs": self.totalPairs,
            "symmetricLinks": self.symmetricLinks,
            "asymmetricLinks": self.asymmetricLinks,
            "noLinks": self.noLinks,
            "nodes": self.nodes,
        }
        return results

