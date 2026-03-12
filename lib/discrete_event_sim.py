import logging

# probably not necessary, but "Environment" seemed too generic to me
from simpy import Environment as SimpyEnvironment

from lib.common import setup_asymmetric_links
from lib.config import Config
from lib.discrete_event import BroadcastPipe
from lib.gui import Graph, run_graph_updates
from lib.node import MeshNode

logger = logging.getLogger(__name__)

class SimulationState:
    """Class to hold all global mutated state of a simulation, not including
    node-specific state such as the position of a moving node.
    """
    def __init__(self, config: Config, env: SimpyEnvironment):
        """Constructor

        Arguments:
        config -- Config object of global sim constants. Only used for NR_NODES.
        env -- SimPy Environment for simulation. Required for internal BroadcastPipe.
        """
        self.env = env
        self.bc_pipe = BroadcastPipe(self.env)
        self.packets = [] # used mostly for data tracking, but also for state
        self.packetsAtN = [[] for _ in range(config.NR_NODES)]
        self.messageSeq = {"val": 0} # TODO: turn this into a locked counter

class SimulationDataTracking:
    """Class to hold data used to monitor a simulation which has no
    impact on the state or progress of the simulation
    """
    pass

class SimulationResults:
    """Class to hold simulation result data. Any interesting or relevant
    statistic/data from a simulation should wind up in here. Reporting
    functions can take this object and present a report to the user however
    makes sense.

    Just a glorified dictionary honestly.
    """
    def __init__(self, results: dict):
        self.results = results.copy() # only a shallow copy

    def finalize(self):
        """Once simulation is finished, calculate any second-order
        data that is generally useful, such as averages.
        """
        pass

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

        # set constant state/initial state from parameters
        self.env = SimpyEnvironment()
        self.conf = config
        self.node_configs = node_configs

        # internal global state which changes
        self.mutated_state = SimulationState(self.conf, self.env)

        # nodes are our actors, so should be separate from our global mutating sim state.
        self.nodes = []

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
            node = MeshNode(
                self.conf,
                self.nodes,
                self.env,
                self.mutated_state.bc_pipe,
                i,
                self.conf.PERIOD,
                self.messages,
                self.mutated_state.packetsAtN,
                self.mutated_state.packets,
                self.delays,
                self.node_configs[i],
                self.mutated_state.messageSeq
            )
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
        # first-order stats/data collection
        results = {
            "packets": self.mutated_state.packets,
            "packetsAtN": self.mutated_state.packetsAtN,
            "messageSeq": self.mutated_state.messageSeq,
            "messages": self.messages,
            "delays": self.delays,
            "totalPairs": self.totalPairs,
            "symmetricLinks": self.symmetricLinks,
            "asymmetricLinks": self.asymmetricLinks,
            "noLinks": self.noLinks,
            "nodes": self.nodes,
        }
        # TODO: add some universally useful result calculations, like the
        # ones common between loraMesh.py and batchSim.py
        return results

