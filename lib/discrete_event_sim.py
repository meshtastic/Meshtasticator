import logging

# probably not necessary, but "Environment" seemed too generic to me
from simpy import Environment as SimpyEnvironment
import numpy as np

from lib.common import setup_asymmetric_links
from lib.config import Config
from lib.discrete_event import BroadcastPipe
from lib.gui import Graph, run_graph_updates
from lib.node import MeshNode
from lib.packet import MeshPacket

logger = logging.getLogger(__name__)

class SimulationState:
    """Class to hold all global mutated state of a simulation, not including
    node-specific state such as the position of a moving node.
    """
    def __init__(self, conf: Config, env: SimpyEnvironment):
        """Constructor

        Arguments:
        conf -- Config object of global sim constants. Only used for NR_NODES.
        env -- SimPy Environment for simulation. Required for internal BroadcastPipe.
        """
        self.env = env
        self.bc_pipe = BroadcastPipe(self.env)
        self.packets = [] # used mostly for data tracking, but also for state
        self.packetsAtN = [[] for _ in range(conf.NR_NODES)]
        self.messageSeq = {"val": 0} # TODO: turn this into a locked counter

class SimulationDataTracking:
    """Class to hold data used to monitor a simulation which has no
    impact on the state or progress of the simulation
    """
    def __init__(self):
        self.messages = []
        self.delays = []
        self.totalPairs = 0
        self.symmetricLinks = 0
        self.asymmetricLinks = 0
        self.noLinks = 0

class SimulationResults:
    """Class to hold simulation result data. Any interesting or relevant
    statistic/data from a simulation should wind up in here. Reporting
    functions can take this object and present a report to the user however
    makes sense.

    Mostly a dictionary with extra features.
    """
    def __init__(self, results: dict):
        """Constructor. Start off results with first-order results.

        Arguments:
        results -- dictionary of first-order results from simulation. MANY keys are assumed to exist!
        """
        self.results = results.copy() # only a shallow copy

    def __getitem__(self, subscript: str):
        """Implement subscript access to support `results_object['datapoint']`.
        Very thin wrapper to index into interior dictionary, allowing Exceptions
        to bubble up to the caller.
        """
        return self.results[subscript]

    def finalize(self, conf: Config, nodes: [MeshNode], packets: [MeshPacket]):
        """Once simulation is finished, calculate any second-order
        data that is generally useful, such as averages. This requires some extra
        state-related info.

        All calculated rates are left as the 'raw' ratio. As in, 50% is 0.5,
        100% is 1, etc. If you want percentages you should scale & round the
        rate however you prefer.

        Arguments:
        conf -- Config object. Simulation config.
        nodes -- list of nodes from simulation.
        packets -- list of packets sent during simulation.
        """
        # replicate result enrichment/calculation from loraMesh.py and batchSim.py
        sent = len(self.results["packets"])
        if conf.DMs:
            self.results["potentialReceivers"] = sent
        else:
            self.results["potentialReceivers"] = sent * (conf.NR_NODES - 1)
        self.results["sent"] = sent

        # TODO: inefficient. Have nodes keep counters for these and just collect them
        self.results["nrCollisions"] = sum([1 for p in packets for n in nodes if p.collidedAtN[n.nodeid] is True])
        self.results["nrSensed"] = sum([1 for p in packets for n in nodes if p.sensedByN[n.nodeid] is True])
        self.results["nrReceived"] = sum([1 for p in packets for n in nodes if p.receivedAtN[n.nodeid] is True])
        self.results["nrUseful"] = sum([n.usefulPackets for n in nodes])

        self.results["meanDelay"] = np.nanmean(self.results["delays"])

        # various division-by-0 guarded calculations
        if conf.NR_NODES != 0 and conf.SIMTIME != 0:
            self.results["txAirUtilizationRate"] = sum([n.txAirUtilization for n in nodes])/conf.NR_NODES/conf.SIMTIME
        else:
            self.results["txAirUtilizationRate"] = np.nan

        if self.results["nrSensed"] != 0:
            self.results["collisionRate"] = self.results["nrCollisions"]/self.results["nrSensed"]
        else:
            self.results["collisionRate"] = np.nan

        if self.results["messageSeq"]["val"] != 0 and conf.NR_NODES - 1 != 0:
            self.results["nodeReach"] = self.results["nrUseful"]/(self.results["messageSeq"]["val"]*(conf.NR_NODES-1))
        else:
            self.results["nodeReach"] = np.nan

        if self.results["nrReceived"] != 0:
            usefulness = self.results["nrUseful"]/self.results["nrReceived"]  # nr of packets that delivered to a packet to a new receiver out of all packets sent
            self.results["usefulness"] = usefulness
        else:
            self.results["usefulness"] = np.nan

        self.results["delayDropped"] = sum(n.droppedByDelay for n in nodes)

        if conf.MODEL_ASYMMETRIC_LINKS and self.results["totalPairs"] != 0:
            asymmetricLinkRate = self.results["asymmetricLinks"] / self.results["totalPairs"]
            symmetricLinkRate = self.results["symmetricLinks"] / self.results["totalPairs"]
            noLinkRate = self.results["noLinks"] / self.results["totalPairs"]
            self.results["asymmetricLinkRate"] = asymmetricLinkRate
            self.results["symmetricLinkRate"] = symmetricLinkRate
            self.results["noLinkRate"] = noLinkRate

        if conf.MOVEMENT_ENABLED:
            self.results["movingNodes"] = sum([1 for n in nodes if n.isMoving is True])
            self.results["gpsEnabled"] = sum([1 for n in nodes if n.gpsEnabled is True])

class DiscreteEventSim:
    """Class for a full Discrete Event Simulation. Contains
    simulation config, all necessary state, and sim plumbing.
    """

    # TODO: once our PR for #48 is merged we'll have a NodeConfig class we can accept, rather than a list of dictionaries
    def __init__(self, conf: Config, node_configs: [] = [], graph: Graph | None = None):
        """Constructor.

        Arguments:
        conf -- Config object defining global constants for simulation.
        node_configs -- Output of parse_params. List of node configurations. Default [].
        graph -- Optional Graph object for GUI. If provided GUI will be used. Default None, for no GUI.
        """

        # set constant state/initial state from parameters
        self.env = SimpyEnvironment()
        self.conf = conf
        self.node_configs = node_configs

        # internal global state which changes
        self.mutated_state = SimulationState(self.conf, self.env)

        # nodes are our actors, so should be separate from our global mutating sim state.
        self.nodes = []

        # stats & data tracking
        self.data_tracking = SimulationDataTracking()

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
                self.data_tracking.messages,
                self.mutated_state.packetsAtN,
                self.mutated_state.packets,
                self.data_tracking.delays,
                self.node_configs[i],
                self.mutated_state.messageSeq
            )
            self.nodes.append(node)
            # fun trick, this works too: graph.add_node(node) if graph else None
            if self.graph is not None:
                self.graph.add_node(node)

        # setup that requires having nodes
        self.data_tracking.totalPairs, self.data_tracking.symmetricLinks, self.data_tracking.asymmetricLinks, self.data_tracking.noLinks = setup_asymmetric_links(self.conf, self.nodes)

        if self.graph is not None and self.conf.MOVEMENT_ENABLED:
            # NOTE: this does not run under test, since we skip creating a GUI
            # TODO: revisit this design decision sometime. Do we want graphing/GUI to be handled in this object,
            # or by some external object the user wires in, like how batchSim.py adds in the simulation_progress process?
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

    def get_results(self) -> SimulationResults:
        # TODO: is it possible to add a check that the sim has finished running?
        # first-order stats/data collection
        first_order_results = {
            "packets": self.mutated_state.packets,
            "packetsAtN": self.mutated_state.packetsAtN,
            "messageSeq": self.mutated_state.messageSeq,
            "messages": self.data_tracking.messages,
            "delays": self.data_tracking.delays,
            "totalPairs": self.data_tracking.totalPairs,
            "symmetricLinks": self.data_tracking.symmetricLinks,
            "asymmetricLinks": self.data_tracking.asymmetricLinks,
            "noLinks": self.data_tracking.noLinks,
            "nodes": self.nodes,
        }
        results = SimulationResults(first_order_results)
        results.finalize(self.conf, self.nodes, self.mutated_state.packets)

        return results

