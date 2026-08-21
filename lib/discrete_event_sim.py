import copy
import logging
from typing import TYPE_CHECKING

# probably not necessary, but "Environment" seemed too generic to me
from simpy import Environment as SimpyEnvironment
import numpy as np

from lib.common import setup_asymmetric_links
from lib.config import Config
from lib.discrete_event_sim_components import SimulationState, SimulationDataTracking
from lib.node import MeshNode, NodeConfig

if TYPE_CHECKING:
    from lib.gui import Graph
from lib.packet import MeshPacket
from lib.phy import estimate_path_loss

logger = logging.getLogger(__name__)

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

    def finalize(self, conf: Config):
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
        nodes = self.results["nodes"]
        packets = self.results["packets"]
        sent = len(packets)
        if conf.DMs:
            self.results["potentialReceivers"] = sent
        else:
            self.results["potentialReceivers"] = sent * (conf.NR_NODES - 1)
        self.results["sent"] = sent

        # TODO: inefficient. Have nodes keep counters for these and just collect them
        self.results["nrCollisions"] = sum([1 for p in packets for n in nodes if p.collidedAtN[n.nodeid] is True])
        self.results["nrSensed"] = sum([1 for p in packets for n in nodes if p.sensedByN[n.nodeid] is True])
        self.results["nrReceived"] = sum([1 for p in packets for n in nodes if p.receivedAtN[n.nodeid] is True])
        self.results["nrPhyLoss"] = sum([
            1
            for p in packets
            for n in nodes
            if n.nodeid < len(getattr(p, "phyLostAtN", []))
            and p.phyLostAtN[n.nodeid] is True
            and p.sensedByN[n.nodeid] is True
            and p.collidedAtN[n.nodeid] is False
        ])
        collision_reasons = {}
        for p in packets:
            for reason in getattr(p, "collisionReasonAtN", []):
                if reason:
                    collision_reasons[reason] = collision_reasons.get(reason, 0) + 1
        self.results["collisionReasons"] = collision_reasons
        terrain_losses = [
            p.terrainLossAtN[n.nodeid]
            for p in packets
            for n in nodes
            if n.nodeid < len(getattr(p, "terrainLossAtN", [])) and p.terrainLossAtN[n.nodeid] > 0
        ]
        self.results["meanTerrainLossDb"] = float(np.nanmean(terrain_losses)) if terrain_losses else 0.0
        self.results["maxTerrainLossDb"] = max(terrain_losses) if terrain_losses else 0.0
        clutter_losses = [
            p.clutterLossAtN[n.nodeid]
            for p in packets
            for n in nodes
            if n.nodeid < len(getattr(p, "clutterLossAtN", [])) and p.clutterLossAtN[n.nodeid] > 0
        ]
        self.results["meanClutterLossDb"] = float(np.nanmean(clutter_losses)) if clutter_losses else 0.0
        self.results["maxClutterLossDb"] = max(clutter_losses) if clutter_losses else 0.0
        self.results["nrUseful"] = sum([n.usefulPackets for n in nodes])

        self.results["meanDelay"] = np.nanmean(self.results["delays"]) if self.results["delays"] else np.nan

        # various division-by-0 guarded calculations
        if conf.NR_NODES != 0 and conf.SIMTIME != 0:
            self.results["txAirUtilizationRate"] = sum([n.txAirUtilization for n in nodes])/conf.NR_NODES/conf.SIMTIME
        else:
            self.results["txAirUtilizationRate"] = np.nan

        if self.results["nrSensed"] != 0:
            self.results["collisionRate"] = self.results["nrCollisions"]/self.results["nrSensed"]
        else:
            self.results["collisionRate"] = np.nan

        if self.results["messageSeq"] != 0 and conf.NR_NODES - 1 != 0:
            self.results["nodeReach"] = self.results["nrUseful"]/(self.results["messageSeq"]*(conf.NR_NODES-1))
        else:
            self.results["nodeReach"] = np.nan

        if self.results["nrReceived"] != 0:
            usefulness = self.results["nrUseful"]/self.results["nrReceived"]  # nr of packets that delivered to a packet to a new receiver out of all packets sent
            self.results["usefulness"] = usefulness
        else:
            self.results["usefulness"] = np.nan

        self.results["delayDropped"] = sum(n.droppedByDelay for n in nodes)
        self.results["dcrTxByCr"] = {
            cr: sum(getattr(n, "dcrTxByCr", {}).get(cr, 0) for n in nodes)
            for cr in (5, 6, 7, 8)
        }
        self.results["dcrAirtimeByCr"] = {
            cr: sum(getattr(n, "dcrAirtimeByCr", {}).get(cr, 0.0) for n in nodes)
            for cr in (5, 6, 7, 8)
        }
        dtp_tx_count = sum(getattr(n, "dtpTxCount", 0) for n in nodes)
        self.results["dtpTxByPower"] = {}
        self.results["dtpTxByCrPower"] = {}
        for n in nodes:
            for power, count in getattr(n, "dtpTxByPower", {}).items():
                self.results["dtpTxByPower"][power] = self.results["dtpTxByPower"].get(power, 0) + count
            for cr_power, count in getattr(n, "dtpTxByCrPower", {}).items():
                self.results["dtpTxByCrPower"][cr_power] = self.results["dtpTxByCrPower"].get(cr_power, 0) + count
        self.results["dtpMeanDetectedByTx"] = (
            sum(getattr(n, "dtpDetectedByTx", 0) for n in nodes) / dtp_tx_count if dtp_tx_count else 0.0
        )
        self.results["dtpMeanSensedByTx"] = (
            sum(getattr(n, "dtpSensedByTx", 0) for n in nodes) / dtp_tx_count if dtp_tx_count else 0.0
        )

        if self.results["totalPairs"] != 0:
            noLinkRate = self.results["noLinks"] / self.results["totalPairs"]
            self.results["noLinkRate"] = noLinkRate

        if conf.MOVEMENT_ENABLED:
            self.results["movingNodes"] = sum([1 for n in nodes if n.isMoving is True])
            self.results["gpsEnabled"] = sum([1 for n in nodes if n.gpsEnabled is True])

class DiscreteEventSim:
    """Class for a full Discrete Event Simulation. Contains
    simulation config, all necessary state, and sim plumbing.
    """

    def __init__(self, conf: Config, node_configs: [NodeConfig], graph: "Graph | None" = None):
        """Constructor.

        Arguments:
        conf -- Config object defining global constants for simulation.
        node_configs -- Output of parse_params. List of node configurations. Default [].
        graph -- Optional Graph object for GUI. If provided GUI will be used. Default None, for no GUI.
        """

        # set constant state/initial state from parameters
        self.env = SimpyEnvironment()
        self.conf = copy.deepcopy(conf) # have our own copy so our setup_asymmetric_links doesn't change whatever config we've been passed.
        self.node_configs = node_configs

        # reset MeshPacket class variables
        MeshPacket.seed_asym_rng(self.conf.SEED)
        MeshPacket.reset_packet_counter()

        # internal global state which changes
        self.mutated_state = SimulationState(self.conf, self.env)

        # stats & data tracking
        self.data_tracking = SimulationDataTracking()

        # note: we allow user to specify if graphing will happen or not
        self.graph = graph

        # use node configs to populate the connectivity matrix and compute
        # initial condition links/no links.  Because we always expect link/no
        # link counts, and thus do an O(n^2) precomputation anyways, just
        # always do this and reserve checking/not checking the map later based
        # on config settings.
        setup_asymmetric_links(self.conf, self.node_configs)
        self.initialize_connectivity_map()

        # node configs provided, create nodes with them
        for cfg in self.node_configs:
            n = MeshNode(self.conf,
                self.mutated_state,
                self.data_tracking,
                cfg
            )
            self.mutated_state.nodes.append(n)

        if self.graph is not None:
            for n in self.mutated_state.nodes:
                self.graph.add_node(n)

        logger.debug(f"connectivity map: {self.mutated_state.connectivity_map}")

        if self.graph is not None and self.conf.MOVEMENT_ENABLED:
            # NOTE: this does not run under test, since we skip creating a GUI
            # TODO: revisit this design decision sometime. Do we want graphing/GUI to be handled in this object,
            # or by some external object the user wires in, like how batchSim.py adds in the simulation_progress process?
            # TODO: batchSim does this, but without the 4th parameter
            from lib.gui import run_graph_updates

            self.env.process(run_graph_updates(self.env, self.graph, self.mutated_state.nodes, self.conf.ONE_MIN_INTERVAL))
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

        # expect to use this very soon
        #node_stats = [n.get_stats() for n in self.mutated_state.nodes]

        first_order_results = {
            "packets": self.mutated_state.packets,
            "packetsAtN": self.mutated_state.packetsAtN,
            "messageSeq": self.mutated_state.messageSeq.peek(),
            "messages": self.data_tracking.messages,
            "delays": self.data_tracking.delays,
            "totalPairs": self.data_tracking.totalPairs,
            "noLinks": self.data_tracking.noLinks,
            "nodes": self.mutated_state.nodes,
        }
        results = SimulationResults(first_order_results)
        results.finalize(self.conf)

        return results

    def initialize_connectivity_map(self):
        '''use node configs to compute the initial connectivity map for later
        lookups. Also, initialize baseline path loss matrix.
        '''
        for tx_node in self.node_configs:
            # compute the set of all nodes our signal is detectable at
            reachable_node_set = set()
            for rx_node in self.node_configs:
                if tx_node.node_id == rx_node.node_id:
                    continue # skip self

                self.data_tracking.totalPairs += 1

                (rssi, pl) = tx_node.compute_rssi_and_pathloss_to(rx_node, self.conf)

                # compare with extra margin (set based on 10-node standard test)
                if rssi + self.conf.CONNECTIVITY_MAP_RSSI_MARGIN > self.conf.current_preset['sensitivity']:
                    reachable_node_set.add(rx_node.node_id)

                # compute total/no links without margin
                if rssi >= self.conf.current_preset['sensitivity']:
                    self.data_tracking.totalLinks += 1
                else:
                    self.data_tracking.noLinks += 1

                # cache path loss (it is symmetric, and static until one of the nodes moves)
                self.mutated_state.baseline_pathloss_matrix[tx_node.node_id][rx_node.node_id] = pl
                self.mutated_state.baseline_pathloss_matrix[rx_node.node_id][tx_node.node_id] = pl

            self.mutated_state.connectivity_map[tx_node.node_id] = reachable_node_set
