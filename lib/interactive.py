import argparse
import cmd
import socket
import sys
import threading
import time
import yaml
import os

import google.protobuf.json_format as proto
from matplotlib import patches
from meshtastic import tcp_interface, BROADCAST_NUM, mesh_pb2, admin_pb2, telemetry_pb2, portnums_pb2, channel_pb2
from pubsub import pub
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from lib.config import Config
import lib.phy as phy
from lib.common import calc_dist, gen_scenario, find_random_position, Graph

conf = Config()
HW_ID_OFFSET = 16
TCP_PORT_OFFSET = 4403
TCP_PORT_CLIENT = 4402
MAX_TO_FROM_RADIO_SIZE = 512
DEVICE_SIM_DOCKER_IMAGE = "meshtastic/device-simulator"
MESHTASTICD_PATH_DOCKER = "./meshtasticd"


class InteractiveNode:
    def __init__(self, nodes, nodeId, hwId, TCPPort, nodeConfig):
        self.nodeid = nodeId
        if nodeConfig is not None:
            self.x = nodeConfig['x']
            self.y = nodeConfig['y']
            self.z = nodeConfig['z']
            self.isRouter = nodeConfig['isRouter']
            self.isRepeater = nodeConfig['isRepeater']
            self.isClientMute = nodeConfig['isClientMute']
            self.hopLimit = nodeConfig['hopLimit']
            self.antennaGain = nodeConfig['antennaGain']
            self.neighborInfo = nodeConfig['neighborInfo']
        else:
            self.x, self.y = find_random_position(conf, nodes)
            self.z = conf.HM
            self.isRouter = conf.router
            self.isRepeater = False
            self.isClientMute = False
            self.hopLimit = conf.hopLimit
            self.antennaGain = conf.GL
            self.neighborInfo = False
        self.iface = None
        self.hwId = hwId
        self.TCPPort = TCPPort
        self.timestamps = []
        self.channelUtilization = []
        self.airUtilTx = []
        self.numPacketsTx = 0
        self.numPacketsRx = 0
        self.numPacketsRxBad = 0
        self.numRxDupe = 0
        self.numTxRelay = 0
        self.numTxRelayCanceled = 0

    def add_interface(self, iface):
        self.iface = iface

    def set_config(self):
        requiresReboot = False
        # Set a long and short name
        p = admin_pb2.AdminMessage()
        p.set_owner.long_name = "Node "+str(self.nodeid)
        p.set_owner.short_name = str(self.nodeid)
        self.iface.localNode._sendAdmin(p)
        if self.hopLimit != 3:
            loraConfig = self.iface.localNode.localConfig.lora
            setattr(self.iface.localNode.localConfig.lora, 'hop_limit', self.hopLimit)
            p = admin_pb2.AdminMessage()
            p.set_config.lora.CopyFrom(loraConfig)
            self.iface.localNode._sendAdmin(p)

        if self.isRouter:
            requiresReboot = True
            deviceConfig = self.iface.localNode.localConfig.device
            setattr(deviceConfig, 'role', "ROUTER")
            p = admin_pb2.AdminMessage()
            p.set_config.device.CopyFrom(deviceConfig)
            self.iface.localNode._sendAdmin(p)
        elif self.isRepeater:
            requiresReboot = True
            deviceConfig = self.iface.localNode.localConfig.device
            setattr(deviceConfig, 'role', "REPEATER")
            p = admin_pb2.AdminMessage()
            p.set_config.device.CopyFrom(deviceConfig)
            self.iface.localNode._sendAdmin(p)
        elif self.isClientMute:
            requiresReboot = True
            deviceConfig = self.iface.localNode.localConfig.device
            setattr(deviceConfig, 'role', "CLIENT_MUTE")
            p = admin_pb2.AdminMessage()
            p.set_config.device.CopyFrom(deviceConfig)
            self.iface.localNode._sendAdmin(p)

        if self.neighborInfo:
            requiresReboot = True
            moduleConfig = self.iface.localNode.moduleConfig.neighbor_info
            setattr(moduleConfig, 'enabled', 1)
            setattr(moduleConfig, 'update_interval', 30)
            p = admin_pb2.AdminMessage()
            p.set_module_config.neighbor_info.CopyFrom(moduleConfig)
            self.iface.localNode._sendAdmin(p)

        base_lat = 44
        base_lon = -105
        conv_factor = 0.0001
        lat = base_lat + (self.y * conv_factor)
        lon = base_lon + (self.x * conv_factor)
        self.iface.sendPosition(lat, lon, 0)

        return requiresReboot

    def add_admin_channel(self):
        ch = self.iface.localNode.getChannelByChannelIndex(1)
        chs = channel_pb2.ChannelSettings()
        chs.psk = b'\xb0X\xad\xb3\xa5\xd0?$\x8c\x92{\xcd^(\xeb\xb7\x01\x84"\xc9\xf4\x06:\x8d\xfdD#\x08\xe5\xc2\xd7\xdc'
        chs.name = "admin"
        ch.settings.CopyFrom(chs)
        ch.role = channel_pb2.Channel.Role.SECONDARY
        self.iface.localNode.channels[ch.index] = ch
        self.iface.localNode.writeChannel(ch.index)
        time.sleep(1)


class InteractivePacket:

    def __init__(self, packet, id):
        self.packet = packet
        self.localId = id

    def setTxRxs(self, transmitter, receivers):
        self.transmitter = transmitter
        self.receivers = receivers

    def setRSSISNR(self, rssis, snrs):
        self.rssis = rssis
        self.snrs = snrs


class InteractiveGraph(Graph):
    def __init__(self):
        super().__init__(conf)
        self.routes = False

    def init_routes(self, sim):
        if not sim.docker:
            sim.close_nodes()
        if not self.routes:
            self.routes = True
            self.sim = sim
            self.arrows = []
            self.txts = []
            self.annots = []
            self.firstTime = True
            self.defaultHopLimit = conf.hopLimit
            self.fig.subplots_adjust(bottom=0.2)
            axbox = self.fig.add_axes([0.5, 0.04, 0.1, 0.06])
            self.text_box = TextBox(axbox, "Message ID: ", initial="0")
            self.text_box.disconnect("button_press_event")
            self.text_box.on_submit(self.submit)
            self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.fig.canvas.mpl_connect("close_event", self.on_close)
            print("On the scenario plot, enter a message ID to show its route. Close the figure to exit.")
            self.fig.canvas.draw_idle()
            self.fig.canvas.get_tk_widget().focus_set()
            plt.show()
        elif sim.docker:
            sim.close_nodes()

    def clear_route(self):
        for arr in self.arrows.copy():
            arr.remove()
            self.arrows.remove(arr)
        for ann in self.annots.copy():
            ann.remove()
            self.annots.remove(ann)

    def plot_route(self, messageId):
        if self.firstTime:
            print('Hover over an arc to show some info and click to remove it afterwards.')
            print('Close the window to exit the simulator.')
        self.firstTime = False
        packets = [p for p in self.packets if p.localId == messageId]
        if len(packets) > 0:
            self.clear_route()
            style = "Simple, tail_width=0.5, head_width=4, head_length=8"
            pairs = dict.fromkeys(list(set(p.transmitter for p in packets)), [])
            for p in packets:
                tx = p.transmitter
                rxs = p.receivers
                rxCnt = 1
                for ri, rx in enumerate(rxs):
                    # calculate how many packets with the same Tx and Rx we have
                    found = False
                    for pi, rxPair in enumerate(pairs.get(tx)):  # pair is rx.nodeid and its count for this transmitter
                        if rxPair[0] == rx.nodeid:
                            found = True
                            rxCnt = rxPair[1] + 1
                            updated = pairs.get(tx).copy()
                            updated[pi] = (rx.nodeid, rxCnt)
                            pairs.update({tx: updated})
                    if not found:
                        rxCnt = 1
                        pairs.get(tx).append((rx.nodeid, rxCnt))
                    kw = dict(arrowstyle=style, color=plt.cm.Set1(tx.nodeid))
                    rad = str(rxCnt*0.1)  # set the rad to Tx-Rx pair count
                    patch = patches.FancyArrowPatch((tx.x, tx.y), (rx.x, rx.y), connectionstyle="arc3,rad="+rad, **kw)
                    self.ax.add_patch(patch)

                    if int(p.packet["to"]) == BROADCAST_NUM:
                        to = "All"
                    else:
                        to = str(p.packet["to"]-HW_ID_OFFSET)

                    if "hopLimit" in p.packet:
                        hopLimit = p.packet["hopLimit"]
                    else:
                        hopLimit = None

                    if p.packet["from"] == tx.hwId:
                        if "requestId" in p.packet["decoded"]:
                            if p.packet["priority"] == "ACK":
                                msgType = "Real ACK"
                            else:
                                msgType = "Response"
                        else:
                            msgType = "Original message"
                    elif "requestId" in p.packet["decoded"]:
                        if p.packet["decoded"]["simulator"]["portnum"] == "ROUTING_APP":
                            msgType = "Forwarding real ACK"
                        else:
                            msgType = "Forwarding response"
                    else:
                        if int(p.packet['from']) == rx.hwId:
                            msgType = "Implicit ACK"
                        else:
                            if to == "All":
                                msgType = "Rebroadcast"
                            else:
                                msgType = "Forwarding message"

                    fields = []
                    msgTypeField = r"$\bf{"+msgType+"}$"
                    fields.append(msgTypeField)
                    origSenderField = "Original sender: "+str(p.packet["from"]-HW_ID_OFFSET)
                    fields.append(origSenderField)
                    destField = "Destination: "+to
                    fields.append(destField)
                    portNumField = "Portnum: "+str(p.packet["decoded"]["simulator"]["portnum"])
                    fields.append(portNumField)
                    if hopLimit:
                        hopLimitField = "HopLimit: "+str(hopLimit)
                        fields.append(hopLimitField)
                    rssiField = "RSSI: "+str(round(p.rssis[ri], 2)) + " dBm"
                    fields.append(rssiField)
                    table = ""
                    for i, f in enumerate(fields):
                        table += f
                        if i != len(fields)-1:
                            table += "\n"
                    annot = self.ax.annotate(table, xy=((tx.x+rx.x)/2, rx.y+150), bbox=dict(boxstyle="round", fc="w"))
                    annot.get_bbox_patch().set_facecolor(patch.get_facecolor())
                    annot.get_bbox_patch().set_alpha(0.4)
                    annot.set_visible(False)
                    self.arrows.append(patch)
                    self.annots.append(annot)
            self.fig.canvas.draw_idle()
            self.fig.suptitle('Route of message '+str(messageId)+' and ACKs')
        else:
            print('Could not find message ID.')

    def hover(self, event):
        if event.inaxes == self.ax:
            for i, a in enumerate(self.arrows):
                annot = self.annots[i]
                cont, _ = a.contains(event)
                if cont:
                    annot.set_visible(True)
                    self.fig.canvas.draw()
                    break

    def on_click(self, event):
        for annot in self.annots:
            if annot.get_visible():
                annot.set_visible(False)
                self.fig.canvas.draw_idle()

    def on_close(self, event):
        plt.close('all')

    def submit(self, val):
        messageId = int(val)
        self.plot_route(messageId)

    def plot_metrics(self, nodes):
        if any(len(n.timestamps) > 1 for n in nodes):
            plt.figure()
            for n in nodes:
                if len(n.timestamps) > 0:
                    initTime = n.timestamps[0]
                    plt.plot([t-initTime for t in n.timestamps], n.channelUtilization, label=str(n.nodeid), marker=".")
            plt.ylabel('Channel utilization (%)')
            plt.xlabel('Time (s)')
            plt.legend(title='Node ID')
            plt.figure()
            for n in nodes:
                if len(n.timestamps) > 0:
                    initTime = n.timestamps[0]
                    plt.plot([t-initTime for t in n.timestamps], n.airUtilTx, label=str(n.nodeid), marker=".")
            plt.ylabel('Hourly Tx air utilization (%)')
            plt.xlabel('Time (s)')
            plt.legend(title='Node ID')

        if any(n.numPacketsRxBad > 0 for n in nodes):  # Only really interesting if there are bad packets (meaning collisions)
            stats = ['Tx', 'Rx', 'Rx bad', 'Rx dupe', 'Tx relay', 'Tx relay canceled']
            num_stats = len(stats)
            num_nodes = len(nodes)
            x = np.arange(num_stats)
            _, ax = plt.subplots(figsize=(12, 6))
            data = [[n.numPacketsTx for n in nodes], [n.numPacketsRx for n in nodes], [n.numPacketsRxBad for n in nodes], [n.numRxDupe for n in nodes], [n.numTxRelay for n in nodes], [n.numTxRelayCanceled for n in nodes]]
            bar_width = 0.15
            for i in range(num_nodes):
                x_positions = x + (i - (num_nodes / 2)) * bar_width + bar_width / 2
                ax.bar(x_positions, [row[i] for row in data], width=bar_width, label=nodes[i].nodeid)
                ax.set_xticks([])
            ax.set_ylabel('Number of packets')
            ax.set_xticks(x)
            ax.set_xticklabels(stats)
            ax.legend(title='Node ID')
            ax.set_title('Packet statistics')


class InteractiveSim:
    def __init__(self):
        self.messages = []
        self.messageId = -1
        self.nodes = []
        foundNodes = False
        # foundPath = False  # never used
        self.eraseFlash = False
        self.clientConnected = False
        self.forwardSocket = None
        self.clientSocket = None
        self.nodeThread = None
        self.clientThread = None
        self.wantExit = False

        config, pathToProgram = self.parse_interactive_args(foundNodes)
        programWithFullPath = os.path.join(pathToProgram, 'program')

        if not self.docker and not sys.platform.startswith('linux'):
            print("Docker is required for non-Linux OS.")
            self.docker = True

        self.graph = InteractiveGraph()
        for n in range(conf.NR_NODES):
            node = InteractiveNode(self.nodes, n, self.node_id_to_hw_id(n), n + TCP_PORT_OFFSET, config[n])
            self.nodes.append(node)
            self.graph.add_node(node)

        print("Booting nodes...")

        if self.docker:
            try:
                import docker
            except ImportError:
                print("Please install the Docker SDK for Python with 'pip3 install docker'.")
                exit(1)
            n0 = self.nodes[0]
            dockerClient = docker.from_env()
            startNode = f"{MESHTASTICD_PATH_DOCKER} "
            if self.removeConfig:
                startNode += "-e "

            if sys.platform == "darwin":
                self.container = dockerClient.containers.run(
                    DEVICE_SIM_DOCKER_IMAGE,
                    f"{startNode} -d /home/node{n0.nodeid} -h {n0.hwId} -p {n0.TCPPort}",
                    ports=dict(zip((f'{n.TCPPort}/tcp' for n in self.nodes), (n.TCPPort for n in self.nodes))),
                    name="Meshtastic", detach=True, auto_remove=True, user="root"
                )
                for n in self.nodes[1:]:
                    if self.emulateCollisions:
                        time.sleep(2)  # Wait a bit to avoid immediate collisions when starting multiple nodes
                    self.container.exec_run(f"{startNode} -d /home/node{n0.nodeid} -h {n.hwId} -p {n.TCPPort}", detach=True, user="root")
                print(f"Docker container with name {self.container.name} is started.")
            else:
                self.container = dockerClient.containers.run(
                    DEVICE_SIM_DOCKER_IMAGE,
                    f"sh -c '{startNode} -d /home/node{n0.nodeid} -h {n0.hwId} -p {n0.TCPPort} > /home/out_{n0.nodeid}.log'",
                    ports=dict(zip((f'{n.TCPPort}/tcp' for n in self.nodes), (n.TCPPort for n in self.nodes))),
                    name="Meshtastic", detach=True, auto_remove=True, user="root",
                    volumes={"Meshtasticator": {'bind': '/home/', 'mode': 'rw'}}
                )
                for n in self.nodes[1:]:
                    if self.emulateCollisions:
                        time.sleep(2)  # Wait a bit to avoid immediate collisions when starting multiple nodes
                    self.container.exec_run(f"sh -c '{startNode} -d /home/node{n.nodeid} -h {n.hwId} -p {n.TCPPort} > /home/out_{n.nodeid}.log'", detach=True, user="root")
                print(f"Docker container with name {self.container.name} is started.")
                print(f"You can check the device logs using 'docker exec -it {self.container.name} cat /home/out_x.log', where x is the node number.")
        else:
            from shutil import which
            if which('gnome-terminal') is not None:
                xterm = False
            elif which('xterm') is not None:
                xterm = True
            else:
                print('The interactive simulator on native Linux (without Docker) requires either gnome-terminal or xterm.')
                exit(1)
            for n in self.nodes:  # [1:]
                if not xterm:
                    newTerminal = f"gnome-terminal --title='Node {n.nodeid}' -- "
                else:
                    newTerminal = f"xterm -title 'Node {n.nodeid}' -e "
                startNode = f"-d {os.path.expanduser('~')}/.portduino/node{n.nodeid} -h {n.hwId} -p {n.TCPPort}"
                if self.removeConfig:
                    startNode += " -e"
                startNode += " &"
                cmdString = f"{newTerminal} {programWithFullPath} {startNode}"
                os.system(cmdString)
                if self.emulateCollisions and n.nodeid != len(self.nodes)-1:
                    time.sleep(2)  # Wait a bit to avoid immediate collisions when starting multiple nodes

        if self.forwardToClient:
            print(f"Please connect with the client to TCP port {TCP_PORT_CLIENT} ...")
            self.forwardSocket = socket.socket()
            self.forwardSocket.bind(('', TCP_PORT_CLIENT))
            self.forwardSocket.listen()
            (clientSocket, _) = self.forwardSocket.accept()
            self.clientSocket = clientSocket
            iface0 = tcp_interface.TCPInterface(hostname="localhost", portNumber=self.nodes[0].TCPPort, connectNow=False)
            self.nodes[0].add_interface(iface0)
            iface0.myConnect()    # setup socket
            self.nodeThread = threading.Thread(target=self.node_reader, args=(), daemon=True)
            self.clientThread = threading.Thread(target=self.client_reader, args=(), daemon=True)
            self.nodeThread.start()
            self.clientThread.start()
        else:
            time.sleep(4)    # Allow instances to start up their TCP service

        try:
            for n in self.nodes[int(self.forwardToClient):]:
                iface = tcp_interface.TCPInterface(hostname="localhost", portNumber=n.TCPPort)
                n.add_interface(iface)
            if self.forwardToClient:
                self.clientConnected = True
                iface0.localNode.nodeNum = self.nodes[0].hwId
                iface0.connect()  # real connection now
            for n in self.nodes:
                requiresReboot = n.set_config()
                if requiresReboot and self.emulateCollisions and n.nodeid != len(self.nodes)-1:
                    time.sleep(2)  # Wait a bit to avoid immediate collisions when starting multiple nodes
            self.reconnect_nodes()
            pub.subscribe(self.on_receive, "meshtastic.receive.simulator")
            pub.subscribe(self.on_receive_metrics, "meshtastic.receive.telemetry")
            if self.forwardToClient:
                pub.subscribe(self.on_receive_all, "meshtastic.receive")
        except Exception as ex:
            print(f"Error: Could not connect to native program: {ex}")
            self.close_nodes()
            sys.exit(1)

    def parse_interactive_args(self, foundNodes):
        parser = argparse.ArgumentParser(prog='interactiveSim')
        parser.add_argument('nrNodes', type=int, nargs='?', choices=range(0, 11), default=0)
        parser.add_argument('-s', '--script', action='store_true')
        parser.add_argument('-d', '--docker', action='store_true')
        parser.add_argument('--from-file', action='store_true')
        parser.add_argument('-f', '--forward', action='store_true')
        parser.add_argument('-p', '--program', type=str, default=os.getcwd())
        parser.add_argument('-c', '--collisions', action='store_true')
        args = parser.parse_args()
        # print(args)

        self.script = args.script
        self.docker = args.docker
        self.forwardToClient = args.forward
        self.emulateCollisions = args.collisions
        self.removeConfig = not args.from_file
        if args.from_file:
            foundNodes = True
            with open(os.path.join("out", "nodeConfig.yaml"), 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            conf.NR_NODES = len(config.keys())
        elif args.nrNodes > 0:    # nrNodes was specified
            conf.NR_NODES = args.nrNodes
            foundNodes = True
            config = [None for _ in range(conf.NR_NODES)]
        if not foundNodes:
            print("nrNodes was not specified, generating scenario...")
            config = gen_scenario(conf)
            conf.NR_NODES = len(config.keys())
        pathToProgram = args.program
        return config, pathToProgram

    def reconnect_nodes(self):
        time.sleep(3)
        for n in self.nodes[int(self.forwardToClient):]:
            try:
                n.iface.close()
                n.iface = None
            except OSError:
                pass
        time.sleep(5)
        for n in self.nodes:
            while not n.iface:
                try:
                    iface = tcp_interface.TCPInterface(hostname="localhost", portNumber=n.TCPPort)
                    n.add_interface(iface)
                except OSError:
                    print("Trying to reconnect to node...")
                    time.sleep(1)
            if self.emulateCollisions and n.nodeid != len(self.nodes)-1:
                time.sleep(2)  # Wait a bit to avoid immediate collisions when starting multiple nodes

    def forward_packet(self, receivers, packet, rssis, snrs):
        data = packet["decoded"]["payload"]
        if getattr(data, "SerializeToString", None):
            data = data.SerializeToString()

        if len(data) > mesh_pb2.Constants.DATA_PAYLOAD_LEN:
            raise Exception("Data payload too big")

        meshPacket = mesh_pb2.MeshPacket()

        meshPacket.decoded.payload = data
        meshPacket.decoded.portnum = portnums_pb2.SIMULATOR_APP
        meshPacket.to = packet["to"]
        setattr(meshPacket, "from", packet["from"])
        meshPacket.id = packet["id"]
        if "wantAck" in packet:
            meshPacket.want_ack = packet["wantAck"]
        if "hopLimit" in packet:
            meshPacket.hop_limit = packet["hopLimit"]
        if "hopStart" in packet:
            meshPacket.hop_start = packet["hopStart"]
        if "viaMQTT" in packet:
            meshPacket.via_mqtt = packet["viaMQTT"]
        if "requestId" in packet["decoded"]:
            meshPacket.decoded.request_id = packet["decoded"]["requestId"]
        if "wantResponse" in packet["decoded"]:
            meshPacket.decoded.want_response = packet["decoded"]["wantResponse"]
        if "channel" in packet:
            meshPacket.channel = int(packet["channel"])
        for i, rx in enumerate(receivers):
            meshPacket.rx_rssi = int(rssis[i])
            meshPacket.rx_snr = snrs[i]
            toRadio = mesh_pb2.ToRadio()
            toRadio.packet.CopyFrom(meshPacket)
            rx.iface._sendToRadio(toRadio)

    def copy_packet(self, packet):
        # print(packet)
        time.sleep(0.01)
        try:
            if 'simulator' in packet or packet["decoded"]["portnum"] == "SIMULATOR_APP":
                return None

            data = packet["decoded"]["payload"]
            if getattr(data, "SerializeToString", None):
                data = data.SerializeToString()

            meshPacket = mesh_pb2.MeshPacket()
            meshPacket.decoded.payload = data
            meshPacket.decoded.portnum = packet["decoded"]["portnum"]
            meshPacket.to = packet["to"]
            setattr(meshPacket, "from", packet["from"])
            meshPacket.id = packet["id"]
            if "wantAck" in packet:
                meshPacket.want_ack = packet["wantAck"]
            if "hopLimit" in packet:
                meshPacket.hop_limit = packet["hopLimit"]
            if "hopStart" in packet:
                meshPacket.hop_start = packet["hopStart"]
            if "viaMQTT" in packet:
                meshPacket.via_mqtt = packet["viaMQTT"]
            if "requestId" in packet["decoded"]:
                meshPacket.decoded.request_id = packet["decoded"]["requestId"]
            if "wantResponse" in packet["decoded"]:
                meshPacket.decoded.want_response = packet["decoded"]["wantResponse"]
            if "channel" in packet:
                meshPacket.channel = int(packet["channel"])
            fromRadio = mesh_pb2.FromRadio()
            fromRadio.packet.CopyFrom(meshPacket)
            return fromRadio
        except Exception:
            return None

    def show_nodes(self, id=None):
        if id is not None:
            print('NodeDB as seen by node', id)
            self.nodes[id].iface.show_nodes()
        else:
            for n in self.nodes:
                print('NodeDB as seen by node', n.nodeid)
                n.iface.show_nodes()

    def send_broadcast(self, text, fromNode):
        self.get_node_iface_by_id(fromNode).sendText(text)

    def send_dm(self, text, fromNode, toNode):
        self.get_node_iface_by_id(fromNode).sendText(text, destinationId=self.node_id_to_hw_id(toNode), wantAck=True)

    def send_ping(self, fromNode, toNode):
        payload = str.encode("test string")
        self.get_node_iface_by_id(fromNode).sendData(
            payload, destinationId=self.node_id_to_hw_id(toNode),
            portNum=portnums_pb2.PortNum.REPLY_APP,
            wantAck=True, wantResponse=True
        )

    def trace_route(self, fromNode, toNode):
        r = mesh_pb2.RouteDiscovery()
        self.get_node_iface_by_id(fromNode).sendData(r, destinationId=self.node_id_to_hw_id(toNode), portNum=portnums_pb2.PortNum.TRACEROUTE_APP, wantResponse=True)

    def request_position(self, fromNode, toNode):
        self.get_node_iface_by_id(fromNode).sendPosition(destinationId=self.node_id_to_hw_id(toNode), wantResponse=True)

    def request_local_stats(self, toNode):
        r = telemetry_pb2.Telemetry()
        r.local_stats.CopyFrom(telemetry_pb2.LocalStats())
        self.get_node_iface_by_id(toNode).sendData(r, destinationId=self.node_id_to_hw_id(toNode), portNum=portnums_pb2.PortNum.TELEMETRY_APP, wantResponse=True)

    def get_node_iface_by_id(self, id):
        for n in self.nodes:
            if n.hwId == self.node_id_to_hw_id(id):
                return n.iface
        return None

    def node_id_to_dest(self, id):
        val = hex(self.node_id_to_hw_id(id)).strip('0x')
        return '!'+'0'*(8-len(val))+val

    def node_id_to_hw_id(self, id):
        return int(id) + HW_ID_OFFSET

    def send_from_to(self, fromNode, toNode):
        return self.get_node_iface_by_id(fromNode).getNode(self.node_id_to_dest(toNode))

    def on_receive(self, interface, packet):
        if "requestId" in packet["decoded"]:
            # Packet with requestId is coupled to original message
            existingMsgId = next((m.localId for m in self.messages if m.packet["id"] == packet["decoded"]["requestId"]), None)
            if existingMsgId is None:
                print('Could not find requestId!\n')
            mId = existingMsgId
        else:
            existingMsgId = next((m.localId for m in self.messages if m.packet["id"] == packet["id"]), None)
            if existingMsgId is not None:
                mId = existingMsgId
            else:
                self.messageId += 1
                mId = self.messageId
        rP = InteractivePacket(packet, mId)
        self.messages.append(rP)

        if self.script:
            print(f"Node {interface.myInfo.my_node_num-HW_ID_OFFSET} sent {packet['decoded']['simulator']['portnum']} with id {mId} over the air!")

        transmitter = next((n for n in self.nodes if n.TCPPort == interface.portNumber), None)
        if transmitter is not None:
            receivers = [n for n in self.nodes if n.nodeid != transmitter.nodeid]
            rxs, rssis, snrs = self.calc_receivers(transmitter, receivers)
            rP.setTxRxs(transmitter, rxs)
            rP.setRSSISNR(rssis, snrs)
            self.forward_packet(rxs, packet, rssis, snrs)
            self.graph.packets.append(rP)

    def on_receive_metrics(self, interface, packet):
        fromNode = next((n for n in self.nodes if n.hwId == packet["from"]), None)
        if fromNode is not None:
            data = packet["decoded"]["payload"]
            if getattr(data, "SerializeToString", None):
                data = data.SerializeToString()
            telemetryPacket = telemetry_pb2.Telemetry()
            telemetryPacket.ParseFromString(data)
            channelUtilization = 0
            airUtilTx = 0
            telemetryDict = proto.MessageToDict(telemetryPacket)
            if 'deviceMetrics' in telemetryDict:
                deviceMetrics = telemetryDict['deviceMetrics']
                if 'time' in telemetryDict:
                    timestamp = int(telemetryDict['time'])
                    # Check whether it is not a duplicate
                    if len(fromNode.timestamps) == 0 or timestamp > fromNode.timestamps[-1]:
                        fromNode.timestamps.append(timestamp)
                        if 'channelUtilization' in deviceMetrics:
                            channelUtilization = float(deviceMetrics['channelUtilization'])
                        fromNode.channelUtilization.append(channelUtilization)
                        if 'airUtilTx' in deviceMetrics:
                            airUtilTx = float(deviceMetrics['airUtilTx'])
                        fromNode.airUtilTx.append(airUtilTx)
            elif 'localStats' in telemetryDict:
                localStats = telemetryDict['localStats']
                if 'numPacketsTx' in localStats:
                    fromNode.numPacketsTx = localStats['numPacketsTx']
                if 'numPacketsRx' in localStats:
                    fromNode.numPacketsRx = localStats['numPacketsRx']
                if 'numPacketsRxBad' in localStats:
                    fromNode.numPacketsRxBad = localStats['numPacketsRxBad']
                if 'numRxDupe' in localStats:
                    fromNode.numRxDupe = localStats['numRxDupe']
                if 'numTxRelay' in localStats:
                    fromNode.numTxRelay = localStats['numTxRelay']
                if 'numTxRelayCanceled' in localStats:
                    fromNode.numTxRelayCanceled = localStats['numTxRelayCanceled']

    def on_receive_all(self, interface, packet):
        if interface.portNumber == 4403:
            fromRadio = self.copy_packet(packet)
            if fromRadio is not None:
                # print("Forward", packet["decoded"])
                b = fromRadio.SerializeToString()
                bufLen = len(b)
                # We convert into a string, because the TCP code doesn't work with byte arrays
                header = bytes([0x94, 0xC3, (bufLen >> 8) & 0xFF, bufLen & 0xFF])
                self.clientSocket.send(header + b)

    def node_reader(self):
        while not self.wantExit and self.nodes[0].iface is not None:
            if self.clientConnected:
                break
            else:
                bytes = self.nodes[0].iface._readBytes(MAX_TO_FROM_RADIO_SIZE)
                if len(bytes) > 0:
                    # print(bytes)
                    self.clientSocket.send(bytes)

    def client_reader(self):
        while not self.wantExit:
            if self.nodes[0].iface is not None:
                bytes = self.clientSocket.recv(MAX_TO_FROM_RADIO_SIZE)
                if len(bytes) > 0:
                    self.nodes[0].iface._writeBytes(bytes)
            else:
                time.sleep(0.1)

    def calc_receivers(self, tx, receivers):
        rxs = []
        rssis = []
        snrs = []
        for rx in receivers:
            dist_3d = calc_dist(tx.x, rx.x, tx.y, rx.y, tx.z, rx.z)
            pathLoss = phy.estimate_path_loss(conf, dist_3d, conf.FREQ, tx.z, rx.z)
            RSSI = conf.PTX + tx.antennaGain + rx.antennaGain - pathLoss
            SNR = RSSI-conf.NOISE_LEVEL
            if RSSI >= conf.SENSMODEM[conf.MODEM]:
                rxs.append(rx)
                rssis.append(RSSI)
                snrs.append(SNR)
        return rxs, rssis, snrs

    def close_nodes(self):
        print("\nClosing all nodes...")
        pub.unsubAll()
        for n in self.nodes:
            n.iface.localNode.exitSimulator()
            n.iface.close()
        if self.docker:
            self.container.stop()
        if self.forwardToClient:
            self._wantExit = True
            self.forwardSocket.close()
            self.clientSocket.close()


class CommandProcessor(cmd.Cmd):

    def cmdloop(self, sim):
        self.sim = sim
        print("Type 'help' to list the available commands for sending messages. Type 'plot' to show the routes or 'exit' to exit the simulator.")
        return cmd.Cmd.cmdloop(self)

    def do_broadcast(self, line):
        """broadcast <fromNode> <txt>
        Send a broadcast from node \x1B[3mfromNode\x1B[0m with text \x1B[3mtxt\x1B[0m."""
        arguments = line.split()
        if len(arguments) < 2:
            print('Please use the syntax: "broadcast <fromNode> <txt>"')
            return False
        fromNode = int(arguments[0])
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print(f'Node ID {fromNode} is not in the list of nodes.')
            return False
        txt = " ".join(arguments[2:])
        print(f'Instructing node {fromNode} to broadcast "{txt}" (message ID = {self.sim.messageId+1})')
        self.sim.send_broadcast(txt, fromNode)

    def do_dm(self, line):
        """DM <fromNode> <toNode> <txt>
        Send a Direct Message from node \x1B[3mfromNode\x1B[0m to node \x1B[3mtoNode\x1B[0m with text \x1B[3mtxt\x1B[0m."""
        arguments = line.split()
        if len(arguments) < 3:
            print('Please use the syntax: "DM <fromNode> <toNode> <txt>"')
            return False
        fromNode = int(arguments[0])
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print(f'Node ID {fromNode} is not in the list of nodes.')
            return False
        toNode = int(arguments[1])
        if self.sim.get_node_iface_by_id(toNode) is None:
            print(f'Node ID {toNode} is not in the list of nodes.')
            return False
        txt = " ".join(arguments[2:])
        print(f'Instructing node {fromNode} to DM node {toNode} "{txt}" (message ID = {self.sim.messageId+1})')
        self.sim.send_dm(txt, fromNode, toNode)

    def do_ping(self, line):
        """ping <fromNode> <toNode>
        Send ping from node \x1B[3mfromNode\x1B[0m to node \x1B[3mtoNode\x1B[0m."""
        arguments = line.split()
        if len(arguments) != 2:
            print('Please use the syntax: "ping <fromNode> <toNode>"')
            return False
        fromNode, toNode = map(int, arguments)
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print('Node ID', fromNode, 'is not in the list of nodes.')
            return False
        if self.sim.get_node_iface_by_id(toNode) is None:
            print('Node ID', toNode, 'is not in the list of nodes.')
            return False
        print(f'Instructing node {fromNode} to send ping to node {toNode} (message ID = {self.sim.messageId+1})')
        self.sim.send_ping(fromNode, toNode)

    def do_traceroute(self, line):
        """traceroute <fromNode> <toNode>
        Send a traceroute request from node \x1B[3mfromNode\x1B[0m to node \x1B[3mtoNode\x1B[0m."""
        arguments = line.split()
        if len(arguments) != 2:
            print('Please use the syntax: "traceroute <fromNode> <toNode>"')
            return False
        fromNode = int(arguments[0])
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print('Node ID', fromNode, 'is not in the list of nodes.')
            return False
        toNode = int(arguments[1])
        if self.sim.get_node_iface_by_id(toNode) is None:
            print('Node ID', toNode, 'is not in the list of nodes.')
            return False
        print(f'Instructing node {fromNode} to send traceroute request to node {toNode} (message ID = {self.sim.messageId+1})')
        print(f'This takes a while, the result will be in the log of node {fromNode}.')
        self.sim.trace_route(fromNode, toNode)

    def do_req_pos(self, line):
        """reqPos <fromNode> <toNode>
        Send a position request from node \x1B[3mfromNode\x1B[0m to node \x1B[3mtoNode\x1B[0m."""
        arguments = line.split()
        if len(arguments) != 2:
            print('Please use the syntax: "reqPos <fromNode> <toNode>"')
            return False
        fromNode = int(arguments[0])
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print(f'Node ID {fromNode} is not in the list of nodes.')
            return False
        toNode = int(arguments[1])
        if self.sim.get_node_iface_by_id(toNode) is None:
            print(f'Node ID {toNode} is not in the list of nodes.')
            return False
        print(f'Instructing node {fromNode} to send position request to node {toNode} (message ID = {self.sim.messageId+1})')
        self.sim.request_position(fromNode, toNode)

    def do_nodes(self, line):
        """nodes <id0> [id1, etc.]
        Show the node list as seen by node(s) \x1B[3mid0\x1B[0m, \x1B[3mid1\x1B[0m., etc."""
        arguments = line.split()
        if len(arguments) < 1:
            print('Please use the syntax: "nodes <id0> [id1, etc.]"')
            return False
        for n in arguments:
            if self.sim.get_node_iface_by_id(n) is None:
                print(f'Node ID {n} is not in the list of nodes.')
                continue
            self.sim.show_nodes(int(n))

    def do_remove(self, line):
        """remove <id>
        Remove node \x1B[3mid\x1B[0m from the current simulation."""
        arguments = line.split()
        if len(arguments) < 1:
            print('Please use the syntax: "remove <id>"')
            return False
        nodeId = (int(arguments[0]))
        if self.sim.get_node_iface_by_id(nodeId) is None:
            print(f'Node ID {nodeId} is not in the list of nodes.')
        else:
            self.sim.get_node_iface_by_id(nodeId).localNode.exitSimulator()
            self.sim.get_node_iface_by_id(nodeId).close()
            del self.sim.nodes[nodeId]

    def do_plot(self, line):
        """plot
        Plot the routes of messages sent and airtime statistics."""
        if self.sim.emulateCollisions:
            for n in self.sim.nodes:
                self.sim.request_local_stats(n.nodeid)
            time.sleep(1)
        self.sim.graph.plot_metrics(self.sim.nodes)
        self.sim.graph.init_routes(self.sim)
        return True

    def do_exit(self, line):
        """exit
        Exit the simulator without plotting routes."""
        self.sim.close_nodes()
        return True
