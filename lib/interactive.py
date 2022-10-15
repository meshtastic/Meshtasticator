from meshtastic import tcp_interface, BROADCAST_NUM, mesh_pb2, \
	admin_pb2, telemetry_pb2, remote_hardware_pb2, portnums_pb2
from pubsub import pub
from matplotlib import patches
from matplotlib.widgets import TextBox
import sys
import time
from . import config as conf
from .common import *
HW_ID_OFFSET = 16
TCP_PORT_OFFSET = 4403


class interactiveNode(): 
  def __init__(self, nodes, nodeId, hwId, TCPPort, x=-1, y=-1):
    self.nodeid = nodeId
    self.x = x
    self.y = y
    self.hwId = hwId
    self.TCPPort = TCPPort 
    if self.x == -1 and self.y == -1:
      self.x, self.y = findRandomPosition(nodes)

  def addInterface(self, iface):
    self.iface = iface


class interactivePacket():
	def __init__(self, packet, id):
		self.packet = packet
		self.localId = id
        
	def setTxRxs(self, transmitter, receivers):
		self.transmitter = transmitter
		self.receivers = receivers

	def setRSSISNR(self, rssis, snrs):
		self.rssis = rssis
		self.snrs = snrs


class interactiveGraph(Graph):
  def __init__(self):
    super().__init__()


  def initRoutes(self):
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
    self.fig.canvas.mpl_connect("button_press_event", self.onClick)
    print("Enter a message ID on the plot to show its route.")
    self.fig.canvas.draw_idle()
    plt.show()


  def clearRoute(self):
    for arr in self.arrows.copy():
      arr.remove()
      self.arrows.remove(arr)
    for ann in self.annots.copy():
      ann.remove()
      self.annots.remove(ann)


  def plotRoute(self, messageId):
    if self.firstTime: 
      print('Hover over an arc to show some info and click to remove it afterwards.')
    self.firstTime = False
    packets = [p for p in self.packets if p.localId == messageId]
    if len(packets) > 0:
      self.clearRoute()
      style = "Simple, tail_width=0.5, head_width=4, head_length=8"
      for p in packets:
        tx = p.transmitter
        rxs = p.receivers
        for i,rx in enumerate(rxs):
          kw = dict(arrowstyle=style, color=plt.cm.Set1(tx.nodeid))
          patch = patches.FancyArrowPatch((tx.x, tx.y), (rx.x, rx.y), connectionstyle="arc3,rad=.1", **kw)
          self.ax.add_patch(patch)

          if int(p.packet["to"]) == BROADCAST_NUM:
            to = "All"
          else: 
            to = str(p.packet["to"]-HW_ID_OFFSET)

          if "hopLimit" in p.packet:
            hopLimit = p.packet["hopLimit"]
          else:
            hopLimit = None

          if p.packet["from"] == tx.hwId and hopLimit == self.defaultHopLimit and not "requestId" in p.packet["decoded"]:
            msgType = "Original\/message"
          elif p.packet["priority"] != "ACK":
            if int(p.packet['from'])-HW_ID_OFFSET == rx.nodeid:
              msgType = "Implicit\/ACK"
            else:
              if to == "All":
                msgType = "Rebroadcast"
              else:
                msgType = "Forwarding\/packet"
          elif p.packet["priority"] == "ACK":
            msgType = "ACK"
          else: 
            msgType = "Other"

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
          rssiField = "RSSI: "+str(round(p.rssis[i], 2)) +" dBm"
          fields.append(rssiField)
          table = ""
          for i,f in enumerate(fields): 
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
      for i,a in enumerate(self.arrows):
        annot = self.annots[i]
        cont, _ = a.contains(event)
        if cont:
          annot.set_visible(True)
          self.fig.canvas.draw()
          break


  def onClick(self, event):
    for annot in self.annots:
      if annot.get_visible():
        annot.set_visible(False)
        self.fig.canvas.draw_idle()


  def submit(self, val):
    messageId = int(val)
    self.plotRoute(messageId)


class interactiveSim(): 
  def __init__(self):
    self.messages = []
    self.messageId = -1
    self.nodes = []
    foundNodes = False
    foundPath = False
    for i in range(1, len(sys.argv)):
      if not "--p" in sys.argv[i] and not "/" in sys.argv[i]:
        if int(sys.argv[i]) > 10:
          print("Not sure if you want to start more than 10 terminals. Exiting.")
          exit(1)
        conf.NR_NODES = int(sys.argv[i])
        xs = []
        ys = []
        foundNodes = True
        break
    if not foundNodes: 
      [xs, ys] = genScenario()
      conf.NR_NODES = len(xs)
    if len(sys.argv) > 2 and "--p" in sys.argv[2]:
      string = sys.argv[3]
      pathToProgram = string
      foundPath = True
    elif len(sys.argv) > 1:
      if "--p" in sys.argv[1]:
        string = sys.argv[2]
        pathToProgram = string
        foundPath = True
    if not foundPath:
      pathToProgram = os.getcwd()+"/"


    self.graph = interactiveGraph()
    for n in range(conf.NR_NODES):
      if len(xs) == 0: 
        node = interactiveNode(self.nodes, n, n+HW_ID_OFFSET, n+TCP_PORT_OFFSET)
      else:
        node = interactiveNode(self.nodes, n, n+HW_ID_OFFSET, n+TCP_PORT_OFFSET, xs[n], ys[n])
      self.nodes.append(node)
      self.graph.addNode(node)

    for n in self.nodes:
      cmdString = "gnome-terminal --title='Node "+str(n.nodeid)+"' -- "+pathToProgram+"program -e -d "+os.path.expanduser('~')+"/.portduino/node"+str(n.nodeid)+" -h "+str(n.hwId)+" -p "+str(n.TCPPort)
      os.system(cmdString) 

    time.sleep(4)  # Allow instances to start up their TCP service 
    try:
      for n in self.nodes:
        iface = tcp_interface.TCPInterface(hostname="localhost", portNumber=n.TCPPort)
        n.addInterface(iface)
      pub.subscribe(self.onReceive, "meshtastic.receive.simulator")
    except(Exception) as ex:
      print(f"Error: Could not connect to native program: {ex}")
      for n in self.nodes:
        n.iface.close()
      os.system("killall "+pathToProgram+"program")
      sys.exit(1)


  def forwardPacket(self, iface, packet, rssi, snr): 
    data = packet["decoded"]["payload"]
    if getattr(data, "SerializeToString", None):
      data = data.SerializeToString()

    if len(data) > mesh_pb2.Constants.DATA_PAYLOAD_LEN:
      raise Exception("Data payload too big")

    if packet["decoded"]["portnum"] == "TEXT_MESSAGE_APP":
      meshPacket = mesh_pb2.MeshPacket()
    elif packet["decoded"]["portnum"] == "ROUTING_APP":
      meshPacket = mesh_pb2.Routing()
    elif packet["decoded"]["portnum"] == "NODEINFO_APP":
      meshPacket = mesh_pb2.NodeInfo()
    elif packet["decoded"]["portnum"] == "POSITION_APP":
      meshPacket = mesh_pb2.Position()
    elif packet["decoded"]["portnum"] == "USER_APP":
      meshPacket = mesh_pb2.User()  
    elif packet["decoded"]["portnum"] == "ADMIN_APP":
      meshPacket = admin_pb2.AdminMessage()
    elif packet["decoded"]["portnum"] == "TELEMETRY_APP":
      meshPacket = telemetry_pb2.Telemetry()
    elif packet["decoded"]["portnum"] == "REMOTE_HARDWARE_APP":
      meshPacket = remote_hardware_pb2.HardwareMessage()
    else:
      meshPacket = mesh_pb2.MeshPacket()

    meshPacket.decoded.payload = data
    meshPacket.decoded.portnum = 69
    meshPacket.to = packet["to"]
    setattr(meshPacket, "from", packet["from"])
    meshPacket.id = packet["id"]
    if "wantAck" in packet:
      meshPacket.want_ack = packet["wantAck"]
    if "hopLimit" in packet:
      meshPacket.hop_limit = packet["hopLimit"]
    if "requestId" in packet["decoded"]:
      meshPacket.decoded.request_id = packet["decoded"]["requestId"]
    if "wantResponse" in packet["decoded"]:
      meshPacket.decoded.want_response = packet["decoded"]["wantResponse"]
    meshPacket.rx_rssi = int(rssi) 
    meshPacket.rx_snr = int(snr)  
    toRadio = mesh_pb2.ToRadio()
    toRadio.packet.CopyFrom(meshPacket)
    iface._sendToRadio(toRadio)


  def sendBroadcast(self, text, fromNode):
    self.nodes[fromNode].iface.sendText(text)


  def sendDM(self, text, fromNode, toNode):
    self.nodes[fromNode].iface.sendText(text, destinationId=self.nodes[toNode].hwId, wantAck=True)


  def sendPing(self, fromNode, toNode):
    payload = str.encode("test string")
    self.nodes[fromNode].iface.sendData(payload, self.nodes[toNode].hwId, portNum=portnums_pb2.PortNum.REPLY_APP,
      wantAck=True, wantResponse=True)


  def onReceive(self, interface, packet): 
    if "requestId" in packet["decoded"]:
      # Packet with requestId is coupled to original message
      existingMsgId = next((m.localId for m in self.messages if m.packet["id"] == packet["decoded"]["requestId"]), None)
      if existingMsgId == None:
          print('Could not find requestId!\n')
      mId = existingMsgId
    else:
      existingMsgId = next((m.localId for m in self.messages if m.packet["id"] == packet["id"]), None)
      if existingMsgId != None:
          mId = existingMsgId
      else: 
          self.messageId += 1
          mId = self.messageId
    rP = interactivePacket(packet, mId)
    self.messages.append(rP)
    print("Node", interface.myInfo.my_node_num-HW_ID_OFFSET, "sent", packet["decoded"]["simulator"]["portnum"], "with id", mId, "over the air!")
    transmitter = next((n for n in self.nodes if n.TCPPort == interface.portNumber), None)
    receivers = [n for n in self.nodes if n.nodeid != transmitter.nodeid]
    rxs, rssis, snrs = self.calcReceivers(transmitter, receivers)
    rP.setTxRxs(transmitter, rxs)
    rP.setRSSISNR(rssis, snrs)
    for i,r in enumerate(rxs):
      self.forwardPacket(r.iface, packet, rssis[i], snrs[i])
    self.graph.packets.append(rP)


  def calcReceivers(self, tx, receivers): 
    rxs = []
    rssis = []
    snrs = []
    for rx in receivers:
      dist_2d = calcDist(tx.x, tx.y, rx.x, rx.y) 
      pathLoss = phy.estimatePathLoss(dist_2d, conf.FREQ)
      RSSI = conf.PTX + conf.GL - pathLoss
      SNR = RSSI-conf.NOISE_LEVEL
      if RSSI >= conf.SENSMODEM[conf.MODEM]:
        rxs.append(rx)
        rssis.append(RSSI)
        snrs.append(SNR)
    return rxs, rssis, snrs


  def closeNodes(self):
    print("\nClosing all nodes...")
    pub.unsubAll()
    for n in self.nodes:
      n.iface.localNode.exitSimulator()