from lib.common import calc_dist
from lib.phy import airtime, estimate_path_loss

NODENUM_BROADCAST = 0xFFFFFFFF


class MeshPacket:
	def __init__(self, conf, nodes, origTxNodeId, destId, txNodeId, plen, seq, genTime, wantAck, isAck, requestId, now, verboseprint):
		self.conf = conf
		self.verboseprint = verboseprint
		self.origTxNodeId = origTxNodeId
		self.destId = destId
		self.txNodeId = txNodeId
		self.wantAck = wantAck
		self.isAck = isAck
		self.seq = seq
		self.requestId = requestId
		self.genTime = genTime
		self.now = now
		self.txpow = self.conf.PTX
		self.LplAtN = [0 for _ in range(self.conf.NR_NODES)]
		self.rssiAtN = [0 for _ in range(self.conf.NR_NODES)]
		self.sensedByN = [False for _ in range(self.conf.NR_NODES)]
		self.detectedByN = [False for _ in range(self.conf.NR_NODES)]
		self.collidedAtN = [False for _ in range(self.conf.NR_NODES)]
		self.receivedAtN = [False for _ in range(self.conf.NR_NODES)]
		self.onAirToN = [True for _ in range(self.conf.NR_NODES)]

		# configuration values
		self.sf = self.conf.SFMODEM[self.conf.MODEM]
		self.cr = self.conf.CRMODEM[self.conf.MODEM]
		self.bw = self.conf.BWMODEM[self.conf.MODEM]
		self.freq = self.conf.FREQ
		self.tx_node = next(n for n in nodes if n.nodeid == self.txNodeId)
		for rx_node in nodes:
			if rx_node.nodeid == self.txNodeId:
				continue
			dist_3d = calc_dist(self.tx_node.x, rx_node.x, self.tx_node.y, rx_node.y, self.tx_node.z, rx_node.z)
			offset = self.conf.LINK_OFFSET[(self.txNodeId, rx_node.nodeid)]
			self.LplAtN[rx_node.nodeid] = estimate_path_loss(self.conf, dist_3d, self.freq, self.tx_node.z, rx_node.z) + offset
			self.rssiAtN[rx_node.nodeid] = self.txpow + self.tx_node.antennaGain - self.LplAtN[rx_node.nodeid]
			if self.rssiAtN[rx_node.nodeid] >= self.conf.SENSMODEM[self.conf.MODEM]:
				self.sensedByN[rx_node.nodeid] = True
			if self.rssiAtN[rx_node.nodeid] >= self.conf.CADMODEM[self.conf.MODEM]:
				self.detectedByN[rx_node.nodeid] = True

		self.packetLen = plen
		self.timeOnAir = airtime(self.conf, self.sf, self.cr, self.packetLen, self.bw)
		self.startTime = 0
		self.endTime = 0

		# Routing
		self.retransmissions = self.conf.maxRetransmission
		self.ackReceived = False
		self.hopLimit = self.tx_node.hopLimit


class MeshMessage:
	def __init__(self, origTxNodeId, destId, genTime, seq):
		self.origTxNodeId = origTxNodeId
		self.destId = destId
		self.genTime = genTime
		self.seq = seq
		self.endTime = 0
