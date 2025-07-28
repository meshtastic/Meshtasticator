import random
from lib.phy import airtime

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
		self.sf = self.conf.current_preset["sf"]
		self.cr = self.conf.current_preset["cr"]
		self.bw = self.conf.current_preset["bw"]
		self.freq = self.conf.FREQ
		self.tx_node = next(n for n in nodes if n.nodeid == self.txNodeId)
		
		# OPTIMIZATION: Use precomputed connectivity matrix if available
		if hasattr(self.conf, 'CONNECTIVITY_MATRIX') and self.txNodeId in self.conf.CONNECTIVITY_MATRIX:
			# Only process nodes that are actually connectable (huge performance gain)
			connectable_nodes = self.conf.CONNECTIVITY_MATRIX[self.txNodeId]
			for rx_nodeid in connectable_nodes:
				rx_node = nodes[rx_nodeid]
				# Use precomputed baseline path loss + dynamic ±5dB variation per packet
				baseline_path_loss = self.conf.BASELINE_PATH_LOSS_MATRIX[self.txNodeId][rx_nodeid]
				dynamic_offset = random.gauss(0, 5)  # ±5dB dynamic variation for fading/interference
				self.LplAtN[rx_node.nodeid] = baseline_path_loss + dynamic_offset
				self.rssiAtN[rx_node.nodeid] = self.txpow + self.tx_node.antennaGain + rx_node.antennaGain - self.LplAtN[rx_node.nodeid]
				if self.rssiAtN[rx_node.nodeid] >= self.conf.current_preset["sensitivity"]:
					self.sensedByN[rx_node.nodeid] = True
				if self.rssiAtN[rx_node.nodeid] >= self.conf.current_preset["cad_threshold"]:
					self.detectedByN[rx_node.nodeid] = True
		else:
			# Fallback: if no precomputed data, only process a reasonable subset for performance
			# (This should not happen in normal operation but prevents O(n²) fallback)
			processed_count = 0
			max_fallback_nodes = min(50, len(nodes))  # Limit fallback processing
			for rx_node in nodes:
				if rx_node.nodeid == self.txNodeId or processed_count >= max_fallback_nodes:
					continue
				processed_count += 1
				# Apply a conservative path loss estimate for fallback
				conservative_path_loss = 140 + random.gauss(0, 5)  # Conservative estimate + variation
				self.LplAtN[rx_node.nodeid] = conservative_path_loss
				self.rssiAtN[rx_node.nodeid] = self.txpow + self.tx_node.antennaGain + rx_node.antennaGain - self.LplAtN[rx_node.nodeid]
				if self.rssiAtN[rx_node.nodeid] >= self.conf.current_preset["sensitivity"]:
					self.sensedByN[rx_node.nodeid] = True
				if self.rssiAtN[rx_node.nodeid] >= self.conf.current_preset["cad_threshold"]:
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
