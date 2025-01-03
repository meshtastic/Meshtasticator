#!/usr/bin/env python3
import matplotlib
try:
	matplotlib.use("TkAgg")
except ImportError:
	print('Tkinter is needed. Install python3-tk with your package manager.')
	exit(1)

from lib.common import *
from lib.packet import *
from lib.mac import *
from lib.discrete_event import *


VERBOSE = False
SAVE = True


class MeshNode():
	def __init__(self, nodes, env, bc_pipe, nodeid, period, messages, packetsAtN, packets, delays, nodeConfig):
		self.nodeid = nodeid
		if nodeConfig is not None: 
			self.x = nodeConfig['x']
			self.y = nodeConfig['y']
			self.z = nodeConfig['z']
			self.isRouter = nodeConfig['isRouter']
			self.isRepeater = nodeConfig['isRepeater']
			self.isClientMute = nodeConfig['isClientMute']
			self.hopLimit = nodeConfig['hopLimit']
			self.antennaGain = nodeConfig['antennaGain']
		else: 
			self.x, self.y = findRandomPosition(nodes)
			self.z = conf.HM
			self.isRouter = conf.router
			self.isRepeater = False
			self.isClientMute = False
			self.hopLimit = conf.hopLimit
			self.antennaGain = conf.GL
		self.messageSeq = messageSeq
		self.env = env
		self.period = period
		self.bc_pipe = bc_pipe
		self.rx_snr = 0
		self.nodes = nodes
		self.messages = messages
		self.packetsAtN = packetsAtN
		self.nrPacketsSent = 0
		self.packets = packets
		self.delays = delays
		self.leastReceivedHopLimit = {}
		self.isReceiving = []
		self.isTransmitting = False
		self.usefulPackets = 0
		self.txAirUtilization = 0
		self.airUtilization = 0

		if not self.isRepeater:  # repeaters don't generate messages themselves
			env.process(self.generateMessage())
		env.process(self.receive(self.bc_pipe.get_output_conn()))
		self.transmitter = simpy.Resource(env, 1)

	def generateMessage(self):
		global messageSeq
		while True:
			nextGen = random.expovariate(1.0/float(self.period))
			# do not generate message near the end of the simulation (otherwise flooding cannot finish in time)
			if self.env.now+nextGen+self.hopLimit*airtime(conf.SFMODEM[conf.MODEM], conf.CRMODEM[conf.MODEM], conf.PACKETLENGTH, conf.BWMODEM[conf.MODEM]) < conf.SIMTIME:
				yield self.env.timeout(nextGen) 

				if conf.DMs:
					destId = random.choice([i for i in range(0, len(nodes)) if i is not self.nodeid])
				else:
					destId = NODENUM_BROADCAST
				messageSeq += 1
				self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
				p = MeshPacket(self.nodes, self.nodeid, destId, self.nodeid, conf.PACKETLENGTH, messageSeq, self.env.now, True, False, None)  
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'generated message', p.seq, 'to', destId)
				self.packets.append(p)
				self.env.process(self.transmit(p))
				while p.wantAck: # ReliableRouter: retransmit message if no ACK received after timeout 
					retransmissionMsec = getRetransmissionMsec(self, p) 
					yield self.env.timeout(retransmissionMsec)

					ackReceived = False  # check whether you received an ACK on the transmitted message
					minRetransmissions = conf.maxRetransmission
					for packetSent in self.packets:
						if packetSent.origTxNodeId == self.nodeid and packetSent.seq == p.seq:
							if packetSent.retransmissions < minRetransmissions:
								minRetransmissions = packetSent.retransmissions
							if packetSent.ackReceived:
								ackReceived = True
					if ackReceived: 
						verboseprint('Node', self.nodeid, 'received ACK on generated message with seq. nr.', p.seq)
						break
					else: 
						if minRetransmissions > 0:  # generate new packet with same sequence number
							pNew = MeshPacket(self.nodes, self.nodeid, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None)  
							pNew.retransmissions = minRetransmissions-1
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'wants to retransmit its generated packet to', destId, 'with seq.nr.', p.seq, 'minRetransmissions', minRetransmissions)
							self.packets.append(pNew)
							self.env.process(self.transmit(pNew))
						else:
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'reliable send of', p.seq, 'failed.')
							break
			else:  # do not send this message anymore, since it is close to the end of the simulation
				break


	def transmit(self, packet):
		with self.transmitter.request() as request:
			yield request

			# listen-before-talk from src/mesh/RadioLibInterface.cpp 
			txTime = setTransmitDelay(self, packet) 
			verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'picked wait time', txTime)
			yield self.env.timeout(txTime)

			# wait when currently receiving or transmitting, or channel is active
			while any(self.isReceiving) or self.isTransmitting or isChannelActive(self, self.env):
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is busy Tx-ing', self.isTransmitting, 'or Rx-ing', any(self.isReceiving), 'else channel busy!')
				txTime = setTransmitDelay(self, packet) 
				yield self.env.timeout(txTime)
			verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'ends waiting')

			# check if you received an ACK for this message in the meantime
			if packet.seq not in self.leastReceivedHopLimit:
				self.leastReceivedHopLimit[packet.seq] = packet.hopLimit+1 
			if self.leastReceivedHopLimit[packet.seq] > packet.hopLimit:  # no ACK received yet, so may start transmitting 
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'started low level send', packet.seq, 'hopLimit', packet.hopLimit, 'original Tx', packet.origTxNodeId)
				self.nrPacketsSent += 1
				for rx_node in self.nodes:
					if packet.sensedByN[rx_node.nodeid] == True:
						if (checkcollision(self.env, packet, rx_node.nodeid, self.packetsAtN) == 0):
							self.packetsAtN[rx_node.nodeid].append(packet)
				packet.startTime = self.env.now
				packet.endTime = self.env.now + packet.timeOnAir
				self.txAirUtilization += packet.timeOnAir
				self.airUtilization += packet.timeOnAir
				self.bc_pipe.put(packet)
				self.isTransmitting = True
				yield self.env.timeout(packet.timeOnAir)
				self.isTransmitting = False
			else:  # received ACK: abort transmit, remove from packets generated 
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'in the meantime received ACK, abort packet with seq. nr', packet.seq)
				self.packets.remove(packet)


	def receive(self, in_pipe):
		global messageSeq
		while True:
			p = yield in_pipe.get()
			if p.sensedByN[self.nodeid] and not p.collidedAtN[self.nodeid] and p.onAirToN[self.nodeid]:  # start of reception
				if not self.isTransmitting:
					verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'started receiving packet', p.seq, 'from', p.txNodeId)
					p.onAirToN[self.nodeid] = False 
					self.isReceiving.append(True)
				else:  # if you were currently transmitting, you could not have sensed it
					verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'was transmitting, so could not receive packet', p.seq)
					p.sensedByN[self.nodeid] = False
					p.onAirToN[self.nodeid] = False
			elif p.sensedByN[self.nodeid]:  # end of reception
				try: 
					self.isReceiving[self.isReceiving.index(True)] = False 
				except: 
					pass
				self.airUtilization += p.timeOnAir
				if p.collidedAtN[self.nodeid]:
					verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'could not decode packet.')
					continue
				p.receivedAtN[self.nodeid] = True
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received packet', p.seq, 'with delay', round(env.now-p.genTime, 2))
				delays.append(env.now-p.genTime)

				# update hopLimit for this message
				if p.seq not in self.leastReceivedHopLimit:  # did not yet receive packet with this seq nr.
					# verboseprint('Node', self.nodeid, 'received packet nr.', p.seq, 'orig. Tx', p.origTxNodeId, "for the first time.")
					self.usefulPackets += 1
					self.leastReceivedHopLimit[p.seq] = p.hopLimit
				if p.hopLimit < self.leastReceivedHopLimit[p.seq]:  # hop limit of received packet is lower than previously received one
					self.leastReceivedHopLimit[p.seq] = p.hopLimit

				# check if implicit ACK for own generated message
				if p.origTxNodeId == self.nodeid:
					if p.isAck:
						verboseprint('Node', self.nodeid, 'received real ACK on generated message.')
					else:
						verboseprint('Node', self.nodeid, 'received implicit ACK on message sent.')
					p.ackReceived = True
					continue

				ackReceived = False
				realAckReceived = False
				for sentPacket in self.packets:
					# check if ACK for message you currently have in queue
					if sentPacket.txNodeId == self.nodeid and sentPacket.seq == p.seq:
						verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received implicit ACK for message in queue.')
						ackReceived = True
						sentPacket.ackReceived = True
					# check if real ACK for message sent
					if sentPacket.origTxNodeId == self.nodeid and p.isAck and sentPacket.seq == p.requestId:
						verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received real ACK.')
						realAckReceived = True
						sentPacket.ackReceived = True

				# send real ACK if you are the destination and you did not yet send the ACK
				if p.wantAck and p.destId == self.nodeid and not any(pA.requestId == p.seq for pA in self.packets):
					verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sends a flooding ACK.')
					messageSeq += 1
					self.messages.append(MeshMessage(self.nodeid, p.origTxNodeId, self.env.now, messageSeq))
					pAck = MeshPacket(self.nodes, self.nodeid, p.origTxNodeId, self.nodeid, conf.ACKLENGTH, messageSeq, env.now, False, True, p.seq) 
					self.packets.append(pAck)
					self.env.process(self.transmit(pAck))
        		# Rebroadcasting Logic for received message. This is a broadcast or a DM not meant for us.
				elif not p.destId == self.nodeid and not ackReceived and not realAckReceived and p.hopLimit > 0:
					# FloodingRouter: rebroadcast received packet
					if conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.MANAGED_FLOOD:
						if not self.isClientMute:
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', p.seq)
							pNew = MeshPacket(self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None) 
							pNew.hopLimit = p.hopLimit-1
							self.packets.append(pNew)
							self.env.process(self.transmit(pNew))
					# BloomRouter: rebroadcast received packet
					elif conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.BLOOM:
						verboseprint('Packet', p.seq, 'received at node', self.nodeid, 'with coverage', p.coverageFilter)
						pNew = MeshPacket(self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, p.coverageFilter)
						pNew.hopLimit = p.hopLimit-1

						rebroadcastProbabilityTest = random.random()
						newCoverage = pNew.checkAdditionalCoverageRatio(p.coverageFilter)
						rebroadcastProbability = conf.BASELINE_REBROADCAST_PROBABILITY + (newCoverage * conf.COVERAGE_RATIO_SCALE_FACTOR)

						'''
						# Check if this node covers any additional nodes by providing the old packets coverage
						newCoverage = pNew.checkAdditionalCoverage(p.coverageFilter)

						# Baseline rebroadcast probability is 0.2 to deal with FPR (False Positive Rate)
						rebroadcastProbability = 0.2
						
						# Piecewise logic for rebroadcast probability
						# If we can add a single node, 80% chance of rebroadcast
						if (newCoverage == 1):
							rebroadcastProbability = 0.8
						# If we can add more than a single node, 100% chance of rebroadcast
						elif (newCoverage > 1):
							rebroadcastProbability = 1
						'''
						# Clamp to values that make sense
						rebroadcastProbability = max(0.0, min(1.0, rebroadcastProbability))
						# Check the random against the probability
						if rebroadcastProbabilityTest <= rebroadcastProbability:
							self.packets.append(pNew)
							self.env.process(self.transmit(pNew))
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', p.seq, 'New Coverage:', newCoverage, 'Rnd:', rebroadcastProbabilityTest, 'Prob:', rebroadcastProbability)
						else:
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'drops received packet due to coverage', p.seq, 'New Coverage:', newCoverage, 'Rnd:', rebroadcastProbabilityTest, 'Prob:', rebroadcastProbability)


if VERBOSE:
	def verboseprint(*args, **kwargs): 
		print(*args, **kwargs)
else:   
	def verboseprint(*args, **kwargs): 
		pass


repetitions = 100
parameters = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25]
collisions = []
reachability = []
usefulness = []
meanDelays = []
meanTxAirUtils = []
collisionStds = []
reachabilityStds = []
usefulnessStds = []
delayStds = []
txAirUtilsStds = []
for p, nrNodes in enumerate(parameters):
	conf.NR_NODES = nrNodes
	nodeReach = [0 for _ in range(repetitions)]
	nodeUsefulness = [0 for _ in range(repetitions)]
	collisionRate = [0 for _ in range(repetitions)]
	meanDelay = [0 for _ in range(repetitions)]
	meanTxAirUtilization = [0 for _ in range(repetitions)]
	print("\nStart of", p+1, "out of", len(parameters), "value", nrNodes)
	for rep in range(repetitions):
		setBatch(rep)
		random.seed(rep)
		env = simpy.Environment()
		bc_pipe = BroadcastPipe(env)

		nodes = []
		messages = []
		packets = []
		delays = []
		packetsAtN = [[] for _ in range(conf.NR_NODES)]
		messageSeq = 0

		found = False
		while not found:
			nodes = []
			for nodeId in range(conf.NR_NODES):
				node = MeshNode(nodes, env, bc_pipe, nodeId, conf.PERIOD, messages, packetsAtN, packets, delays, None)
				if node.x == None:
					break
				nodes.append(node)
			if len(nodes) == conf.NR_NODES:
				found = True

		# start simulation
		env.run(until=conf.SIMTIME)
		nrCollisions = sum([1 for p in packets for n in nodes if p.collidedAtN[n.nodeid] == True])
		nrSensed = sum([1 for p in packets for n in nodes if p.sensedByN[n.nodeid] == True])
		nrReceived = sum([1 for p in packets for n in nodes if p.receivedAtN[n.nodeid] == True])
		nrUseful = sum([n.usefulPackets for n in nodes])
		if nrSensed != 0:
			collisionRate[rep] = float((nrCollisions)/nrSensed)*100
		else:
			collisionRate[rep] = np.NaN
		if messageSeq != 0: 
			nodeReach[rep] = nrUseful/(messageSeq*(conf.NR_NODES-1))*100
		else: 
			nodeReach[rep] = np.NaN
		if nrReceived != 0:
			nodeUsefulness[rep] = nrUseful/nrReceived*100  # nr of packets that delivered to a message to a new receiver out of all packets received
		else: 
			nodeUsefulness[rep] = np.NaN
		meanDelay[rep] = np.nanmean(delays)
		meanTxAirUtilization[rep] = sum([n.txAirUtilization for n in nodes])/conf.NR_NODES
	if SAVE:
		print('Saving to file...')
		data = {
			"CollisionRate": collisionRate,
			"Reachability": nodeReach, 
			"Usefulness": nodeUsefulness,
			"meanDelay": meanDelay,
			"meanTxAirUtil": meanTxAirUtilization,
			"nrCollisions": nrCollisions,
			"nrSensed": nrSensed,
			"nrReceived": nrReceived,
			"usefulPackets": nrUseful, 
			"MODEM": conf.NR_NODES, 
			"MODEL": conf.MODEL,
			"NR_NODES": conf.NR_NODES,
			"INTERFERENCE_LEVEL": conf.INTERFERENCE_LEVEL,
			"COLLISION_DUE_TO_INTERFERENCE": conf.COLLISION_DUE_TO_INTERFERENCE,
			"XSIZE": conf.XSIZE, 
			"YSIZE": conf.YSIZE,
			"MINDIST": conf.MINDIST,
			"SIMTIME": conf.SIMTIME,
			"PERIOD": conf.PERIOD,
			"PACKETLENGTH": conf.PACKETLENGTH,  
			"nrMessages": messageSeq
		}
		subdir = "hopLimit3"
		simReport(data, subdir, nrNodes)
	print('Collision rate average:', round(np.nanmean(collisionRate), 2))
	print('Reachability average:', round(np.nanmean(nodeReach), 2))
	print('Usefulness average:', round(np.nanmean(nodeUsefulness), 2))
	print('Delay average:', round(np.nanmean(meanDelay), 2))
	print('Tx air utilization average:', round(np.nanmean(meanTxAirUtilization), 2))
	collisions.append(np.nanmean(collisionRate))
	reachability.append(np.nanmean(nodeReach))
	usefulness.append(np.nanmean(nodeUsefulness))
	meanDelays.append(np.nanmean(meanDelay))
	meanTxAirUtils.append(np.nanmean(meanTxAirUtilization))
	collisionStds.append(np.nanstd(collisionRate))
	reachabilityStds.append(np.nanstd(nodeReach))
	usefulnessStds.append(np.nanstd(nodeUsefulness))
	delayStds.append(np.nanstd(meanDelay))
	txAirUtilsStds.append(np.nanstd(meanTxAirUtilization))


plt.errorbar(parameters, collisions, collisionStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Collision rate (%)')
plt.figure()
plt.errorbar(parameters, meanDelays, delayStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Average delay (ms)')
plt.figure()
plt.errorbar(parameters, meanTxAirUtils, txAirUtilsStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Average Tx air utilization (ms)')
plt.figure()
plt.errorbar(parameters, reachability, reachabilityStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Reachability (%)')
plt.figure()
plt.errorbar(parameters, usefulness, usefulnessStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Usefulness (%)')
plt.show()
