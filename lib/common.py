import random
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.widgets import Button, Slider, RadioButtons, TextBox

from lib import phy

try:
	matplotlib.use("TkAgg")
except ImportError:
	print('Tkinter is needed. Install python3-tk with your package manager.')
	exit(1)


def gen_scenario(conf):
	save = True  # set to True if you want to save the coordinates of the nodes
	nodeX = []
	nodeY = []
	nodeZ = []
	circles = []
	nodeRouter = []
	nodeRepeater = []
	nodeClientMute = []
	nodeHopLimit = []
	nodeTxts = []
	gains = []
	neighborInfo = []

	fig = plt.figure()
	ax = fig.add_subplot(111)
	fig.subplots_adjust(bottom=0.20, right=0.85)  # Make room for button and config
	title = "Double click to place a node. Then change its config (optional)."
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_xlim(-(conf.XSIZE/2+1)+conf.OX, conf.OX+conf.XSIZE/2+1)
	ax.set_ylim(-(conf.YSIZE/2+1)+conf.OY, conf.OY+conf.YSIZE/2+1)
	ax.set_title(title)
	# 'Start simulation' button
	button_ax = fig.add_axes([0.37, 0.05, 0.2, 0.06])
	button = Button(button_ax, 'Start simulation', color='red', hovercolor='green')
	# Role selection
	role_ax = fig.add_axes([0.84, 0.61, 0.12, 0.2])
	role_ax.set_axis_off()
	roleButton = RadioButtons(role_ax, ['Client', 'Client mute', 'Router', 'Repeater'], active=1 if conf.router else 0)
	role_ax.set_visible(False)
	# HopLimit slider
	slider_ax = fig.add_axes([0.86, 0.34, 0.1, 0.22])
	slider = Slider(slider_ax, 'HopLimit', 0, 7, valinit=conf.hopLimit, valstep=1, orientation="vertical")
	slider_ax.set_visible(False)
	# Height textbox
	height_ax = fig.add_axes([0.89, 0.22, 0.05, 0.04])
	height_textbox = TextBox(height_ax, 'Height (m)', conf.HM, textalignment='center')
	height_ax.set_visible(False)
	textBoxLabel = height_textbox.ax.get_children()[0]
	textBoxLabel.set_position([0.5, 1.75])
	textBoxLabel.set_verticalalignment('top')
	textBoxLabel.set_horizontalalignment('center')
	# Antenna gain textbox
	gain_ax = fig.add_axes([0.89, 0.11, 0.05, 0.04])
	gain_textbox = TextBox(gain_ax, 'Antenna \ngain (dBi)', conf.GL, textalignment='center')
	gain_ax.set_visible(False)
	gainLabel = gain_textbox.ax.get_children()[0]
	gainLabel.set_position([0.5, 2.5])
	gainLabel.set_verticalalignment('top')
	gainLabel.set_horizontalalignment('center')

	def plotting():
    # Add latest node 
		nx = nodeX[-1]
		ny = nodeY[-1]
		ax.annotate(str(len(nodeX)-1), (nx-5, ny+5))
		circle = plt.Circle((nx, ny), radius=phy.MAXRANGE, color=plt.cm.Set1(len(nodeX)-1), alpha=0.1)
		circles.append(circle)
		ax.add_patch(circle)
		ax.scatter(nx, ny) # small dot in the middle

		if len(nodeTxts) > 0:
			# Remove last 'Configure node x' text
			nodeTxts[-1].set_visible(False)
		else:
			# After first node is placed, display config options
			role_ax.set_visible(True)
			slider_ax.set_visible(True)
			height_ax.set_visible(True)
			gain_ax.set_visible(True)
		nodeTxts.append(
			plt.text(
				0.92, 0.80, 'Configure \nnode '+str(len(nodeX)-1)+':', fontweight='bold', horizontalalignment='center', transform=fig.transFigure
			)
		)

		fig.canvas.draw_idle()
		fig.canvas.get_tk_widget().focus_set()

	def submit(mouse_event):
		if (len(nodeX)) < 2:
			print("Need at least two nodes.")
			exit(1)
		# Save last config
		nodeZ.append(float(height_textbox.text))
		nodeRouter.append(roleButton.value_selected == 'Router')
		nodeRepeater.append(roleButton.value_selected == 'Repeater')
		nodeClientMute.append(roleButton.value_selected == 'Client mute')
		nodeHopLimit.append(slider.val)
		gains.append(float(gain_textbox.text))
		neighborInfo.append(bool(0))
		fig.canvas.mpl_disconnect(cid)
		plt.close()
	button.on_clicked(submit)
	
	def submit_gain(text):
		circles[-1].set_radius(phy.estimate_max_range(float(text)))
		fig.canvas.draw_idle()
	gain_textbox.on_submit(submit_gain)

	def onclick(event):
		if event.dblclick:
			if len(nodeX) > 0:
				# Save config of previous node
				nodeZ.append(float(height_textbox.text))
				nodeRouter.append(roleButton.value_selected == 'Router')
				nodeRepeater.append(roleButton.value_selected == 'Repeater')
				nodeClientMute.append(roleButton.value_selected == 'Client mute')
				nodeHopLimit.append(slider.val)
				gains.append(float(gain_textbox.text))
				neighborInfo.append(bool(0))

			# New node placement
			nodeX.append(float(event.xdata))
			nodeY.append(float(event.ydata))
			plotting()
			
			# Reset config values only after new node is placed
			roleButton.set_active(1 if conf.router else 0)
			height_textbox.set_val(conf.HM)
			slider.set_val(conf.hopLimit)
			gain_textbox.set_val(conf.GL)

	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()
	# Save node configuration in a dictionary
	nodeDict = {n: {
		'x': nodeX[n], 'y': nodeY[n], 'z': nodeZ[n],
		'isRouter': nodeRouter[n],
		'isRepeater': nodeRepeater[n],
		'isClientMute': nodeClientMute[n],
		'hopLimit': nodeHopLimit[n],
		'antennaGain': gains[n],
		'neighborInfo': neighborInfo[n],
	} for n in range(len(nodeX))}
	if save:
		if not os.path.isdir("out"):
			os.mkdir("out")
		with open(os.path.join("out", "nodeConfig.yaml"), 'w') as file:
			yaml.dump(nodeDict, file)

	return nodeDict


def find_random_position(conf, nodes):
	foundMin = True
	foundMax = False
	tries = 0
	x = 0
	y = 0
	while not (foundMin and foundMax):
		a = random.random()
		b = random.random()
		posx = a*conf.XSIZE+conf.OX-conf.XSIZE/2
		posy = b*conf.YSIZE+conf.OY-conf.YSIZE/2
		if len(nodes) > 0:
			for n in nodes:
				dist = calc_dist(n.x, posx, n.y, posy)
				if dist < conf.MINDIST:
					foundMin = False
					break
				pathLoss = phy.estimate_path_loss(conf, dist, conf.FREQ)
				rssi = conf.PTX + 2*conf.GL - pathLoss
				# At least one node should be able to reach it
				if rssi >= conf.SENSMODEM[conf.MODEM]:
					foundMax = True
			if foundMin and foundMax:
				x = posx
				y = posy
		else:
			x = posx
			y = posy
			foundMin = True
			foundMax = True
		tries += 1
		if tries > 1000:
			print('Could not find a location to place the node. Try increasing XSIZE/YSIZE or decreasing MINDIST.')
			break
	return max(-conf.XSIZE/2, x), max(-conf.YSIZE/2, y)


def run_graph_updates(env, graph, nodes, interval):
	while True:
		# Wait 'interval' sim-mseconds
		yield env.timeout(interval)
		# Now update the positions in the graph
		graph.update_positions(nodes)


def calc_dist(x0, x1, y0, y1, z0=0, z1=0):
	return np.sqrt(((abs(x0-x1))**2)+((abs(y0-y1))**2)+((abs(z0-z1)**2)))


scheduleIdx = 0


def plot_schedule(conf, packets, messages):
	def draw_schedule(i):
		t = timeSequences[i]
		plt.suptitle('Time schedule {}/{}\nDouble click to continue.'.format(i+1, len(timeSequences)))
		for p in packets:  # collisions
			if p.seq in [m.seq for m in t]:
				for rxId, bool in enumerate(p.collidedAtN):
					if bool:
						plt.barh(rxId, p.timeOnAir, left=p.startTime, color='red', edgecolor='r')
		for p in packets:  # transmissions
			if p.seq in [m.seq for m in t]:
				color = 'orange' if p.isAck else 'blue'
				plt.barh(p.txNodeId, p.timeOnAir, left=p.startTime, color=color, edgecolor='k')
				plt.text(p.startTime+p.timeOnAir/2, p.txNodeId, str(p.seq), horizontalalignment='center', verticalalignment='center', fontsize=12)
		for p in packets:  # receptions
			if p.seq in [m.seq for m in t]:
				for rxId, bool in enumerate(p.receivedAtN):
					if bool:
						plt.barh(rxId, p.timeOnAir, left=p.startTime, color='green', edgecolor='green')
		maxTime = 0
		for m in t:  # message generations
			plt.arrow(m.genTime, m.origTxNodeId - 0.4, 0, 0.5, head_width=0.02 * (m.endTime - m.genTime), head_length=0.3, fc='k', ec='k')
			plt.text(m.genTime, m.origTxNodeId + 0.51, str(m.seq), horizontalalignment='center', verticalalignment='center', fontsize=12)
		maxTime = max([m.endTime for m in t])
		minTime = min([m.genTime for m in t])

		plt.xlabel('Time (ms)')
		plt.ylabel('Node ID')
		plt.yticks([0] + list(range(conf.NR_NODES)), label=[str(n) for n in [0] + list(range(conf.NR_NODES))])
		plt.xlim(minTime - 0.03 * (maxTime - minTime), maxTime)
		plt.show()

	# combine all messages with overlapping packets in one time sequence
	overlapping = [[m] for m in messages]
	for m in messages:
		m.endTime = max([p.endTime for p in packets if p.seq == m.seq])
	for m1 in messages:
		for m2 in messages:
			if m1 != m2:
				if m2.genTime <= m1.endTime and m2.endTime > m1.genTime:
					overlapping[m1.seq - 1].append(m2)
	timeSequences = []
	multiples = [[] for _ in overlapping]
	for ind, o1 in enumerate(overlapping):
		for o2 in overlapping:
			if set(o1).issubset(set(o2)):
				multiples[ind].append(set(o2))
		maxSet = max(multiples[ind], key=len)
		if maxSet not in timeSequences:
			timeSequences.append(maxSet)
	# do not plot time sequences with messages that were only generated but not sent
	timeSequences = [t for t in timeSequences if max([m.endTime for m in t]) != 0]

	# plot each time sequence
	fig = plt.figure()
	move_figure(fig, 900, 200)

	def onclick(event):
		if event.dblclick:
			global scheduleIdx
			plt.cla()
			scheduleIdx += 1
			if scheduleIdx < len(timeSequences):
				draw_schedule(scheduleIdx)
			else:
				plt.close('all')

	fig.canvas.mpl_connect('button_press_event', onclick)
	draw_schedule(0)


def move_figure(fig, x, y):
	fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))


class Graph:
	def __init__(self, conf):
		self.conf = conf
		self.xmax = conf.XSIZE / 2 + 1
		self.ymax = conf.YSIZE / 2 + 1
		self.packets = []
		self.fig, self.ax = plt.subplots()
		plt.suptitle('Placement of {} nodes'.format(conf.NR_NODES))
		self.ax.set_xlim(-self.xmax + conf.OX, self.xmax + conf.OX)
		self.ax.set_ylim(-self.ymax + conf.OY, self.ymax + conf.OY)
		self.ax.set_xlabel('x (m)')
		self.ax.set_ylabel('y (m)')
		move_figure(self.fig, 200, 200)

		# --- new: keep track of plot elements ---
		self.node_circles = {}
		self.node_markers = {}
		# If you want labels (text annotations) also updated:
		self.node_labels = {}

	def update_positions(self, nodes):
		for node in nodes:
			node_id = node.nodeid

			# 1) Update the marker
			marker = self.node_markers[node_id]
			marker.set_xdata([node.x])
			marker.set_ydata([node.y])

			# 2) Update the circle center
			circle = self.node_circles[node_id]
			circle.center = (node.x, node.y)

			# 3) (Optional) Update the text label, if you have one
			if node_id in self.node_labels:
				self.node_labels[node_id].set_position((node.x - 5, node.y + 5))

		# 4) Redraw the canvas
		self.fig.canvas.draw_idle()
		# A short pause to let the UI update
		plt.pause(0.01)

	def add_node(self, node):
		# place the node with label, marker, and circle
		txt = self.ax.annotate(str(node.nodeid), (node.x - 5, node.y + 5))
		self.node_labels[node.nodeid] = txt

		# Plot the node marker
		(marker,) = self.ax.plot(
			node.x, node.y,
			marker="o", markersize=2.5, color="grey"
		)
		self.node_markers[node.nodeid] = marker

		# Plot the coverage circle
		circle = plt.Circle(
			(node.x, node.y),
			radius=phy.estimate_max_range(node.antennaGain),
			color=plt.cm.Set1(node.nodeid),
			alpha=0.1
		)
		self.ax.add_patch(circle)
		self.node_circles[node.nodeid] = circle

		self.fig.canvas.draw_idle()
		plt.pause(0.1)

	def save(self):
		os.makedirs(os.path.join("out", "graphics"), exist_ok=True)
		plt.savefig(os.path.join("out", "graphics", "placement_" + str(self.conf.NR_NODES)))


def setup_asymmetric_links(conf, nodes):
	asymLinkRng = random.Random(conf.SEED)
	totalPairs = 0
	symmetricLinks = 0
	asymmetricLinks = 0
	noLinks = 0
	for i in range(conf.NR_NODES):
		for b in range(conf.NR_NODES):
			if i != b:
				if conf.MODEL_ASYMMETRIC_LINKS:
					conf.LINK_OFFSET[(i, b)] = asymLinkRng.gauss(conf.MODEL_ASYMMETRIC_LINKS_MEAN, conf.MODEL_ASYMMETRIC_LINKS_STDDEV)
				else:
					conf.LINK_OFFSET[(i, b)] = 0

	for a in range(conf.NR_NODES):
		for b in range(conf.NR_NODES):
			if a != b:
				# Calculate constant RSSI in both directions
				nodeA = nodes[a]
				nodeB = nodes[b]
				distAB = calc_dist(nodeA.x, nodeB.x, nodeA.y, nodeB.y, nodeA.z, nodeB.z)
				pathLossAB = phy.estimate_path_loss(conf, distAB, conf.FREQ, nodeA.z, nodeB.z)

				offsetAB = conf.LINK_OFFSET[(a, b)]
				offsetBA = conf.LINK_OFFSET[(b, a)]

				rssiAB = conf.PTX + nodeA.antennaGain - pathLossAB - offsetAB
				rssiBA = conf.PTX + nodeB.antennaGain - pathLossAB - offsetBA

				canAhearB = (rssiAB >= conf.SENSMODEM[conf.MODEM])
				canBhearA = (rssiBA >= conf.SENSMODEM[conf.MODEM])

				totalPairs += 1
				if canAhearB and canBhearA:
					symmetricLinks += 1
				elif canAhearB or canBhearA:
					asymmetricLinks += 1
				else:
					noLinks += 1

	return totalPairs, symmetricLinks, asymmetricLinks, noLinks
