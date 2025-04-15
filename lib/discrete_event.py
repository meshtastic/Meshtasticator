import os

import pandas as pd
import simpy


def simReport(conf, data, subdir, param):
	fname = "simReport_{}_{}.csv".format(conf.MODEM, param)
	if not os.path.isdir(os.path.join("out", "report", subdir)):
		if not os.path.isdir("out"):
			os.mkdir("out")
		if not os.path.isdir(os.path.join("out", "report")):
			os.mkdir(os.path.join("out", "report"))
		os.mkdir(os.path.join("out", "report", subdir))
	df_new = pd.DataFrame(data)
	df_new.to_csv(os.path.join("out", "report", subdir, fname), index=False)


class BroadcastPipe:
	def __init__(self, env, capacity=simpy.core.Infinity):
		self.env = env
		self.capacity = capacity
		self.pipes = []

	def latency(self, packet):
		# wait time that packet is on the air
		yield self.env.timeout(packet.timeOnAir)
		if not self.pipes:
			raise RuntimeError('There are no output pipes.')
		events = [store.put(packet) for store in self.pipes]
		return self.env.all_of(events)

	def put(self, packet):
		self.env.process(self.latency(packet))
		# this mimics start of reception
		if not self.pipes:
			raise RuntimeError('There are no output pipes.')
		events = [store.put(packet) for store in self.pipes]
		return self.env.all_of(events)

	def get_output_conn(self):
		pipe = simpy.Store(self.env, capacity=self.capacity)
		self.pipes.append(pipe)
		return pipe
