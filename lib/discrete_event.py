import os

import pandas as pd
import simpy


def sim_report(conf, data, subdir, param):
	os.makedirs(os.path.join("out", "report", subdir), exist_ok=True)
	fname = f"simReport_{conf.MODEM}_{param}.csv"
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
