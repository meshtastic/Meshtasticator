#!/usr/bin/env python3
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    matplotlib.use("TkAgg")
except ImportError:
    print('Tkinter is needed. Install python3-tk with your package manager.')

hopLimits = [1, 2, 3, 4, 5, 6, 7]
nrNodes = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25]
collisionRates = [[] for _ in hopLimits]
collisionStds = [[] for _ in hopLimits]
reachability = [[] for _ in hopLimits]
reachabilityStds = [[] for _ in hopLimits]
usefulness = [[] for _ in hopLimits]
usefulnessStds = [[] for _ in hopLimits]
meanDelays = [[] for _ in hopLimits]
delayStds = [[] for _ in hopLimits]
meanTxAirUtils = [[] for _ in hopLimits]
txAirUtilsStds = [[] for _ in hopLimits]


for hi, h in enumerate(hopLimits):
    for n in nrNodes:
        fname = "out/report/hopLimit"+str(h)+"/simReport_5_"+str(n)+".csv"
        data = pd.read_csv(fname, header=0)
        collisionRates[hi].append(np.nanmean(data["CollisionRate"]))
        collisionStds[hi].append(np.nanstd(data["CollisionRate"]))
        reachability[hi].append(np.nanmean(data["Reachability"]))
        reachabilityStds[hi].append(np.nanstd(data["Reachability"]))
        usefulness[hi].append(np.nanmean(data["Usefulness"]))
        usefulnessStds[hi].append(np.nanstd(data["Usefulness"]))
        meanDelays[h].append(np.nanmean(data["meanDelay"]))
        delayStds[h].append(np.nanstd(data["meanDelay"]))
        meanTxAirUtils[h].append(np.nanmean(data["meanTxAirUtil"]))
        txAirUtilsStds[h].append(np.nanstd(data["meanTxAirUtil"]))

for hi, h in enumerate(hopLimits):
    plt.errorbar(nrNodes, collisionRates[hi], collisionStds[hi], color=plt.cm.Set1(h), capsize=3, elinewidth=0.5, capthick=0.5, label=str(h))
plt.legend(title="HopLimit")
plt.xlabel("#nodes")
plt.ylabel("Collision rate (%)")

plt.figure()
for hi, h in enumerate(hopLimits):
    plt.errorbar(nrNodes, reachability[hi], reachabilityStds[hi], color=plt.cm.Set1(h), capsize=3, elinewidth=0.5, capthick=0.5, label=str(h))
plt.legend(title="HopLimit")
plt.xlabel("#nodes")
plt.ylabel("Reachability (%)")

plt.figure()
for hi, h in enumerate(hopLimits):
    plt.errorbar(nrNodes, usefulness[hi], usefulnessStds[hi], color=plt.cm.Set1(h), capsize=3, elinewidth=0.5, capthick=0.5, label=str(h))
plt.legend(title="HopLimit")
plt.xlabel("#nodes")
plt.ylabel("Usefulness (%)")

plt.figure()
for hi, h in enumerate(hopLimits):
    plt.errorbar(nrNodes, meanDelays[hi], delayStds[hi], color=plt.cm.Set1(h), capsize=3, elinewidth=0.5, capthick=0.5, label=str(h))
plt.legend(title="HopLimit")
plt.xlabel("#nodes")
plt.ylabel("Average delay (ms)")

plt.figure()
for hi, h in enumerate(hopLimits):
    plt.errorbar(nrNodes, meanTxAirUtils[hi], txAirUtilsStds[hi], color=plt.cm.Set1(h), capsize=3, elinewidth=0.5, capthick=0.5, label=str(h))
plt.legend(title="HopLimit")
plt.xlabel("#nodes")
plt.ylabel("Average Tx air utilization per node (ms)")

plt.show()
