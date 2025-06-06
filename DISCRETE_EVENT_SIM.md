# Discrete-event simulator 
The discrete-event simulator mimics the radio section of the device software. It is currently based on Meshtastic 2.1. 

## Usage
Please `git clone` or download this repository, navigate to the Meshtasticator folder (optionally create a virtual environment) and install the necessary requirements using: 
```pip install -r requirements.txt```.

To start one simulation with the default configurations, run:

```python3 loraMesh.py [nr_nodes]``` 

If no argument is given, you first have to place the nodes on a plot. After you place a node, you can change its [role](https://meshtastic.org/docs/settings/config/device#role), hopLimit, height (elevation) and antenna gain. These settings will automatically save when you place a new node or when you start the simulation.

![](/img/configNode.png)

If the number of nodes is given, it will randomly place nodes in the area. It makes sure that each node can reach at least one other node. Furthermore, all nodes are placed at a configurable minimum distance (MINDIST) from each other. 

If you placed the nodes yourself, after a simulation the number of nodes, their coordinates and configuration are automatically saved and you can rerun the scenario with:

 ```python3 loraMesh.py --from-file```

If you want to change any of the configurations, adapt the file *out/nodeConfig.yaml* before running it with the above command.

For running multiple repetitions of simulations for a set of parameters, e.g. the number of nodes, run: 

```python3 batchSim.py``` 

After the simulations are done, it plots relevant metrics obtained from the simulations. It saves these metrics in */out/report/* to analyze them later on. See *plotExample.py* for an example Python script to plot the results.  

To simulate different parameters, you will have to change the *batchSim.py* script yourself. 

## Custom configurations
Here we list some of the configurations, which you can change to model your scenario in */lib/config.py*. These apply to all nodes, except those that you configure per node when using the plot.
### Modem
The LoRa modem ([see Meshtastic radio settings](https://meshtastic.org/docs/overview/radio-settings#predefined-channels)) that is used, as defined below:
|Modem  | Name | Bandwidth (kHz) | Coding rate | Spreading Factor | Data rate (kbps)
|--|--|--|--|--|--| 
| 0 |Short Fast|250|4/8|7|6.8
| 1 |Short Slow|250|4/8|8|3.9
| 2 |Mid Fast|250|4/8|9|2.2
| 3 |Mid Slow|250|4/8|10|1.2
| 4 |Long Fast|250|4/8|11|0.67
| 5 |Long Moderate|125|4/8|11|0.335
| 6 |Long Slow|125|4/8|12|0.18
| 7 |Very Long Slow|62.5|4/8|12|0.09

### Period
Mean period (in ms) with which the nodes generate a new message following an exponential distribution. E.g. if you set it to 300s, each node will generate a message on average once every five minutes. 

### Packet length 
Payload size of each generated message in bytes. For a position packet, it will be around 40 bytes. 

### Model
This feature is referred to the path loss model, i.e. what the simulator uses to calculate how well a signal will propagate. Note that this is only a rough estimation of the physical environment and will not be 100% accurate, as it depends on a lot of factors. The implemented pathloss models are:
* ```0``` set the log-distance model  
* ```1``` set the Okumura-Hata for small and medium-size cities model  
* ```2``` set the Okumura-Hata for metropolitan areas  
* ```3``` set the Okumura-Hata for suburban environments
* ```4``` set the Okumura-Hata for rural areas  
* ```5``` set the 3GPP for suburban macro-cell  
* ```6``` set the 3GPP for metropolitan macro-cell  

### Broadcasts or direct messages (DMs)
By default, *DMs* is set to False, meaning it will send broadcast messages only. If you set it to True, each node will only send DMs to a random other node in the network.

## Explanation
A discrete-event simulator jumps from event to event over time, where an event is a change in the state of the system. It is therefore well-suited for simulating communication networks.

For every node in the simulation, an instance is created that mimics the [Meshtastic logic](https://meshtastic.org/docs/overview/mesh-algo). Each node runs three processes in parallel: *generateMessage*, *transmit* and *receive*. The first creates an event by constructing a new message with unique sequence number at a random time, taken from an exponential distribution. For now, each generated message is of the same payload size. The second and third processes model the actual transmitting and receiving behavior, respectively. 

The model of the LoRa physical (PHY) layer is in */lib/phy.py*. Depending on the modem used, it is calculated what the airtime of a packet is. The PHY layer uses a configurable pathloss model to estimate whether nodes at a specific distance can sense each other's packets. Furthermore, it determines whether two packets collide, which depends on the frequency, spreading factor, received time and received power of the two packets.  

The routing behavior is implemented in each of the processes of the node. Inside *generateMessage*, reliable retransmissions are handled if no implicit acknowledgement is received. A MeshPacket (defined in */lib/packet.py*) is created to transfer the message. Note that there may be multiple packets created containing the same message, due to retransmissions and rebroadcasting. In *receive*, it is decided what to do on reception of a packet. A packet is flooded if its hoplimit is not zero and no rebroadcast of this packet was heard before. In *transmit*, delays of the Medium Access Control (MAC) layer are called from */lib/mac.py*. The MAC uses a listen-before-talk mechanism, including introducing (random or SNR-based) delays before transmitting a packet. When a packet is ready to be transferred over the air, it is first checked whether in the meantime still no acknowledgement was received, otherwise the transmission is canceled.

The actual communication between processes of different nodes is handled by a BroadcastPipe of [Simpy](https://simpy.readthedocs.io/en/latest/examples/process_communication.html). This ensures that a transmitted packet by one node creates events (one at the start of a packet and one at the end) at the receiving nodes. 