# Discrete-event simulator
The discrete-event simulator mimics the radio section of the device software. It is currently based on Meshtastic 2.1.

## Usage
Please `git clone` or download this repository, navigate to the Meshtasticator folder (optionally create a virtual environment) and install the necessary requirements using:
```pip install -r requirements.txt```.

To start one simulation with the default configurations, run:

```python3 loraMesh.py [nr_nodes]```

If no argument is given, you first have to place the nodes on a plot. After you place a node, you can change its [role](https://meshtastic.org/docs/settings/config/device#role), hopLimit, antenna height above local ground, and antenna gain. These settings will automatically save when you place a new node or when you start the simulation.

![](/img/configNode.png)

If the number of nodes is given, it will randomly place nodes in the area. It makes sure that each node can reach at least one other node. Furthermore, all nodes are placed at a configurable minimum distance (MINDIST) from each other.

For non-interactive smoke tests or CI runs, pass `--no-gui` together with either a node count or `--from-file`. This skips the Tk/Matplotlib placement graph and the final schedule plot while keeping the simulation logic unchanged:

```python3 loraMesh.py 10 --no-gui```

Short deterministic smoke runs can also override the configured duration and message period from the command line:

```python3 loraMesh.py 2 --no-gui --simtime-seconds 5 --period-seconds 0.5```

The same headless path can import positioned real-mesh nodes. `--from-map`
reads a Meshtastic map `/api/v1/nodes` JSON endpoint; the public default is
`https://meshtastic.liamcottle.net/api/v1/nodes`, but you can pass another
compatible endpoint URL. These map endpoints usually return a broad node list,
so pass a local area-of-interest bounding box. `--map-bbox` uses the common
`min_lat,min_lon,max_lat,max_lon` order that most GIS tools call
`south,west,north,east`; you can copy those four numbers from OpenStreetMap's
Export panel, geojson.io's bbox readout, QGIS, or any other tool that shows the
extent of the map view or selected polygon. Keep the box tight enough for the
local scenario you want to simulate:

```python3 loraMesh.py --from-map https://meshtastic.liamcottle.net/api/v1/nodes --map-bbox 41.50,41.50,41.82,41.86 --map-limit 50 --no-gui```

You can also import positioned nodes from the NodeDB cached by a local
Meshtastic device. This uses the Python client `interface.nodesByNum` data that
backs `meshtastic --nodes`, not the pretty-printed table. Use TCP for a network
device, or omit `--nodedb-host` to use Meshtastic serial auto-detection. For a
quick local-device run, pass the device address and cap the imported node count:

```python3 loraMesh.py --from-nodedb --nodedb-host 192.168.1.23 --map-limit 50 --no-gui```

NodeDB often contains old or far-away positions. Add `--map-bbox` when you want
to restrict the run to one local area:

```python3 loraMesh.py --from-nodedb --nodedb-host 192.168.1.23 --map-bbox 41.50,41.50,41.82,41.86 --map-limit 50 --no-gui```

Imported nodes use the same `HM` antenna height and `hopLimit` defaults as
generated and file-backed scenarios. Change those config values when the
position source does not carry the simulation value you want.

Terrain obstruction can be added to map, NodeDB, or origin-backed scenario inputs
without creating a custom terrain file. `--terrain-srtm` downloads missing SRTM
HGT tiles from Mapzen Terrain Tiles on AWS into a local cache and feeds the
terrain grid directly into terrain-aware node geometry:

```python3 loraMesh.py --from-nodedb --nodedb-host 192.168.1.23 --map-limit 50 --terrain-srtm --no-gui```

With an explicit `--map-bbox`, SRTM samples that whole requested rectangle. When
the terrain bbox is derived from imported or file-backed nodes, Meshtasticator
keeps the download smaller: it loads tiles around the selected nodes and along
flat-link candidate paths, instead of downloading every tile in a large
edge-to-edge rectangle. When publishing screenshots, reports, or derived
datasets from this terrain source, attribute the terrain data to
[Mapzen Terrain Tiles on AWS](https://registry.opendata.aws/terrain-tiles/),
SRTM/NASA, and their underlying open elevation sources:

```python3 loraMesh.py --from-map https://meshtastic.liamcottle.net/api/v1/nodes --map-bbox 41.50,41.50,41.82,41.86 --map-limit 50 --terrain-srtm --no-gui```

Map payload `altitude` values are absolute GPS/MSL altitude, not antenna height,
so map import keeps using `HM` as the fallback antenna height above local
ground. When `--terrain-srtm` is enabled, each map node is checked
against its own SRTM ground sample: plausible positive map altitudes are used as
absolute node altitude, while missing, below-ground, or implausibly high values
fall back to `SRTM ground + antenna height` for 3D distance calculations.

Land-cover clutter is a separate optional CSV grid. Use it for broad urban,
open, water, or forest excess-loss inputs without pretending Meshtasticator is a
building-level ray tracer:

```python3 loraMesh.py --from-file nodeConfig.yaml --terrain-srtm --clutter-grid clutter.csv --no-gui```

`tools/osm_to_clutter_csv.py` can build a coarse clutter grid from public
OpenStreetMap building, landuse, natural, and water polygons. The simulator
never fetches OpenStreetMap data implicitly.

Two optional RF models can make dense or weak-link runs less binary:

```python3 loraMesh.py 20 --no-gui --phy-loss-model --capture-collision-model```

`--phy-loss-model` keeps RSSI/sensitivity as the hearability gate, then applies
a smooth SNR-to-payload-success curve that depends on packet size and LoRa
coding rate. `--capture-collision-model` keeps CAD-detectable but undecodable
packets on the RF timeline as interference energy, and uses capture/preamble
overlap rules instead of treating every overlap as identical.

Packaged real-mesh presets can be listed and loaded directly:

```python3 loraMesh.py --list-presets```

The `batumi` preset includes sanitized Batumi/Georgia-area node geometry, a
matching bundled terrain grid, an OpenStreetMap-derived land-cover clutter grid,
and an aggregate radio calibration over generated path features. Terrain,
clutter, and the fitted link-calibration model are enabled automatically for the
preset; use `--terrain-srtm` for a fresh SRTM terrain sample, `--clutter-grid`
for a different land-cover grid, or `--no-clutter` for old-style comparison
runs. The calibration report is in `docs/batumi_radio_calibration.md`.

```python3 loraMesh.py --preset batumi --no-gui --simtime-seconds 5 --period-seconds 2 --phy-loss-model --capture-collision-model```

Dynamic Coding Rate is opt-in and chooses LoRa CR 4/5..4/8 per outgoing packet
without changing the preset's SF or bandwidth:

```python3 loraMesh.py 20 --no-gui --phy-loss-model --capture-collision-model --dcr```

The policy keeps ordinary first-attempt traffic compact, spends extra FEC on
quiet retries, ACKs, non-busy direct relays, and last-hop relays, then records
`dcrTxByCr` and `dcrAirtimeByCr` in simulation results. This keeps idle airtime
as a reserve instead of turning every quiet packet into CR 4/8.

Dynamic TX Power is also opt-in:

```python3 loraMesh.py 20 --no-gui --capture-collision-model --dtp```

DTP keeps configured `PTX` as the maximum regional/base power and only applies
temporary reductions just before transmission. Origin packets stay at max power;
relay packets may shrink power when channel pressure is high or the prior hop
was strong enough. Final retries and CR 4/8 rescue packets stay at full power so
the interference-reduction knob does not fight the reliability knob.

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
| Modem | Name | Bandwidth (kHz) | Base coding rate | Spreading Factor | Nominal data rate (kbps) |
|--|--|--:|--:|--:|--:|
| 0 | Short Turbo | 500 | 4/5 | 7 | 21.9 |
| 1 | Short Fast | 250 | 4/5 | 7 | 10.9 |
| 2 | Short Slow | 250 | 4/5 | 8 | 6.25 |
| 3 | Medium Fast | 250 | 4/5 | 9 | 3.52 |
| 4 | Medium Slow | 250 | 4/5 | 10 | 1.95 |
| 5 | Long Turbo | 500 | 4/8 | 11 | 1.34 |
| 6 | Long Fast | 250 | 4/5 | 11 | 1.07 |
| 7 | Long Moderate | 125 | 4/8 | 11 | 0.336 |
| 8 | Long Slow | 125 | 4/8 | 12 | 0.183 |
| 9 | Very Long Slow | 62.5 | 4/8 | 12 | 0.0916 |

The simulator stores coding rates as their LoRa denominators (`5` through
`8`, meaning CR 4/5 through 4/8). This table shows the configured base CR; when
`--dcr` is enabled, the simulator may select a different CR for each outgoing
packet while leaving the preset's SF and bandwidth unchanged.

DCR and DTP can be combined. DCR changes airtime and forward-error-correction
strength; DTP changes how many receivers can CAD-detect, demodulate, or collide
with the packet.

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
