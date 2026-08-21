# Meshtasticator
Discrete-event and interactive simulator for [Meshtastic](https://meshtastic.org/). 

## Quick start

Install the Python dependencies, then ask the CLI what runnable scenarios it
already knows about:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

./loraMesh.py --list-presets
./loraMesh.py --list-modem-presets
```

Run the packaged Batumi/Georgia-area radio scenario headlessly:

```bash
./loraMesh.py --preset batumi --no-gui --simtime-seconds 60 --period-seconds 5
```

Run the radio-physics comparison workflow in one command:

```bash
python3 tools/radio_policy_compare.py --policies static,dcr,dtp --simtime-seconds 60 --period-seconds 5
```

For CI-style runs, write JSON/Markdown artifacts:

```bash
python3 tools/radio_policy_compare.py \
  --simtime-seconds 120 \
  --period-seconds 5 \
  --json-output out/radio_policy_compare.json \
  --markdown-output out/radio_policy_compare.md
```

Threshold flags such as `--max-reach-drop-pp` are accepted only when a later
policy such as `dcr` or `dtp` is compared against the `static` baseline.

For manual radio-physics experiments, keep the same scenario and traffic load
while enabling packet loss and capture-aware collisions:

```bash
./loraMesh.py --preset batumi --no-gui --simtime-seconds 60 --period-seconds 5 \
  --phy-loss-model --capture-collision-model
```

See [Radio Physics Quickstart](docs/radio_physics_quickstart.md) for what the
flags mean and which result fields to compare.

## Discrete-event simulator
The discrete-event simulator mimics the radio section of the device software in order to understand its working. It can also be used to assess the performance of your scenario, or the scalability of the protocol. 

See [this document](DISCRETE_EVENT_SIM.md) for a usage guide. 

After a simulation, it plots the placement of nodes and time schedule for each set of overlapping messages that were sent.

![](/img/placement_schedule.png)

It can be used to analyze the network for a set of parameters. For example, these are the results of 100 simulations of 200s with a different hop limit and number of nodes. As expected, the average number of nodes reached for each generated message increases as the hop limit increases. 

![](/img/reachability_hops.png)

However, it comes at the cost of usefulness, i.e., the amount of received packets that contain a new message (not a duplicate due to rebroadcasting) out of all packets received. 

![](/img/usefulness_hops.png)

## Interactive simulator
The interactive simulator uses the [Linux native application of Meshtastic](https://meshtastic.org/docs/development/linux/), i.e. the real device software, while simulating some of the hardware interfaces, including the LoRa chip. Can also be used on a Windows or macOS host with Docker.

See [this document](INTERACTIVE_SIM.md) for a usage guide. 

It allows for debugging multiple communicating nodes without having real devices. 

https://user-images.githubusercontent.com/78759985/209952664-1a571fc8-65d1-4277-8516-2822f60a5dd0.mp4

Furthermore, since the simulator has an 'oracle view' of the network, it allows to visualize the route messages take. 

![](/img/route_plot.png)

# Tests

Unit tests can be executed by running `python3 -m unittest` from the root of the repo. Don't forget to activate your virtual env before running tests.

## License
Part of the source code is based on the work in [1], which eventually stems from [2]. The LoRaSim library from [2] can be found [here](https://www.lancaster.ac.uk/scc/sites/lora/lorasim.html).

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). 

## References
1. [S. Spinsante, L. Gioacchini and L. Scalise, "A novel experimental-based tool for the design of LoRa networks," 2019 II Workshop on Metrology for Industry 4.0 and IoT (MetroInd4.0&IoT), 2019, pp. 317-322, doi: 10.1109/METROI4.2019.8792833.](https://ieeexplore.ieee.org/document/8792833)
2. [Martin C. Bor, Utz Roedig, Thiemo Voigt, and Juan M. Alonso, "Do LoRa Low-Power Wide-Area Networks Scale?", In Proceedings of the 19th ACM International Conference on Modeling, Analysis and Simulation of Wireless and Mobile Systems (MSWiM '16), 2016. Association for Computing Machinery, New York, NY, USA, 59–67.](https://doi.org/10.1145/2988287.2989163)
