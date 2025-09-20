from enum import Enum
import numpy as np


class Config:

    class ROUTER_TYPE(Enum):
        MANAGED_FLOOD = 'MANAGED_FLOOD'

    def __init__(self):
        self.MODEL = 5  # Path loss model to use (see README)

        self.XSIZE = 15000  # horizontal size of the area to simulate in m
        self.YSIZE = 15000  # vertical size of the area to simulate in m
        self.OX = 0.0  # origin x-coordinate
        self.OY = 0.0  # origin y-coordinate
        self.MINDIST = 10  # minimum distance between each node in the area in m

        self.GL = 0  # antenna gain of each node in dBi
        self.HM = 1.0  # height of each node in m

        ### Meshtastic specific ###
        self.hopLimit = 3  # default 3
        self.router = False  # set role of each node as router (True) or normal client (False)
        self.maxRetransmission = 3  # default 3 -- not configurable by Meshtastic
        ### End of Meshtastic specific ###

        self.ONE_SECOND_INTERVAL = 1000
        self.TEN_SECONDS_INTERVAL = self.ONE_SECOND_INTERVAL * 10
        self.ONE_MIN_INTERVAL = self.TEN_SECONDS_INTERVAL * 6
        self.ONE_HR_INTERVAL = self.ONE_MIN_INTERVAL * 60

        ### Discrete-event specific ###
        self.MODEM = 4  # LoRa modem to use: 0 = ShortFast, 1 = Short Slow, ... 7 = Very Long Slow (default 4 is LongFast)
        self.PERIOD = 100 * self.ONE_SECOND_INTERVAL  # mean period of generating a new message with exponential distribution in ms
        self.PACKETLENGTH = 40  # payload in bytes
        self.SIMTIME = 30 * self.ONE_MIN_INTERVAL  # duration of one simulation in ms
        self.INTERFERENCE_LEVEL = 0.05  # chance that at a given moment there is already a LoRa packet being sent on your channel, outside of the Meshtastic traffic. Given in a ratio from 0 to 1.
        self.COLLISION_DUE_TO_INTERFERENCE = False
        self.DMs = False  # Set True for sending DMs (with random destination), False for broadcasts
        # from RadioInterface.cpp RegionInfo regions[]
        self.regions = {
            "US": {"freq_start": 902e6, "freq_end": 928e6, "power_limit": 30},
            "EU433": {"freq_start": 433e6, "freq_end": 434e6, "power_limit": 12},
            "EU868": {"freq_start": 868e6, "freq_end": 868e6, "power_limit": 27},
        }
        self.REGION = self.regions["US"]  # Select a different region here
        self.CHANNEL_NUM = 27  # Channel number

        self.PLOT = True # whether to plot the time schedule of packets after the simulation
        ### End of discrete-event specific ###

        ### PHY parameters (normally no change needed) ###
        self.PTX = self.REGION["power_limit"]
        # from RadioInterface::applyModemConfig()
        self.BWMODEM = np.array([250e3, 250e3, 250e3, 250e3, 250e3, 125e3, 125e3, 62.5e3])  # bandwidth
        self.SFMODEM = np.array([7, 8, 9, 10, 11, 11, 12, 12])  # spreading factor
        self.CRMODEM = np.array([8, 8, 8, 8, 8, 8, 8, 8])  # coding rate
        # minimum sensitivity from https://www.rfwireless-world.com/calculators/LoRa-Sensitivity-Calculator.html
        self.SENSMODEM = np.array([-121.5, -124.0, -126.5, -129.0, -131.5, -134.5, -137.0, -140.0])
        # minimum received power for CAD (3dB less than sensitivity)
        self.CADMODEM = np.array([-124.5, -127.0, -129.5, -132.0, -134.5, -137.5, -140.0, -143.0])
        self.FREQ = self.REGION["freq_start"]+self.BWMODEM[self.MODEM]*self.CHANNEL_NUM
        self.HEADERLENGTH = 16  # number of Meshtastic header bytes
        self.ACKLENGTH = 2  # ACK payload in bytes
        self.NOISE_LEVEL = -119.25  # some noise level in dB, based on SNR_MIN and minimum receiver sensitivity
        self.GAMMA = 2.08  # PHY parameter
        self.D0 = 40.0  # PHY parameter
        self.LPLD0 = 127.41  # PHY parameter
        self.NPREAM = 16   # number of preamble symbols from RadioInterface.h
        ### End of PHY parameters ###

        # Misc
        self.SEED = 44  # random seed to use
        # End of misc

        # Initializers
        self.NR_NODES = None
        # End of initializers

        ############################
        ####### ROUTER TYPE ########
        ############################
        # This can also be overwritten by scenarios defined in batchSim.py
        # or by passing this as the second command line param to loraMesh.py
        self.SELECTED_ROUTER_TYPE = self.ROUTER_TYPE.MANAGED_FLOOD

        #####################################################
        ####### ASYMMETRIC LINK SIMULATION VARIABLES ########
        #####################################################
        # Set this to True to enable the asymmetric link model
        # Adds a random offset to the link quality of each link
        self.MODEL_ASYMMETRIC_LINKS = True
        self.MODEL_ASYMMETRIC_LINKS_MEAN = 0
        self.MODEL_ASYMMETRIC_LINKS_STDDEV = 3
        # Stores the offset for each link
        # Populated when the simulator first starts
        self.LINK_OFFSET = {}

        #################################################
        ####### MOVING NODE SIMULATION VARIABLES ########
        #################################################
        self.MOVEMENT_ENABLED = True
        # The average number of meters a human walks in a minute
        self.WALKING_METERS_PER_MIN = 96
        # The average number of meters a human bikes in a minute
        self.BIKING_METERS_PER_MIN = 390
        # The average number of meters a human drives in a minute
        self.DRIVING_METERS_PER_MIN = 1500
        # The % of nodes that end up mobile in the simulation 0.4 = ~40%
        self.APPROX_RATIO_NODES_MOVING = 0.3
        # The % of mobile nodes that have GPS enabled 0.5 = 50%
        self.APPROX_RATIO_OF_NODES_MOVING_W_GPS_ENABLED = 0.3

        # 100 meters
        self.SMART_POSITION_DISTANCE_THRESHOLD = 100
        # 30s minimum time in firmware
        self.SMART_POSITION_DISTANCE_MIN_TIME = 30 * self.ONE_SECOND_INTERVAL
        # This mirrors the firmware's approach to monitoring channel utilization
        self.CHANNEL_UTILIZATION_PERIODS = 6

    # Function that needs to be run to ensure the router dependent variables change appropriately
    def update_router_dependencies(self):
        # Example: Overwrite hop limit in the case of X new awesome routing algorithm
        # if self.SELECTED_ROUTER_TYPE == self.ROUTER_TYPE.AWESOME_ROUTER:
        #     Change config values if necessary for your router here
        return
