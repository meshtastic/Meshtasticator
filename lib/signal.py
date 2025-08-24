"""
Signal computation utilities for Meshtastic simulation.

This module provides centralized signal calculation functions to ensure
consistency across all simulation components.

All functions here are deterministic - randomness/noise is handled
in the path loss modeling functions.
"""

import numpy as np
from .common import calc_dist


def calculate_link_rssi(tx_config, rx_config, path_loss):
    """
    Calculate RSSI for a link between two nodes (deterministic).
    
    Args:
        tx_config (dict): Transmitter node configuration with 'ptx' and 'antennaGain'
        rx_config (dict): Receiver node configuration with 'antennaGain'  
        path_loss (float): Path loss in dB (may include random effects from path loss model)
        
    Returns:
        float: RSSI in dBm
    """
    return tx_config['ptx'] + tx_config['antennaGain'] + rx_config['antennaGain'] - path_loss


def can_nodes_communicate(tx_config, rx_config, path_loss, sensitivity_threshold):
    """
    Check if two nodes can communicate based on signal strength (pure function).
    
    Args:
        tx_config (dict): Transmitter node configuration
        rx_config (dict): Receiver node configuration
        path_loss (float): Path loss in dB (includes any random effects)
        sensitivity_threshold (float): Receiver sensitivity in dBm
        
    Returns:
        bool: True if nodes can communicate
    """
    rssi = calculate_link_rssi(tx_config, rx_config, path_loss)
    return rssi >= sensitivity_threshold


def analyze_link_quality(tx_config, rx_config, distance, conf, path_loss_func):
    """
    Complete link analysis with RSSI, path loss, and connectivity.
    
    Args:
        tx_config (dict): Transmitter node configuration
        rx_config (dict): Receiver node configuration  
        distance (float): Distance between nodes in meters
        conf: Configuration object with sensitivity and frequency settings
        path_loss_func: Function to calculate path loss (handles randomness)
        
    Returns:
        dict: Link analysis results containing:
            - distance: Distance in meters
            - path_loss: Path loss in dB
            - rssi: RSSI in dBm
            - can_communicate: Boolean connectivity
            - signal_margin: dB above/below sensitivity threshold
    """
    # Create mock nodes for path loss calculation if needed
    if hasattr(path_loss_func, '__call__'):
        # For functions that need node objects
        class MockNode:
            def __init__(self, config):
                self.nodeConfig = config
                self.isRouter = config.get('isRouter', False)
                self.z = config.get('z', 1.5)
        
        tx_node = MockNode(tx_config)
        rx_node = MockNode(rx_config)
        path_loss = path_loss_func(conf, tx_node, rx_node, distance, conf.FREQ)
    else:
        # For simple path loss values
        path_loss = path_loss_func
    
    rssi = calculate_link_rssi(tx_config, rx_config, path_loss)
    sensitivity = conf.current_preset["sensitivity"]
    can_communicate = rssi >= sensitivity
    signal_margin = rssi - sensitivity
    
    return {
        'distance': distance,
        'path_loss': path_loss,
        'rssi': rssi,
        'can_communicate': can_communicate,
        'signal_margin': signal_margin,
        'tx_config': tx_config,
        'rx_config': rx_config
    }


def calculate_node_distance(tx_config, rx_config):
    """
    Calculate 3D distance between two nodes.
    
    Args:
        tx_config (dict): Transmitter node configuration with x, y, z
        rx_config (dict): Receiver node configuration with x, y, z
        
    Returns:
        float: Distance in meters
    """
    return calc_dist(
        tx_config['x'], rx_config['x'],
        tx_config['y'], rx_config['y'], 
        tx_config['z'], rx_config['z']
    )


def analyze_link(tx_config, rx_config, conf, path_loss_func):
    """
    Analyze a link between any two nodes using standard simulation methodology.
    
    Args:
        tx_config (dict): Transmitter node configuration
        rx_config (dict): Receiver node configuration
        conf: Configuration object
        path_loss_func: Path loss calculation function (handles randomness)
        
    Returns:
        dict: Complete link analysis
    """
    distance = calculate_node_distance(tx_config, rx_config)
    return analyze_link_quality(tx_config, rx_config, distance, conf, path_loss_func)


def get_signal_strength_category(rssi):
    """
    Categorize signal strength for visualization.
    
    Args:
        rssi (float): RSSI in dBm
        
    Returns:
        dict: Signal category with color, thickness, alpha, and label
    """
    if rssi >= -80:
        return {
            'color': 'green',
            'thickness': 2.0,
            'alpha': 0.8,
            'label': 'Strong'
        }
    elif rssi >= -100:
        return {
            'color': 'yellow', 
            'thickness': 1.5,
            'alpha': 0.6,
            'label': 'Medium'
        }
    elif rssi >= -120:
        return {
            'color': 'red',
            'thickness': 1.0,
            'alpha': 0.4, 
            'label': 'Weak'
        }
    else:
        return {
            'color': 'gray',
            'thickness': 0.5,
            'alpha': 0.2,
            'label': 'Very Weak'
        }