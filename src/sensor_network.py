#!/usr/bin/env python3
"""
sensor_network.py

A module for simulating a wireless sensor network deployed in a w x w area.
Nodes are deployed uniformly at random, and an edge is added between two nodes if
they are within a specified communication radius. A special base station is added,
and a spanning tree (using BFS) rooted at the base station is computed.

This version supports:
  - Temporal correlation using an AR(1) model.
  - Spatial correlation via a smooth bias function.
  - Heterogeneous sensors: honest sensors can follow a correlated Gaussian (AR(1)) or Uniform model;
    adversarial nodes generate data with extreme variance.
"""

import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc
import pandas as pd

# Set plotting style.
plt.style.use('seaborn-whitegrid')
#rc('text', usetex=True)
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-ticks")


#############################
# DataStream Base and Derived Classes
#############################

class DataStream:
    """Base class for a sensor's data stream."""

    def __init__(self, d):
        self.d = d

    def get_next(self):
        """Return the next d-dimensional data vector."""
        raise NotImplementedError("Subclasses should implement get_next().")


class CorrelatedGaussianDataStream(DataStream):
    """
    Generates data from a correlated Gaussian process.
    Each new vector is generated via an AR(1) process:
       v(t) = bias + rho*(v(t-1) - bias) + noise,
    where noise ~ N(0, sigma^2 I). The spatial bias is computed as a function
    of the sensor's (x,y) position.
    """

    def __init__(self, d, x, y, rho=0.8, sigma=5.0, w=100.0):
        """
        Parameters:
          d    : Dimension of the vector.
          x, y : Sensor's (x,y) position.
          rho  : AR(1) coefficient (0 < rho < 1).
          sigma: Standard deviation of the noise.
          w    : Width of the deployment area (used for spatial bias computation).
        """
        super().__init__(d)
        self.rho = rho
        self.sigma = sigma
        self.w = w
        self.x = x
        self.y = y
        # Initialize with the spatial bias.
        self.prev = self.spatial_bias()

    def spatial_bias(self):
        """
        Compute a spatial bias based on the sensor's position.
        For example, bias = 50 + 10*sin(2*pi*x/w)*cos(2*pi*y/w).
        """
        a = random.randint(45, 60)
        b = random.randint(10, 30)
        return 1.0 * a + 1.0 * b * math.sin(2 * math.pi * self.x / self.w) * math.cos(2 * math.pi * self.y / self.w)

    def get_next(self):
        noise = np.random.normal(0, self.sigma, size=self.d)
        bias = self.spatial_bias()
        new_val = bias + self.rho * (self.prev - bias) + noise
        self.prev = new_val
        return new_val.tolist()


class UniformDataStream(DataStream):
    """Generates data uniformly from a specified range."""

    def __init__(self, d, low=0.0, high=100.0):
        super().__init__(d)
        self.low = low
        self.high = high

    def get_next(self):
        return np.random.uniform(self.low, self.high, size=self.d).tolist()


class AdversarialDataStream(DataStream):
    """
    Generates adversarial data from a Gaussian distribution with a fixed mean of 0.0.
    The standard deviation is specified by std_dev.
    """

    def __init__(self, d, std_dev, mean=0.0):
        super().__init__(d)
        self.mean = mean
        self.std_dev = std_dev

    def get_next(self):
        return np.random.normal(loc=self.mean, scale=self.std_dev, size=self.d).tolist()


#############################
# Sensor Node and Base Station
#############################

class SensorNode:
    """
    Represents a wireless sensor node with an ID, position, and an associated data stream.
    """

    def __init__(self, node_id, x, y, data_stream: DataStream, weight=1.0):
        self.id = node_id
        self.x = x
        self.y = y
        self.data_stream = data_stream
        self.weight = weight

    def position(self):
        return (self.x, self.y)

    def get_data(self):
        return self.data_stream.get_next()

    def __repr__(self):
        return f"SensorNode({self.id}, pos=({self.x:.2f}, {self.y:.2f}))"


class BaseStation:
    """
    Represents the base station.
    """

    def __init__(self, node_id, x, y, d):
        self.id = node_id
        self.x = x
        self.y = y
        self.d = d
        # For simplicity, base station data is static.
        self.data = np.zeros(d)

    def position(self):
        return (self.x, self.y)

    def get_data(self):
        return self.data.tolist()

    def __repr__(self):
        return f"BaseStation({self.id}, pos=({self.x:.2f}, {self.y:.2f}))"


#############################
# Sensor Network and Graph Construction
#############################

class SensorNetwork:
    """
    Represents a sensor network deployed in a w x w area.
    Nodes are deployed uniformly at random.
    Two nodes are connected if they are within the communication radius.
    A base station (default at the center) is added, and a spanning tree (using BFS)
    rooted at the base station is computed.

    Honest sensors use either a correlated Gaussian (AR(1)) or Uniform data stream.
    Adversarial nodes use an adversarial data stream with high variance.
    """

    def __init__(self, n, w, d, comm_radius, base_station_pos=None, seed=None,
                 adversarial_fraction=0.0, adversarial_std_dev=50.0, honest_distribution="correlated_gaussian"):
        """
        Parameters:
          n                   -- Number of sensor nodes (excluding base station).
          w                   -- Width of the deployment area ([0, w] x [0, w]).
          d                   -- Dimension of each sensor's data vector.
          comm_radius         -- Communication radius.
          base_station_pos    -- Base station position; defaults to center if None.
          seed                -- Random seed (optional).
          adversarial_fraction-- Fraction of nodes that are adversarial.
          adversarial_std_dev -- Standard deviation for adversarial nodes.
          honest_distribution -- "correlated_gaussian" or "uniform" for honest nodes.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.n = n
        self.w = w
        self.d = d
        self.comm_radius = comm_radius
        self.adversarial_fraction = adversarial_fraction
        self.adversarial_std_dev = adversarial_std_dev
        self.honest_distribution = honest_distribution

        self.nodes = []
        self.base_station = None
        self.graph = None
        self.spanning_tree = None

        self._deploy_nodes()
        self._add_base_station(base_station_pos)
        self._build_graph()
        self._ensure_connected()
        self._compute_spanning_tree()

    def _deploy_nodes(self):
        num_adv = int(self.adversarial_fraction * self.n)
        adversarial_ids = set(random.sample(range(self.n), num_adv))
        for i in range(self.n):
            x = random.uniform(0, self.w)
            y = random.uniform(0, self.w)
            if i in adversarial_ids:
                stream = AdversarialDataStream(self.d, std_dev=self.adversarial_std_dev, mean=0.0)
            else:
                if self.honest_distribution == "uniform":
                    stream = UniformDataStream(self.d, low=0.0, high=self.w)
                else:
                    stream = CorrelatedGaussianDataStream(self.d, x, y, rho=0.8, sigma=5.0, w=self.w)
            node = SensorNode(i, x, y, stream, weight=1.0)
            self.nodes.append(node)

    def _add_base_station(self, pos):
        if pos is None:
            pos = (self.w / 2.0, self.w / 2.0)
        # Base station uses a correlated Gaussian stream for consistency.
        self.base_station = SensorNode('base', pos[0], pos[1],
                                       CorrelatedGaussianDataStream(self.d, pos[0], pos[1], rho=0.8, sigma=5.0,
                                                                    w=self.w),
                                       weight=1.0)

    def _build_graph(self):
        self.graph = nx.Graph()
        # Add sensor nodes.
        for node in self.nodes:
            self.graph.add_node(node.id, pos=node.position(), type='sensor')
        # Add base station.
        self.graph.add_node(self.base_station.id, pos=self.base_station.position(), type='base')

        all_nodes = self.nodes + [self.base_station]
        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                pos_i = np.array(all_nodes[i].position())
                pos_j = np.array(all_nodes[j].position())
                if np.linalg.norm(pos_i - pos_j) <= self.comm_radius:
                    self.graph.add_edge(all_nodes[i].id, all_nodes[j].id)

    def _ensure_connected(self):
        if not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            for comp in components:
                if self.base_station.id not in comp:
                    node_id = next(iter(comp))
                    self.graph.add_edge(node_id, self.base_station.id)
            assert nx.is_connected(self.graph), "Graph is still not connected after adding base station edges."

    def _compute_spanning_tree(self):
        self.spanning_tree = nx.bfs_tree(self.graph, source=self.base_station.id)
        for node in self.spanning_tree.nodes():
            self.spanning_tree.nodes[node]['pos'] = self.graph.nodes[node]['pos']

    def _euclidean_distance(self, node1, node2):
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        return math.hypot(dx, dy)

    def draw_network(self):
        pos = nx.get_node_attributes(self.graph, 'pos')
        plt.figure(figsize=(8, 8))
        sensor_nodes = [node for node, data in self.graph.nodes(data=True) if data.get('type') == 'sensor']
        base_nodes = [node for node, data in self.graph.nodes(data=True) if data.get('type') == 'base']
        nx.draw_networkx_nodes(self.graph, pos, nodelist=sensor_nodes, node_color='blue', node_size=10,
                               label='Sensor Nodes')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=base_nodes, node_color='red', node_size=15,
                               label='Base Station')
        nx.draw_networkx_edges(self.graph, pos, alpha=0.15)
        nx.draw_networkx_edges(self.spanning_tree, pos, edge_color='green', width=1, arrows=False,
                               label='Spanning Tree')
        plt.legend()
        plt.axis('equal')
        # plt.savefig('network.pdf', dpi=300)
        plt.show()


if __name__ == '__main__':
    # Example parameters for testing.
    n = 450
    w = 100.0
    d = 5
    comm_radius = 8
    seed = 42
    adversarial_fraction = 0.1
    adversarial_std_dev = 50.0

    network = SensorNetwork(n, w, d, comm_radius, seed=seed,
                            adversarial_fraction=adversarial_fraction,
                            adversarial_std_dev=adversarial_std_dev,
                            honest_distribution="correlated_gaussian")
    print("Generated sensor network with nodes:")
    for node in network.nodes:
        print(node)
    print("\nBase station:", network.base_station)
    network.draw_network()