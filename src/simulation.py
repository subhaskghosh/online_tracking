#!/usr/bin/env python3
"""
simulation.py

Simulate a distributed robust monitoring system using the decentralized
β‐trimmed mean aggregation (via Q‐Digest sketches) and compare it with a
central baseline aggregator that uses a list‐of‐histograms (SimpleListAggregator).

The simulation supports:
  1. Varying number of nodes: [50, 100, 150, 200, ...]
  2. Changing the dimension of the vector at each node: [10, 20, 30, 40, ...]
  3. Changing the value of β in [0.1, 0.2, 0.3, 0.4]
  4. Changing the value of ε in [0.01, 0.05, 0.1, 0.15]
  5. Changing the adversarial standard deviation in [1.0, 10.0, 100.0, 200.0]
  6. Comparing the decentralized (periodic and event‐driven) algorithms with a
     centralized baseline aggregator (SimpleListAggregator) that uses the spanning tree.
  7. Measuring the monitoring function value, absolute error, communication overhead,
     storage, power consumption, and estimated lifetime.
  8. Simulating under different sensor distributions ("correlated_gaussian" or "uniform")
  9. Saving all results in a CSV file with checkpointing for restart
 10. Each simulation defines a threshold for the monitoring function and checks if f(v) exceeds it
 11. Each simulation runs for a (small) number of iterations (here set to 10) to simulate streaming data.
"""

import os
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
rc('text', usetex=True)
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-ticks")

# Define the DISTANCE_SCALE constant
DISTANCE_SCALE = 100  # or another value appropriate for your simulation

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


#########################################
# QDigest Implementation
#########################################

class QDigest:
    """
    An implementation of a Q-Digest sketch for integer values in [0, R-1],
    with compression determined by a parameter k.

    The Q-Digest is maintained as a dictionary mapping buckets to a tuple (count, total),
    where a bucket is identified by a key (L, s) representing the interval [L, L+s-1].

    The Q-Digest is designed so that if N is the total count of inserted values,
    then a bucket v is stored only if:

      (1) count(v) ≤ ⌊N/k⌋, and

      (2) count(v) + count(v_p) + count(v_s) > ⌊N/k⌋,

    where v_p is the parent bucket (the bucket corresponding to the union of two sibling intervals)
    and v_s is the sibling of v.

    When the condition (2) fails (i.e. the bucket’s count together with its sibling and current parent
    do not exceed the threshold), the buckets are merged (i.e. their counts are pushed up into the parent)
    and not stored separately.
    """

    def __init__(self, R, k):
        """
        Parameters:
          R: The domain size, so values are in [0, R-1].
          k: The compression parameter.
        """
        self.R = R
        self.k = k
        # buckets is a dictionary mapping a tuple (L, s) -> (count, total)
        self.buckets = {}

    def insert(self, value, count=1):
        """
        Insert an integer value (with optional multiplicity) into the digest.
        Initially, each insertion creates a leaf bucket of size 1.
        """
        if value < 0 or value >= self.R:
            raise ValueError(f"Value {value} out of domain [0, {self.R - 1}]")
        key = (value, 1)  # leaf bucket covering [value, value]
        if key in self.buckets:
            cnt, tot = self.buckets[key]
            self.buckets[key] = (cnt + count, tot + value * count)
        else:
            self.buckets[key] = (count, value * count)
        # Compress frequently to maintain the Q-Digest invariant.
        self.compress()

    def compress(self):
        """
        Bottom-up compression of buckets.
        Let N = total count; let threshold = floor(N/k).
        For each bucket with interval size s (starting from s=1 upward), if the bucket and its sibling
        together have a combined count ≤ threshold, merge them into the parent bucket.
        """
        N = self.total_count()
        if N == 0:
            return
        threshold = math.floor(N / self.k)
        # Process buckets from smallest intervals upward.
        keys_sorted = sorted(self.buckets.keys(), key=lambda x: (x[1], x[0]))
        for key in keys_sorted:
            if key not in self.buckets:
                continue
            s = key[1]
            L = key[0]
            if s >= self.R:
                continue

            parent_size = s * 2
            parent_low = L - (L % parent_size)
            parent_key = (parent_low, parent_size)

            # Determine sibling key.
            if L % parent_size == 0:
                sibling_key = (L + s, s)
            else:
                sibling_key = (L - s, s)

            current_count, current_total = self.buckets[key]
            sibling_count = 0
            sibling_total = 0
            if sibling_key in self.buckets:
                sibling_count, sibling_total = self.buckets[sibling_key]

            combined = current_count + sibling_count
            if combined <= threshold:
                # Remove the current bucket and its sibling.
                del self.buckets[key]
                if sibling_key in self.buckets:
                    del self.buckets[sibling_key]
                # Add the combined counts to the parent's bucket.
                if parent_key in self.buckets:
                    p_count, p_total = self.buckets[parent_key]
                    self.buckets[parent_key] = (p_count + combined, p_total + current_total + sibling_total)
                else:
                    self.buckets[parent_key] = (combined, current_total + sibling_total)

    def merge(self, other):
        """
        Merge another QDigest (with the same R and k) into this one.
        The merge is performed by summing counts for matching buckets and then compressing.
        """
        if self.R != other.R or self.k != other.k:
            raise ValueError("Cannot merge QDigests with different R or k values.")
        new_q = QDigest(self.R, self.k)
        for key, (cnt, tot) in self.buckets.items():
            new_q.buckets[key] = (cnt, tot)
        for key, (cnt, tot) in other.buckets.items():
            if key in new_q.buckets:
                cnt0, tot0 = new_q.buckets[key]
                new_q.buckets[key] = (cnt0 + cnt, tot0 + tot)
            else:
                new_q.buckets[key] = (cnt, tot)
        new_q.compress()
        return new_q

    def total_count(self):
        """Return the total number of inserted items."""
        return sum(cnt for cnt, tot in self.buckets.values())

    def approximate_rank(self, x):
        """
        Approximate the rank of value x.
        The rank is computed as the total counts of all buckets that lie entirely below x,
        plus a fractional contribution from any bucket that straddles x.
        """
        rank = 0.0
        for (L, s), (cnt, tot) in self.buckets.items():
            high = L + s - 1
            if high < x:
                rank += cnt
            elif L < x <= high:
                fraction = (x - L) / s
                rank += cnt * fraction
        return rank

    def approximate_range(self, L_val, U_val):
        """
        Approximate the total count and total sum for values in the interval [L_val, U_val].
        For buckets that partially overlap [L_val, U_val], a proportional fraction is taken.
        """
        total_c = 0.0
        total_s = 0.0
        for (L, s), (cnt, tot) in self.buckets.items():
            high = L + s - 1
            if L >= L_val and high <= U_val:
                total_c += cnt
                total_s += tot
            elif L < U_val and high > L_val:
                overlap_low = max(L, L_val)
                overlap_high = min(high, U_val)
                if overlap_high >= overlap_low:
                    fraction = (overlap_high - overlap_low + 1) / s
                    total_c += cnt * fraction
                    total_s += tot * fraction
        return total_c, total_s

#########################################
# Decentralized Aggregator Implementation
#########################################

class TrimmedMeanAggregator:
    """
    Implements decentralized β-trimmed mean aggregation using Q-Digest sketches.
    """

    def __init__(self, T, R, beta, epsilon, k=None):
        """
        Parameters:
          T: a networkx DiGraph representing the spanning tree (rooted at the base station).
          R: integer domain size for each coordinate.
          beta: trimming parameter in [0, 0.5).
          epsilon: relative error for the q-digest sketches.
          k: compression parameter for Q-Digest; if None, defaults to max(1, int(1/epsilon)).
        """
        self.T = T
        self.R = R
        self.beta = beta
        self.epsilon = epsilon
        self.k = k if k is not None else max(1, int(1.0 / epsilon))

    def build_leaf_digest(self, value_vector):
        d = len(value_vector)
        digests = []
        for k in range(d):
            qd = QDigest(self.R, self.k)
            qd.insert(value_vector[k])
            digests.append(qd)
        return digests

    def merge_digests(self, digests_list):
        if not digests_list:
            return None
        d = len(digests_list[0])
        # Initialize merged buckets with empty Q-Digests.
        merged = [QDigest(self.R, self.k) for _ in range(d)]
        # Start with the first digest list.
        for k in range(d):
            merged[k] = digests_list[0][k]
        # Merge subsequent digest lists into merged.
        for digests in digests_list[1:]:
            for k in range(d):
                merged[k] = merged[k].merge(digests[k])
        return merged

    def aggregate_up(self, node, local_data):
        if self.T.out_degree(node) == 0:
            # If node is a leaf, return its own digest.
            return self.build_leaf_digest(local_data[node])
        child_digests = []
        for child in self.T.successors(node):
            child_digest = self.aggregate_up(child, local_data)
            child_digests.append(child_digest)
        merged_children = self.merge_digests(child_digests)
        # For the current node, if local data is provided, use it; otherwise, use zeros.
        if node in local_data:
            local_digest = self.build_leaf_digest(local_data[node])
        else:
            d = len(merged_children)
            local_digest = self.build_leaf_digest([0] * d)
        d = len(local_digest)
        final_digest = []
        for k in range(d):
            final_digest.append(local_digest[k].merge(merged_children[k]))
        return final_digest

    def compute_trimmed_mean(self, merged_digests):
        d = len(merged_digests)
        trimmed_mean = np.zeros(d)
        for k in range(d):
            Q = merged_digests[k]
            M = Q.total_count()
            if M == 0:
                trimmed_mean[k] = 0
                continue
            lower_rank = self.beta * M
            upper_rank = (1 - self.beta) * M
            L_k = 0
            for x in range(self.R):
                if Q.approximate_rank(x) >= lower_rank:
                    L_k = x
                    break
            U_k = self.R - 1
            for x in range(self.R):
                if Q.approximate_rank(x) >= upper_rank:
                    U_k = x
                    break
            count_range, sum_range = Q.approximate_range(L_k, U_k)
            if count_range > 0:
                trimmed_mean[k] = sum_range / count_range
            else:
                trimmed_mean[k] = (L_k + U_k) / 2.0
        return trimmed_mean

    def compute_global_aggregator(self, local_data):
        # Identify the root: the node with in_degree == 0 in the spanning tree.
        root_candidates = [node for node in self.T.nodes() if self.T.in_degree(node) == 0]
        if not root_candidates:
            raise ValueError("Spanning tree has no root.")
        root = root_candidates[0]
        merged_digests = self.aggregate_up(root, local_data)
        return self.compute_trimmed_mean(merged_digests)

class QDigestEDTM:
    """
    An implementation of a Q-Digest sketch for integer values in [0, R-1],
    with compression determined by a parameter k.

    The Q-Digest is maintained as a dictionary mapping buckets to a tuple (count, total),
    where a bucket is identified by a key (L, s) representing the interval [L, L+s-1].

    The Q-Digest is designed so that if N is the total count of inserted values,
    then a bucket v is stored only if:

      (1) count(v) ≤ ⌊N/k⌋, and

      (2) count(v) + count(v_p) + count(v_s) > ⌊N/k⌋,

    where v_p is the parent bucket (the bucket corresponding to the union of two sibling intervals)
    and v_s is the sibling of v.

    When the condition (2) fails (i.e. the bucket’s count together with its “sibling” and current parent
    do not exceed the threshold), the buckets are merged (i.e. their counts are pushed up into the parent)
    and not stored separately.
    """

    def __init__(self, R, k):
        """
        Parameters:
          R: The domain size, so values are in [0, R-1].
          k: The compression parameter.
        """
        self.R = R
        self.k = k
        # buckets is a dictionary mapping a tuple (L, s) -> (count, total)
        self.buckets = {}

    def insert(self, value, count=1):
        """
        Insert an integer value (with optional multiplicity) into the digest.
        Initially, each insertion creates a leaf bucket of size 1.
        """
        if value < 0 or value >= self.R:
            raise ValueError(f"Value {value} out of domain [0, {self.R - 1}]")
        key = (value, 1)  # leaf bucket covering [value, value]
        if key in self.buckets:
            cnt, tot = self.buckets[key]
            self.buckets[key] = (cnt + count, tot + value * count)
        else:
            self.buckets[key] = (count, value * count)
        # Compress frequently to maintain the Q-Digest invariant.
        self.compress()

    def compress(self):
        """
        Bottom-up compression of buckets.
        Let N = total count; let threshold = floor(N/k).
        For each bucket with interval size s (starting from s=1 upward), if the bucket and its sibling
        together have a combined count ≤ threshold, merge them into the parent bucket.
        """
        N = self.total_count()
        if N == 0:
            return
        threshold = math.floor(N / self.k)
        # Process buckets from smallest intervals upward.
        keys_sorted = sorted(self.buckets.keys(), key=lambda x: (x[1], x[0]))
        for key in keys_sorted:
            if key not in self.buckets:
                continue
            s = key[1]
            L = key[0]
            if s >= self.R:
                continue

            parent_size = s * 2
            parent_low = L - (L % parent_size)
            parent_key = (parent_low, parent_size)

            # Determine sibling key.
            if L % parent_size == 0:
                sibling_key = (L + s, s)
            else:
                sibling_key = (L - s, s)

            current_count, current_total = self.buckets[key]
            sibling_count = 0
            sibling_total = 0
            if sibling_key in self.buckets:
                sibling_count, sibling_total = self.buckets[sibling_key]

            combined = current_count + sibling_count
            if combined <= threshold:
                # Merge: remove current bucket and its sibling, and add to parent's bucket.
                del self.buckets[key]
                if sibling_key in self.buckets:
                    del self.buckets[sibling_key]
                if parent_key in self.buckets:
                    p_count, p_total = self.buckets[parent_key]
                    self.buckets[parent_key] = (p_count + combined, p_total + current_total + sibling_total)
                else:
                    self.buckets[parent_key] = (combined, current_total + sibling_total)
        # End for

    def merge(self, other):
        """
        Merge another QDigest (with the same R and k) into this one.
        The merge is performed by summing counts for matching buckets and then compressing.
        """
        if self.R != other.R or self.k != other.k:
            raise ValueError("Cannot merge QDigests with different R or k values.")
        new_q = QDigest(self.R, self.k)
        for key, (cnt, tot) in self.buckets.items():
            new_q.buckets[key] = (cnt, tot)
        for key, (cnt, tot) in other.buckets.items():
            if key in new_q.buckets:
                cnt0, tot0 = new_q.buckets[key]
                new_q.buckets[key] = (cnt0 + cnt, tot0 + tot)
            else:
                new_q.buckets[key] = (cnt, tot)
        new_q.compress()
        return new_q

    def total_count(self):
        """Return the total number of inserted items."""
        return sum(cnt for cnt, tot in self.buckets.values())

    def approximate_rank(self, x):
        """
        Approximate the rank of value x.
        The rank is computed as the total counts of all buckets that lie entirely below x,
        plus a fractional contribution from any bucket that straddles x.
        """
        rank = 0.0
        for (L, s), (cnt, tot) in self.buckets.items():
            high = L + s - 1
            if high < x:
                rank += cnt
            elif L < x <= high:
                fraction = (x - L) / s
                rank += cnt * fraction
        return rank

    def approximate_range(self, L_val, U_val):
        """
        Approximate the total count and total sum for values in the interval [L_val, U_val].
        For buckets that partially overlap [L_val, U_val], a proportional fraction is taken.
        """
        total_c = 0.0
        total_s = 0.0
        for (L, s), (cnt, tot) in self.buckets.items():
            high = L + s - 1
            if L >= L_val and high <= U_val:
                total_c += cnt
                total_s += tot
            elif L < U_val and high > L_val:
                overlap_low = max(L, L_val)
                overlap_high = min(high, U_val)
                if overlap_high >= overlap_low:
                    fraction = (overlap_high - overlap_low + 1) / s
                    total_c += cnt * fraction
                    total_s += tot * fraction
        return total_c, total_s


#########################################
# Event-Driven Aggregator Implementation
#########################################

class EventDrivenTrimmedMeanAggregator:
    """
    Implements an event-driven robust aggregator based on the coordinate-wise β-trimmed mean
    using Q-Digest sketches. In this design, each node's local data is summarized into a set of Q-Digest
    sketches (one per coordinate) that are merged along a spanning tree. Additionally, each node maintains
    safe intervals for its local data. An update is triggered only if any node's local data falls outside its safe interval.
    """
    def __init__(self, spanning_tree, R, beta, epsilon, delta_frac=0.12, k=None):
        """
        Parameters:
          spanning_tree : a networkx DiGraph representing the spanning tree (rooted at the base station)
          R             : integer domain size (data values in [0, R-1])
          beta          : trimming parameter (in [0, 0.5))
          epsilon       : approximation parameter for the quantile sketches
          delta_frac    : fraction used to set safe intervals (default 0.12)
          k             : compression parameter for Q-Digest; if None, a default value is set based on epsilon.
        """
        self.tree = spanning_tree
        self.R = R
        self.beta = beta
        self.epsilon = epsilon
        self.delta_frac = delta_frac
        self.k = k if k is not None else max(1, int(1.0 / epsilon))
        self.last_agg = None
        self.safe_intervals = {}  # Mapping: node id -> list of (L, U) intervals per coordinate.
        self.initialized = False
        self.triggered = []

    def build_leaf_digest(self, value_vector):
        """Create a list of Q-Digest sketches (one per coordinate) for a leaf node."""
        d = len(value_vector)
        digests = []
        for i in range(d):
            qd = QDigest(self.R, self.k)
            qd.insert(value_vector[i])
            digests.append(qd)
        return digests

    def merge_digests(self, digests_list):
        """Merge a list of digest lists (one per node) coordinate-wise."""
        if not digests_list:
            return None
        d = len(digests_list[0])
        merged = [digests_list[0][k] for k in range(d)]
        for digests in digests_list[1:]:
            for k in range(d):
                merged[k] = merged[k].merge(digests[k])
        return merged

    def aggregate_up(self, node, local_data):
        """
        Recursively merge Q-Digest sketches from the subtree rooted at 'node'.
        """
        if self.tree.out_degree(node) == 0:
            return self.build_leaf_digest(local_data[node])
        child_digests = []
        for child in self.tree.successors(node):
            child_digest = self.aggregate_up(child, local_data)
            child_digests.append(child_digest)
        merged_children = self.merge_digests(child_digests)
        if node in local_data:
            local_digest = self.build_leaf_digest(local_data[node])
        else:
            d = len(merged_children)
            local_digest = self.build_leaf_digest([0] * d)
        d = len(local_digest)
        final_digest = []
        for k in range(d):
            final_digest.append(local_digest[k].merge(merged_children[k]))
        return final_digest

    def compute_trimmed_mean(self, merged_digests):
        """
        Compute the coordinate-wise β-trimmed mean from the merged Q-Digest sketches.
        """
        d = len(merged_digests)
        trimmed_mean = np.zeros(d)
        for k in range(d):
            Q = merged_digests[k]
            M = Q.total_count()
            if M == 0:
                trimmed_mean[k] = 0
                continue
            lower_rank = self.beta * M
            upper_rank = (1 - self.beta) * M
            L_k = 0
            for x in range(self.R):
                if Q.approximate_rank(x) >= lower_rank:
                    L_k = x
                    break
            U_k = self.R - 1
            for x in range(self.R):
                if Q.approximate_rank(x) >= upper_rank:
                    U_k = x
                    break
            count_range, sum_range = Q.approximate_range(L_k, U_k)
            if count_range > 0:
                trimmed_mean[k] = sum_range / count_range
            else:
                trimmed_mean[k] = (L_k + U_k) / 2.0
        return trimmed_mean

    def _compute_aggregator_from_sketches(self, local_data):
        """
        Compute the global aggregator using Q-Digest sketches.
        """
        root_candidates = [node for node in self.tree.nodes() if self.tree.in_degree(node) == 0]
        if not root_candidates:
            raise ValueError("Spanning tree has no root.")
        root = root_candidates[0]
        merged_digests = self.aggregate_up(root, local_data)
        return self.compute_trimmed_mean(merged_digests)

    def _initialize_safe_intervals(self, local_data):
        """
        Initialize safe intervals for each node based on current local data.
        For simplicity, set a fixed margin delta around each coordinate.
        """
        delta = self.delta_frac * self.R  # scalar margin
        for node_id, vec in local_data.items():
            intervals = []
            for x in vec:
                # Here we assume x is a scalar number.
                L = max(0, x - delta)
                U = min(self.R - 1, x + delta)
                intervals.append((L, U))
            self.safe_intervals[node_id] = intervals

    def get_triggered_nodes(self):
        return self.triggered

    def _get_triggered_nodes(self, local_data):
        """
        Return a list of node IDs which has been triggered in the last round.
        """
        triggered = []
        for node_id, vec in local_data.items():
            if node_id not in self.safe_intervals:
                triggered.append(node_id)
            else:
                intervals = self.safe_intervals[node_id]
                for i, x in enumerate(vec):
                    L, U = intervals[i]
                    if x < L or x > U:
                        triggered.append(node_id)
                        break
        return triggered

    def check_safe_intervals(self, local_data):
        """
        Return True if no node's local data falls outside its safe interval.
        """
        return len(self._get_triggered_nodes(local_data)) == 0

    def update_safe_intervals(self, local_data):
        """
        Update safe intervals based on the current local data.
        For simplicity, we reinitialize the safe intervals.
        """
        self._initialize_safe_intervals(local_data)

    def initialize(self, local_data):
        """
        Perform an initial global aggregation and initialize safe intervals.
        """
        global_agg = self._compute_aggregator_from_sketches(local_data)
        self.last_agg = global_agg
        self.triggered = self._get_triggered_nodes(local_data)
        self._initialize_safe_intervals(local_data)
        self.initialized = True
        return global_agg

    def compute_global_aggregator(self, local_data):
        """
        Compute the global aggregator in an event-driven manner.
        If all nodes' local data remain within their safe intervals, return the last computed aggregator.
        Otherwise, trigger an update by recomputing the aggregator and update safe intervals.
        """
        if not self.initialized:
            return self.initialize(local_data)
        self.triggered = self._get_triggered_nodes(local_data)
        if len(self.triggered) == 0:
            return self.last_agg
        else:
            new_agg = self._compute_aggregator_from_sketches(local_data)
            self.last_agg = new_agg
            self.update_safe_intervals(local_data)
            return new_agg


class SimpleListAggregator:
    """
    Implements a centralized aggregation algorithm using exact histograms.

    For each coordinate, each sensor node maintains a histogram (dictionary) with a bucket width of 1.
    The histogram records all distinct sensor values and their frequencies in the subtree rooted at the node.
    During a bottom-up pass in the spanning tree, these histograms are merged exactly.
    The base station (root) then computes an exact β-trimmed mean for each coordinate.
    """
    def __init__(self, T, R, beta):
        """
        Parameters:
          T: a networkx DiGraph representing the spanning tree (rooted at the base station).
          R: integer domain size for each coordinate (all sensor values are assumed to lie in [0, R-1]).
          beta: trimming parameter in [0, 0.5) (fraction to trim from each tail).
        """
        self.T = T
        self.R = R
        self.beta = beta

    def build_leaf_histogram(self, value_vector):
        """
        Build a histogram for a given sensor node's local data.

        Parameters:
          value_vector: A list (or array) of sensor values (one per coordinate), assumed to be integers in [0, R-1].

        Returns:
          hist_list: A list of histograms (one dictionary per coordinate) where keys are sensor values and values are counts.
        """
        d = len(value_vector)
        hist_list = []
        for k in range(d):
            # For a single sensor, the histogram has one key with count 1.
            hist = {value_vector[k]: 1}
            hist_list.append(hist)
        return hist_list

    def merge_histograms(self, h1, h2):
        """
        Merge two histograms, h1 and h2, by summing counts of common keys.

        Parameters:
          h1, h2: Dictionaries mapping sensor values to their counts.

        Returns:
          merged: A dictionary containing the merged counts.
        """
        merged = h1.copy()
        for value, count in h2.items():
            merged[value] = merged.get(value, 0) + count
        return merged

    def merge_histogram_list(self, histogram_list):
        """
        Merge a list of histograms (for the same coordinate) into a single histogram.

        Parameters:
          histogram_list: List of histogram dictionaries.

        Returns:
          merged: A dictionary with the merged histogram.
        """
        if not histogram_list:
            return {}
        merged = histogram_list[0]
        for h in histogram_list[1:]:
            merged = self.merge_histograms(merged, h)
        return merged

    def aggregate_up(self, node, local_data):
        """
        Recursively aggregates histograms from the subtree rooted at the given node.

        Parameters:
          node: The current node (identifier) in the spanning tree.
          local_data: A dictionary mapping node id to its local value vector.

        Returns:
          final_hist: A list of histograms (one per coordinate) aggregated from the subtree.
        """
        # Build this node's local histogram.
        if node in local_data:
            local_hist = self.build_leaf_histogram(local_data[node])
        else:
            # If no data is available, assume a default vector of zeros.
            d = len(next(iter(local_data.values())))
            local_hist = self.build_leaf_histogram([0] * d)
        # Get the children of this node in the spanning tree.
        children = list(self.T.successors(node))
        if not children:
            # If leaf, simply return its histogram.
            return local_hist

        # Recursively aggregate histograms from each child.
        children_histograms = []
        for child in children:
            child_hist = self.aggregate_up(child, local_data)
            children_histograms.append(child_hist)

        # For each coordinate, merge the local histogram with those from children.
        d = len(local_hist)
        final_hist = []
        for k in range(d):
            # Get histograms for coordinate k from this node and its children.
            h_list = [local_hist[k]] + [child_hist[k] for child_hist in children_histograms]
            merged = self.merge_histogram_list(h_list)
            final_hist.append(merged)
        return final_hist

    def compute_trimmed_mean(self, histogram_list):
        """
        For each coordinate, compute the exact β-trimmed mean using the aggregated histogram.

        Parameters:
          histogram_list: A list of histograms (one per coordinate).

        Returns:
          trimmed_mean: A numpy array where each entry is the trimmed mean for a coordinate.
        """
        d = len(histogram_list)
        trimmed_mean = np.zeros(d)
        for k in range(d):
            hist = histogram_list[k]
            # Sort histogram items (value, count) by the sensor value.
            items = sorted(hist.items(), key=lambda x: x[0])
            total_count = sum(count for _, count in items)
            trim_count = int(self.beta * total_count)

            # Determine lower cutoff index.
            cum_count = 0
            lower_index = 0
            while lower_index < len(items) and cum_count < trim_count:
                cum_count += items[lower_index][1]
                lower_index += 1

            # Determine upper cutoff index.
            cum_count = 0
            upper_index = len(items) - 1
            while upper_index >= 0 and cum_count < trim_count:
                cum_count += items[upper_index][1]
                upper_index -= 1

            # Sum the values and counts of the remaining items.
            remaining_count = 0
            weighted_sum = 0.0
            for i in range(lower_index, upper_index + 1):
                value, count = items[i]
                weighted_sum += value * count
                remaining_count += count

            if remaining_count > 0:
                trimmed_mean[k] = weighted_sum / remaining_count
            else:
                # Fallback: if trimming discards all data, average lower and upper cutoff.
                trimmed_mean[k] = (items[0][0] + items[-1][0]) / 2.0
        return trimmed_mean

    def compute_mean(self, histogram_list):
        """
        Compute the exact mean for each coordinate using the aggregated histogram.

        Parameters:
          histogram_list: A list of histograms (one per coordinate). Each histogram is a dictionary
                          where keys are sensor values and values are the corresponding counts.

        Returns:
          mean_values: A numpy array where each entry is the mean for that coordinate.
        """
        d = len(histogram_list)
        mean_values = np.zeros(d)
        for k in range(d):
            hist = histogram_list[k]
            total_count = sum(hist.values())
            weighted_sum = sum(value * count for value, count in hist.items())
            if total_count > 0:
                mean_values[k] = weighted_sum / total_count
            else:
                mean_values[k] = 0.0  # In case there is no data, default to 0.
        return mean_values

    def compute_global_aggregator(self, local_data):
        """
        Compute the global aggregator via exact histogram merging.

        Parameters:
          local_data: A dictionary mapping node id to its local integer value vector.

        Returns:
          aggregator: A numpy array containing the β-trimmed mean for each coordinate.
        """
        # Identify the spanning tree root (node with in-degree 0).
        roots = [node for node in self.T.nodes() if self.T.in_degree(node) == 0]
        if not roots:
            raise ValueError("Spanning tree has no root.")
        root = roots[0]
        aggregated_histograms = self.aggregate_up(root, local_data)
        aggregator = self.compute_trimmed_mean(aggregated_histograms)
        return aggregator

    def compute_global_aggregator_wo_trimming(self, local_data):
        """
        Compute the global aggregator via exact histogram merging.

        Parameters:
          local_data: A dictionary mapping node id to its local integer value vector.

        Returns:
          aggregator: A numpy array containing the β-trimmed mean for each coordinate.
        """
        # Identify the spanning tree root (node with in-degree 0).
        roots = [node for node in self.T.nodes() if self.T.in_degree(node) == 0]
        if not roots:
            raise ValueError("Spanning tree has no root.")
        root = roots[0]
        aggregated_histograms = self.aggregate_up(root, local_data)
        aggregator = self.compute_mean(aggregated_histograms)
        return aggregator
######################################################
# Monitoring Function and Baseline Aggregator
######################################################

def monitoring_function(v):
    """Monitoring function defined as the average of the aggregated vector."""
    return np.mean(v)


######################################################
# Functions for the SimpleListAggregator baseline.
######################################################

def simplelist_serialized_size(histogram, bits_per_bucket=96):
    """
    Return the serialized size (in bits) of a histogram.
    The histogram is represented as a dict mapping distinct sensor values (buckets) to counts.
    Each bucket is assumed to require 'bits_per_bucket' bits.
    """
    return bits_per_bucket * len(histogram)

def estimate_comm_overhead_from_simple_list(aggregator, local_data, tree, bits_per_bucket=96):
    """
    Estimate the total communication overhead (in bits) for one full upward aggregation
    using the SimpleListAggregator. This function traverses the spanning tree and
    accumulates the serialized sizes of histograms transmitted on each edge.
    Histograms are merged pairwise.
    """
    total_bits = 0

    def traverse(node):
        nonlocal total_bits
        if tree.out_degree(node) == 0:
            leaf_hist = aggregator.build_leaf_histogram(local_data[node])
            msg_size = sum(simplelist_serialized_size(hist, bits_per_bucket) for hist in leaf_hist)
            total_bits += msg_size
            return leaf_hist
        child_histograms_list = []
        for child in tree.successors(node):
            child_hist = traverse(child)
            child_histograms_list.append(child_hist)
        merged_children = child_histograms_list[0]
        for hist in child_histograms_list[1:]:
            d = len(merged_children)
            new_merged = []
            for k in range(d):
                new_merged.append(aggregator.merge_histograms(merged_children[k], hist[k]))
            merged_children = new_merged
        if node in local_data:
            local_hist = aggregator.build_leaf_histogram(local_data[node])
        else:
            d = len(merged_children)
            local_hist = aggregator.build_leaf_histogram([0] * d)
        d = len(local_hist)
        final_hist = []
        msg_size = 0
        for k in range(d):
            merged_hist = aggregator.merge_histograms(local_hist[k], merged_children[k])
            final_hist.append(merged_hist)
            msg_size += simplelist_serialized_size(merged_hist, bits_per_bucket)
        total_bits += msg_size
        return final_hist

    root_candidates = [node for node in tree.nodes() if tree.in_degree(node) == 0]
    if not root_candidates:
        raise ValueError("Spanning tree has no root.")
    root = root_candidates[0]
    traverse(root)
    return total_bits

def estimate_storage_from_simple_list(aggregator, local_data, bits_per_bucket=96):
    """
    Estimate the per-node storage requirement (in bits) for the SimpleListAggregator,
    based on the serialized size of the histograms produced at each node.
    """
    sizes = []
    for node_id, vec in local_data.items():
        leaf_hist = aggregator.build_leaf_histogram(vec)
        size = sum(simplelist_serialized_size(hist, bits_per_bucket) for hist in leaf_hist)
        sizes.append(size)
    return np.mean(sizes)

def estimate_power_and_lifetime_from_simple_list(T, aggregator, local_data, n, d, constant=1.0,
                                                 battery_capacity=10.0, E_elec=50e-9, epsilon_amp=100e-12,
                                                 gamma=2, bits_per_bucket=96):
    """
    Estimate energy consumption and lifetime for one update event using the SimpleListAggregator.
    The message size (in bits) is computed from the serialized histogram sizes.
    """
    message_size = estimate_comm_overhead_from_simple_list(aggregator, local_data, T, bits_per_bucket)
    total_energy = compute_transmission_energy(T, message_size, E_elec, epsilon_amp, gamma)
    avg_energy = total_energy / n
    lifetime_rounds = battery_capacity / avg_energy if avg_energy > 0 else float('inf')
    return total_energy, avg_energy, lifetime_rounds

######################################################
# Functions for Q-Digest based aggregators.
######################################################

def qdigest_serialized_size(qdigest, bits_per_bucket=96):
    """
    Return the serialized size (in bits) of a QDigest sketch.
    """
    return bits_per_bucket * len(qdigest.buckets)

def estimate_comm_overhead_from_digests(aggregator, local_data, tree, bits_per_bucket=96):
    """
    Estimate the total communication overhead (in bits) for one full upward aggregation,
    based on the serialized sizes of the QDigest sketches transmitted on each edge.
    For periodic aggregators, the entire tree is traversed.
    """
    total_bits = 0

    def traverse(node):
        nonlocal total_bits
        if tree.out_degree(node) == 0:
            leaf_digest = aggregator.build_leaf_digest(local_data[node])
            msg_size = sum(qdigest_serialized_size(qd, bits_per_bucket) for qd in leaf_digest)
            total_bits += msg_size
            return leaf_digest
        child_digests = []
        for child in tree.successors(node):
            child_digest = traverse(child)
            child_digests.append(child_digest)
        merged_children = aggregator.merge_digests(child_digests)
        if node in local_data:
            local_digest = aggregator.build_leaf_digest(local_data[node])
        else:
            d = len(merged_children)
            local_digest = aggregator.build_leaf_digest([0] * d)
        d = len(local_digest)
        final_digest = []
        msg_size = 0
        for k in range(d):
            merged_qd = local_digest[k].merge(merged_children[k])
            final_digest.append(merged_qd)
            msg_size += qdigest_serialized_size(merged_qd, bits_per_bucket)
        total_bits += msg_size
        return final_digest

    root_candidates = [node for node in tree.nodes() if tree.in_degree(node) == 0]
    if not root_candidates:
        raise ValueError("Spanning tree has no root.")
    root = root_candidates[0]
    traverse(root)
    return total_bits

def estimate_comm_overhead_from_digests_event(aggregator, local_data, tree, bits_per_bucket=96):
    """
    Estimate the communication overhead (in bits) for the event-driven aggregator.
    Only count message sizes along the upward paths from nodes that have triggered an update.
    If no node has triggered, return 0.
    """
    # We assume the aggregator has a method get_triggered_nodes(local_data)
    # that returns a list of node IDs where the safe interval was violated.
    if not hasattr(aggregator, "get_triggered_nodes"):
        # Fall back to computing overhead over all nodes.
        return estimate_comm_overhead_from_digests(aggregator, local_data, tree, bits_per_bucket)

    triggered = aggregator.get_triggered_nodes()
    if not triggered:
        return 0

    # In a spanning tree, each triggered node sends an updated message on the unique path from that node
    # to the root. We collect all edges on these paths (avoiding duplicate counting).
    triggered_edges = set()
    root_candidates = [node for node in tree.nodes() if tree.in_degree(node) == 0]
    if not root_candidates:
        raise ValueError("Spanning tree has no root.")
    root = root_candidates[0]
    for node in triggered:
        current = node
        while current != root:
            parent = list(tree.predecessors(current))[0]  # Unique predecessor in the spanning tree.
            triggered_edges.add((parent, current))
            current = parent

    total_bits = 0
    # For each edge (u, v) in the union, compute the message size transmitted from v.
    # (Assume that if v is triggered, its updated digest is transmitted.)
    for (u, v) in triggered_edges:
        if v in local_data:
            digest = aggregator.build_leaf_digest(local_data[v])
        else:
            # If local_data is not available, assume a default zero digest.
            d = len(next(iter(aggregator.build_leaf_digest([0]*1))) )  # infer dimension; adjust as needed.
            digest = aggregator.build_leaf_digest([0] * d)
        msg_size = sum(qdigest_serialized_size(qd, bits_per_bucket) for qd in digest)
        total_bits += msg_size
    return total_bits

def estimate_storage_from_digests(aggregator, local_data, bits_per_bucket=96):
    """
    Estimate the storage requirement (in bits) per node based on the serialized size of the QDigest sketches.
    """
    sizes = []
    for node_id, vec in local_data.items():
        leaf_digest = aggregator.build_leaf_digest(vec)
        size = sum(qdigest_serialized_size(qd, bits_per_bucket) for qd in leaf_digest)
        sizes.append(size)
    return np.mean(sizes)

def compute_transmission_energy(T, message_size, E_elec=50e-9, epsilon_amp=100e-12, gamma=2):
    """
    Compute the total transmission energy (in Joules) for one upward pass along the spanning tree T.
    For each edge (u,v) in T, the energy for a message of size 'message_size' bits transmitted over
    a distance d (in meters) is:
      E_tx = E_elec * message_size + epsilon_amp * message_size * (d^gamma).
    DISTANCE_SCALE scales the distances.
    """
    total_energy = 0.0
    for u, v in T.edges():
        pos_u = np.array(T.nodes[u]['pos'])
        pos_v = np.array(T.nodes[v]['pos'])
        distance = np.linalg.norm(pos_u - pos_v) * DISTANCE_SCALE
        E_tx = E_elec * message_size + epsilon_amp * message_size * (distance ** gamma)
        total_energy += E_tx
    return total_energy

def estimate_power_and_lifetime_from_digests(T, aggregator, local_data, n, d, epsilon,
                                             constant=1.0, battery_capacity=10.0,
                                             E_elec=50e-9, epsilon_amp=100e-12, gamma=2,
                                             bits_per_bucket=96):
    """
    Estimate energy consumption and lifetime for one update event using Q-Digest sketches.
    The message size is computed from the serialized sizes of the QDigest sketches transmitted in the spanning tree.
    """
    # For event-driven aggregation, use the event-specific overhead function.
    if hasattr(aggregator, "get_triggered_nodes"):
        message_size = estimate_comm_overhead_from_digests_event(aggregator, local_data, T, bits_per_bucket)
    else:
        message_size = estimate_comm_overhead_from_digests(aggregator, local_data, T, bits_per_bucket)
    total_energy = compute_transmission_energy(T, message_size, E_elec, epsilon_amp, gamma)
    avg_energy_per_node = total_energy / n
    lifetime_rounds = battery_capacity / avg_energy_per_node if avg_energy_per_node > 0 else float('inf')
    return total_energy, avg_energy_per_node, lifetime_rounds

######################################################
# Simulation Parameters and Main Loop
######################################################

# Simulation parameter lists.
nodes_list = [50, 100, 200, 400, 600]
dimensions = [10, 20, 40, 60, 80, 100]
beta_values = [0.1, 0.2, 0.3, 0.4]
epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5]
adversarial_std_devs = [1.0, 10.0, 100.0, 200.0]
honest_distributions = ["correlated_gaussian", "uniform"]
delta_frac = [0.01, 0.1, 0.2, 0.3, 0.4]
R_values = [100, 500, 1000, 5000, 10000]            # Domain for aggregators (sensor values in [0, R-1]).

# Other simulation parameters.
w = 100.0           # Deployment area: 100 x 100 square.
comm_radius = [30.0, 25.0, 10.0, 8.0, 8.0]  # Communication radii.
time_iterations = 100  # Number of simulation iterations.
seed = 81278181      # Random seed for reproducibility.

tau = 50.0           # Monitoring threshold.

# Checkpoint file for results.
results_filename = "simulation_results.csv"
if os.path.exists(results_filename):
    df_results = pd.read_csv(results_filename)
else:
    df_results = pd.DataFrame(columns=[
        "R", "nodes", "dimension", "beta", "epsilon", "delta_frac", "adv_std_dev",
        "honest_distribution", "iteration", "f_approx", "f_event", "f_central",
        "abs_error_approx", "abs_error_event", "comm_overhead_approx",
        "comm_overhead_event", "comm_overhead_central", "storage_approx",
        "storage_event", "storage_central", "power_tx_approx",
        "power_tx_event", "power_tx_central", "avg_energy_per_node_approx",
        "avg_energy_per_node_event", "avg_energy_per_node_central", "lifetime_rounds_approx",
        "lifetime_rounds_event", "lifetime_rounds_central", "threshold",
        "f_approx_exceeds_threshold", "f_event_exceeds_threshold",
        "f_central_untrimmed_mean", "abs_error_untrimmed"
    ])

random.seed(seed)
np.random.seed(seed)

for R in R_values:
    for n, cr in zip(nodes_list, comm_radius):
        for d in dimensions:
            for beta in beta_values:
                for epsilon in epsilon_values:
                    for adv_std in adversarial_std_devs:
                        for honest_dist in honest_distributions:
                            for d_frac in delta_frac:
                                config_str = f"n={n}, d={d}, beta={beta}, epsilon={epsilon}, delta_frac={d_frac}, adv_std={adv_std}, honest_dist={honest_dist}"
                                print("Running simulation for config:", config_str)

                                # Create sensor network with current configuration.
                                adversarial_fraction = beta  # (Ensure adversarial_fraction <= beta as required.)
                                network = SensorNetwork(n, w, d, cr, seed=seed,
                                                        adversarial_fraction=adversarial_fraction,
                                                        adversarial_std_dev=adv_std,
                                                        honest_distribution=honest_dist)
                                spanning_tree = network.spanning_tree

                                # Instantiate aggregators.
                                aggregator_periodic = TrimmedMeanAggregator(spanning_tree, R, beta, epsilon)
                                aggregator_event = EventDrivenTrimmedMeanAggregator(spanning_tree, R, beta, epsilon, d_frac)
                                aggregator_central = SimpleListAggregator(spanning_tree, R, beta)
                                aggregator_central_untrimmed = SimpleListAggregator(spanning_tree, R, beta)

                                config_results = []
                                for t in range(time_iterations):
                                    raw_data_list = []
                                    local_data = {}
                                    for node in network.nodes:
                                        vec = node.get_data()
                                        # Map sensor values in [0, 100] to integers in [0, R-1].
                                        int_vec = [min(max(int(round(x)), 0), R - 1) for x in vec]
                                        local_data[node.id] = int_vec
                                        raw_data_list.append(int_vec)
                                    # Also include the base station's data.
                                    base_vec = network.base_station.get_data()
                                    int_base_vec = [min(max(int(round(x)), 0), R - 1) for x in base_vec]
                                    local_data[network.base_station.id] = int_base_vec
                                    raw_data_list.append(int_base_vec)

                                    raw_data = np.array(raw_data_list)  # Shape: (n+1, d)

                                    # Compute aggregators.
                                    approx_agg = aggregator_periodic.compute_global_aggregator(local_data)
                                    event_agg = aggregator_event.compute_global_aggregator(local_data)
                                    central_agg = aggregator_central.compute_global_aggregator(local_data)
                                    central_agg_untrimmed = aggregator_central_untrimmed.compute_global_aggregator_wo_trimming(local_data)

                                    f_approx = monitoring_function(approx_agg)
                                    f_event = monitoring_function(event_agg)
                                    f_central = monitoring_function(central_agg)
                                    f_central_untrimmed_mean = monitoring_function(central_agg_untrimmed)
                                    abs_error_approx = abs(f_approx - f_central)
                                    abs_error_event = abs(f_event - f_central)
                                    abs_error_untrimmed = abs(f_central_untrimmed_mean - f_central)

                                    comm_overhead_approx = estimate_comm_overhead_from_digests(aggregator_periodic, local_data, spanning_tree)
                                    comm_overhead_event = estimate_comm_overhead_from_digests_event(aggregator_event, local_data, spanning_tree)
                                    comm_overhead_central = estimate_comm_overhead_from_simple_list(aggregator_central, local_data, spanning_tree)

                                    storage_approx = estimate_storage_from_digests(aggregator_periodic, local_data)
                                    storage_event = estimate_storage_from_digests(aggregator_event, local_data)
                                    storage_central = estimate_storage_from_simple_list(aggregator_central, local_data)

                                    total_energy_approx, avg_energy_approx, lifetime_rounds_approx = estimate_power_and_lifetime_from_digests(
                                        spanning_tree, aggregator_periodic, local_data, n, d, epsilon,
                                        constant=1.0, battery_capacity=10.0, E_elec=50e-9, epsilon_amp=100e-12, gamma=2,
                                        bits_per_bucket=96)
                                    total_energy_event, avg_energy_event, lifetime_rounds_event = estimate_power_and_lifetime_from_digests(
                                        spanning_tree, aggregator_event, local_data, n, d, epsilon,
                                        constant=1.0, battery_capacity=10.0, E_elec=50e-9, epsilon_amp=100e-12, gamma=2,
                                        bits_per_bucket=96)
                                    total_energy_central, avg_energy_central, lifetime_rounds_central = estimate_power_and_lifetime_from_simple_list(
                                        spanning_tree, aggregator_central, local_data, n, d, constant=1.0, battery_capacity=10.0,
                                        E_elec=50e-9, epsilon_amp=100e-12, gamma=2, bits_per_bucket=96)

                                    f_approx_exceeds = f_approx > tau
                                    f_event_exceeds = f_event > tau

                                    config_results.append({
                                        "R": R,
                                        "nodes": n,
                                        "dimension": d,
                                        "beta": beta,
                                        "epsilon": epsilon,
                                        "delta_frac": d_frac,
                                        "adv_std_dev": adv_std,
                                        "honest_distribution": honest_dist,
                                        "iteration": t,
                                        "f_approx": f_approx,
                                        "f_event": f_event,
                                        "f_central": f_central,
                                        "abs_error_approx": abs_error_approx,
                                        "abs_error_event": abs_error_event,
                                        "comm_overhead_approx": comm_overhead_approx,
                                        "comm_overhead_event": comm_overhead_event,
                                        "comm_overhead_central": comm_overhead_central,
                                        "storage_approx": storage_approx,
                                        "storage_event": storage_event,
                                        "storage_central": storage_central,
                                        "power_tx_approx": total_energy_approx,
                                        "power_tx_event": total_energy_event,
                                        "power_tx_central": total_energy_central,
                                        "avg_energy_per_node_approx": avg_energy_approx,
                                        "avg_energy_per_node_event": avg_energy_event,
                                        "avg_energy_per_node_central": avg_energy_central,
                                        "lifetime_rounds_approx": lifetime_rounds_approx,
                                        "lifetime_rounds_event": lifetime_rounds_event,
                                        "lifetime_rounds_central": lifetime_rounds_central,
                                        "threshold": tau,
                                        "f_approx_exceeds_threshold": f_approx_exceeds,
                                        "f_event_exceeds_threshold": f_event_exceeds,
                                        "f_central_untrimmed_mean": f_central_untrimmed_mean,
                                        "abs_error_untrimmed": abs_error_untrimmed
                                    })

                                df_config = pd.DataFrame(config_results)
                                df_results = pd.concat([df_results, df_config], ignore_index=True)
                                df_results.to_csv(results_filename, index=False)
                                print(f"Config {config_str} completed with {time_iterations} iterations.\n")