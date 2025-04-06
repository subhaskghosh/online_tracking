#!/usr/bin/env python3
"""
event_driven_trimmed_mean.py

Implements an event‐driven robust aggregator based on the coordinate‐wise β‐trimmed mean
using Q‐Digest sketches. This aggregator maintains per‐node safe intervals for each coordinate,
and only triggers an update (i.e., recomputes the global aggregator) if any node's local data
falls outside its assigned safe interval.

This file is used by simulation.py to compare the event‐driven aggregator with the periodic
(standard) aggregator.
"""

import math
import numpy as np

#########################################
# QDigest Implementation
#########################################

class QDigest:
    """
    A simple Q-Digest sketch implemented as a histogram for integer values in [0, R-1].
    Each bucket corresponds to an integer value and stores a tuple (count, total), where
    'total' is the sum of the values in that bucket.
    """
    def __init__(self, R):
        self.R = R  # Domain is [0, R-1]
        self.buckets = {}  # Mapping from integer value to (count, total)

    def insert(self, value, count=1):
        if value < 0 or value >= self.R:
            raise ValueError(f"Value {value} out of domain [0, {self.R - 1}]")
        if value in self.buckets:
            cnt, tot = self.buckets[value]
            self.buckets[value] = (cnt + count, tot + value * count)
        else:
            self.buckets[value] = (count, value * count)

    def merge(self, other):
        if self.R != other.R:
            raise ValueError("Cannot merge QDigests with different R values.")
        new_q = QDigest(self.R)
        # Copy buckets from self.
        for key, (cnt, tot) in self.buckets.items():
            new_q.buckets[key] = (cnt, tot)
        # Merge buckets from other.
        for key, (cnt, tot) in other.buckets.items():
            if key in new_q.buckets:
                cnt0, tot0 = new_q.buckets[key]
                new_q.buckets[key] = (cnt0 + cnt, tot0 + tot)
            else:
                new_q.buckets[key] = (cnt, tot)
        return new_q

    def total_count(self):
        return sum(cnt for cnt, _ in self.buckets.values())

    def approximate_rank(self, x):
        rank = 0
        for key, (cnt, _) in self.buckets.items():
            if key < x:
                rank += cnt
            elif key == x:
                rank += cnt / 2.0
        return rank

    def approximate_range(self, L, U):
        total_c = 0
        total_s = 0
        for key, (cnt, tot) in self.buckets.items():
            if L <= key <= U:
                total_c += cnt
                total_s += tot
        return total_c, total_s


#########################################
# Event-Driven Aggregator Implementation
#########################################

class EventDrivenTrimmedMeanAggregator:
    """
    Implements an event-driven robust aggregator based on the coordinate-wise β-trimmed mean
    using Q-Digest sketches. In this design, each node's local data is summarized into a set of Q-Digest
    sketches (one per coordinate) that are merged along a spanning tree. Additionally, each node maintains
    safe intervals for its local data. An update is triggered only if a node's local data falls outside its safe interval.
    """
    def __init__(self, spanning_tree, R, beta, epsilon, delta_frac = 0.12):
        """
        Parameters:
          spanning_tree : a networkx DiGraph representing the spanning tree (rooted at the base station)
          R             : integer domain size (data values in [0, R-1])
          beta          : trimming parameter (in [0, 0.5))
          epsilon       : approximation parameter for the quantile sketches
        """
        self.tree = spanning_tree
        self.R = R
        self.delta_frac = delta_frac
        self.beta = beta
        self.epsilon = epsilon
        self.last_agg = None
        self.safe_intervals = {}  # Mapping: node id -> list of (L, U) intervals per coordinate.
        self.initialized = False

    def build_leaf_digest(self, value_vector):
        """Create a list of Q-Digest sketches (one per coordinate) for a leaf node."""
        d = len(value_vector)
        digests = []
        for k in range(d):
            qd = QDigest(self.R)
            qd.insert(value_vector[k])
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
            # Leaf node: return its own digest.
            return self.build_leaf_digest(local_data[node])
        child_digests = []
        for child in self.tree.successors(node):
            child_digest = self.aggregate_up(child, local_data)
            child_digests.append(child_digest)
        merged_children = self.merge_digests(child_digests)
        # Get local digest for the current node.
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
        # Identify the root of the spanning tree (node with in-degree 0).
        root_candidates = [node for node in self.tree.nodes() if self.tree.in_degree(node) == 0]
        if not root_candidates:
            raise ValueError("Spanning tree has no root.")
        root = root_candidates[0]
        merged_digests = self.aggregate_up(root, local_data)
        return self.compute_trimmed_mean(merged_digests)

    def _initialize_safe_intervals(self, local_data):
        """
        Initialize safe intervals for each node based on current local data.
        For simplicity, we set a fixed margin delta around each coordinate.
        """
        delta = self.delta_frac * self.R  # Example margin (tunable parameter)
        for node_id, vec in local_data.items():
            intervals = []
            for x in vec:
                L = max(0, x - delta)
                U = min(self.R - 1, x + delta)
                intervals.append((L, U))
            self.safe_intervals[node_id] = intervals

    def initialize(self, local_data):
        """
        Perform an initial global aggregation (using the Q-Digest method) and initialize safe intervals.
        """
        global_agg = self._compute_aggregator_from_sketches(local_data)
        self.last_agg = global_agg
        self._initialize_safe_intervals(local_data)
        self.initialized = True
        return global_agg

    def check_safe_intervals(self, local_data):
        """
        Check whether every node's local data is within its assigned safe intervals.
        """
        for node_id, vec in local_data.items():
            if node_id not in self.safe_intervals:
                return False
            intervals = self.safe_intervals[node_id]
            for i, x in enumerate(vec):
                L, U = intervals[i]
                if x < L or x > U:
                    return False
        return True

    def update_safe_intervals(self, local_data):
        """
        Update safe intervals based on the current local data.
        For simplicity, we reinitialize the safe intervals.
        """
        self._initialize_safe_intervals(local_data)

    def compute_global_aggregator(self, local_data):
        """
        Compute the global aggregator in an event-driven manner.
        If all nodes' local data remain within their safe intervals, return the last computed aggregator.
        Otherwise, trigger an update by recomputing the global aggregator using the Q-Digest sketches,
        update the safe intervals, and return the new aggregator.
        """
        if not self.initialized:
            return self.initialize(local_data)
        if self.check_safe_intervals(local_data):
            # No update is triggered.
            return self.last_agg
        else:
            new_agg = self._compute_aggregator_from_sketches(local_data)
            self.last_agg = new_agg
            self.update_safe_intervals(local_data)
            return new_agg


#########################################
# Example Usage (for testing)
#########################################
if __name__ == '__main__':
    from sensor_network import SensorNetwork  # Assuming sensor_network.py is available

    # Simulation parameters.
    n = 50          # Number of sensor nodes.
    w = 100.0       # Deployment area (100x100).
    d = 5           # Dimension of sensor data vectors.
    comm_radius = 30.0  # Communication radius.
    R = 100         # Integer domain size for Q-Digest (data in [0, R-1]).
    beta = 0.1      # Trimming parameter.
    epsilon = 0.05  # Approximation parameter for Q-Digest sketches.
    adversarial_fraction = 0.1  # Fraction of adversarial nodes.
    seed = 42

    # Create a sensor network (spanning tree built within SensorNetwork).
    network = SensorNetwork(n, w, d, comm_radius, seed=seed,
                            adversarial_fraction=adversarial_fraction)
    # Prepare local data: map each node's sensor reading to integers in [0, R-1].
    local_data = {}
    for node in network.nodes:
        vec = node.get_data()
        int_vec = [min(max(int(round(x)), 0), R - 1) for x in vec]
        local_data[node.id] = int_vec

    # Also include the base station's data.
    base_vec = network.base_station.get_data()
    int_base_vec = [min(max(int(round(x)), 0), R - 1) for x in base_vec]
    local_data[network.base_station.id] = int_base_vec

    spanning_tree = network.spanning_tree
    aggregator = EventDrivenTrimmedMeanAggregator(spanning_tree, R, beta, epsilon)
    robust_vector = aggregator.compute_global_aggregator(local_data)
    print("Event-driven Approximate β-Trimmed Mean Aggregator (per coordinate):")
    print(robust_vector)