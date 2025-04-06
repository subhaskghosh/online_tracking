#!/usr/bin/env python3
"""
approximate_trimmed_mean.py

This module implements a decentralized β‐trimmed mean aggregation algorithm
using Q‐Digest sketches. The network topology (a spanning tree) is assumed to be
generated in sensor_network.py.
"""

import math
import networkx as nx
import numpy as np
from sensor_network import SensorNetwork  # Assuming sensor_network.py is in the same directory


class QDigest:
    """
    A simple Q-Digest sketch implemented as a histogram for integer values in [0, R-1].
    Each bucket corresponds to an integer value and stores a tuple (count, total), where
    'total' is the sum of the values in that bucket.
    """

    def __init__(self, R):
        self.R = R  # Domain is [0, R-1]
        self.buckets = {}  # Dictionary mapping integer value to (count, total)

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
        # Copy self buckets
        for key, (cnt, tot) in self.buckets.items():
            new_q.buckets[key] = (cnt, tot)
        # Merge buckets from other
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


class TrimmedMeanAggregator:
    """
    Implements decentralized β-trimmed mean aggregation using Q-Digest sketches.
    """

    def __init__(self, T, R, beta, epsilon):
        """
        Parameters:
          T: a networkx DiGraph representing the spanning tree (rooted at the base station).
          R: integer domain size for each coordinate.
          beta: trimming parameter in [0, 0.5).
          epsilon: relative error for the q-digest sketches.
        """
        self.T = T
        self.R = R
        self.beta = beta
        self.epsilon = epsilon

    def build_leaf_digest(self, value_vector):
        d = len(value_vector)
        digests = []
        for k in range(d):
            qd = QDigest(self.R)
            qd.insert(value_vector[k])
            digests.append(qd)
        return digests

    def merge_digests(self, digests_list):
        if not digests_list:
            return None
        d = len(digests_list[0])
        merged = [QDigest(self.R) for _ in range(d)]
        for k in range(d):
            merged[k] = digests_list[0][k]
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
        # For the current node, if local data is not provided, use a default vector (e.g., zeros).
        if node in local_data:
            local_digest = self.build_leaf_digest(local_data[node])
        else:
            d = len(next(iter(merged_children)))
            local_digest = [[0] * d for _ in range(d)]  # This case should not occur.
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


if __name__ == '__main__':
    # Example usage:
    from sensor_network import SensorNetwork  # assuming sensor_network.py exists

    n = 50  # number of sensor nodes
    w = 100.0  # deployment area is 100 x 100
    d = 5  # dimension of data vectors
    comm_radius = 30.0  # communication radius
    R = 100  # integer domain size, values in [0, R-1]
    beta = 0.1  # trimming parameter (10% trimmed from each tail)
    epsilon = 0.05  # relative error parameter for q-digest
    adversarial_fraction = 0.1  # 10% adversarial nodes
    seed = 42

    # Create sensor network (spanning tree built inside SensorNetwork)
    network = SensorNetwork(n, w, d, comm_radius, seed=seed, adversarial_fraction=adversarial_fraction)

    # Prepare local data for each sensor node.
    # For each sensor node, generate a data vector and map its coordinates to integers in [0, R-1].
    local_data = {}
    for node in network.nodes:
        vec = node.get_data()
        # Assume sensor values are in [0,100]. Map to [0, R-1] using a simple rounding.
        int_vec = [min(max(int(round(x)), 0), R - 1) for x in vec]
        local_data[node.id] = int_vec

    # Also include the base station's local data.
    base_vec = network.base_station.get_data()
    int_base_vec = [min(max(int(round(x)), 0), R - 1) for x in base_vec]
    local_data[network.base_station.id] = int_base_vec

    # Compute the global aggregator using the spanning tree.
    spanning_tree = network.spanning_tree
    aggregator = TrimmedMeanAggregator(spanning_tree, R, beta, epsilon)
    robust_vector = aggregator.compute_global_aggregator(local_data)
    print("Approximate β-Trimmed Mean Aggregator (per coordinate):")
    print(robust_vector)