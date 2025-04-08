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


#########################################
# Example Usage (for testing)
#########################################
if __name__ == '__main__':
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

    spanning_tree = network.spanning_tree
    aggregator = TrimmedMeanAggregator(spanning_tree, R, beta, epsilon)
    robust_vector = aggregator.compute_global_aggregator(local_data)
    print("Approximate β-Trimmed Mean Aggregator (per coordinate):")
    print(robust_vector)