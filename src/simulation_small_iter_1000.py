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
import math
import random
import numpy as np
import pandas as pd
import networkx as nx

from sensor_network import SensorNetwork  # Provided in sensor_network.py
from approximate_trimmed_mean import TrimmedMeanAggregator  # Q-Digest based aggregator (periodic)
from event_driven_trimmed_mean import EventDrivenTrimmedMeanAggregator  # Event-driven aggregator
from simple_list import SimpleListAggregator  # New baseline aggregator using list-of-histograms

# Define the DISTANCE_SCALE constant
DISTANCE_SCALE = 100  # or another value appropriate for your simulation

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
nodes_list = [100]
dimensions = [100]
beta_values = [0.1, 0.2, 0.3, 0.4]
epsilon_values = [0.1, 0.2, 0.3]
adversarial_std_devs = [200.0]
honest_distributions = ["correlated_gaussian"]
delta_frac = [0.01]
R_values = [1000]            # Domain for aggregators (sensor values in [0, R-1]).

# Other simulation parameters.
w = 100.0           # Deployment area: 100 x 100 square.
comm_radius = [25.0]  # Communication radii.
time_iterations = 1000  # Number of simulation iterations.
seed = 81278181      # Random seed for reproducibility.

tau = 50.0           # Monitoring threshold.

# Checkpoint file for results.
results_filename = "simulation_results_small_itr_1000.csv"
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