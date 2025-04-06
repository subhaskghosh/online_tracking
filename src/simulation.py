#!/usr/bin/env python3
"""
simulation.py

Simulate a distributed robust monitoring system using the decentralized
β‐trimmed mean aggregation (via Q‐Digest sketches) and compare it with a
centralized algorithm that has complete information.

The simulation supports:
  1. Varying number of nodes: [50, 100, 150, 200]
  2. Changing the dimension of the vector at each node: [5, 10, 15, 20]
  3. Changing the value of β in [0.1, 0.2, 0.3, 0.4]
  4. Changing the value of ε in [0.01, 0.05]
  5. Changing the adversarial standard deviation in [1.0, 10.0, 100.0]
  6. Comparing the decentralized (periodic and event-driven) algorithms with a centralized aggregator
  7. Measuring the monitoring function value, absolute error, communication overhead, storage,
     power consumption, and estimated lifetime.
  8. Simulating under different distributions (for honest sensors: "correlated_gaussian" or "uniform")
  9. Saving all results in a CSV file with checkpointing for restart
 10. Each simulation defines a threshold for the monitoring function and checks if $f(v)$ exceeds it
 11. Each simulation runs for 1000 iterations to simulate streaming data

This file uses approximate_trimmed_mean.py, event_driven_trimmed_mean.py, and sensor_network.py.
"""

import os
import math
import random
import numpy as np
import pandas as pd
import networkx as nx

from sensor_network import SensorNetwork  # Provided in sensor_network.py
from approximate_trimmed_mean import TrimmedMeanAggregator  # Periodic aggregator
from event_driven_trimmed_mean import EventDrivenTrimmedMeanAggregator  # Event-driven aggregator


######################################################
# Monitoring Function and Centralized Aggregator
######################################################

def monitoring_function(v):
    """Monitoring function defined as the average of the aggregated vector."""
    return np.mean(v)


def centralized_beta_trimmed_mean(data_matrix, beta):
    """
    Compute the exact coordinate-wise β-trimmed mean from the raw data.
    data_matrix: an (n x d) array of sensor data.
    """
    n, d = data_matrix.shape
    agg = np.zeros(d)
    for k in range(d):
        col = np.sort(data_matrix[:, k])
        r = int(math.floor(beta * n))
        if 2 * r >= n:
            agg[k] = 0
        else:
            agg[k] = np.mean(col[r:n - r])
    return agg


######################################################
# Communication, Storage, and Energy Estimates
######################################################

# Set a scaling factor for physical distances.
# Even though w=100 for simulation purposes, we treat each simulation unit as 100 meters.
DISTANCE_SCALE = 100  # 1 unit = 100 m so that a 100x100 grid corresponds to 10 km x 10 km


def qdigest_serialized_size(qdigest, bits_per_bucket=96):
    """
    Return the serialized size (in bits) of a QDigest sketch.
    """
    return bits_per_bucket * len(qdigest.buckets)


def estimate_comm_overhead_from_digests(aggregator, local_data, tree, bits_per_bucket=96):
    """
    Estimate the total communication overhead (in bits) for one full upward aggregation,
    based on the actual serialized sizes of the QDigest sketches transmitted on each edge.
    """
    total_bits = 0

    def traverse(node):
        nonlocal total_bits
        # If leaf: its message size is the sum over its d QDigest sizes.
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

    # Identify the root of the spanning tree.
    root_candidates = [node for node in tree.nodes() if tree.in_degree(node) == 0]
    if not root_candidates:
        raise ValueError("Spanning tree has no root.")
    root = root_candidates[0]
    traverse(root)
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
    For each edge (u,v) in T, energy for a message of size 'message_size' bits transmitted over
    distance d (in meters) is given by:

    E_tx = E_elec * message_size + epsilon_amp * message_size * (d^gamma).

    Here, we scale the distances by DISTANCE_SCALE.
    """
    total_energy = 0.0
    for u, v in T.edges():
        pos_u = np.array(T.nodes[u]['pos'])
        pos_v = np.array(T.nodes[v]['pos'])
        # Scale the simulation distance to physical distance.
        distance = np.linalg.norm(pos_u - pos_v) * DISTANCE_SCALE
        E_tx = E_elec * message_size + epsilon_amp * message_size * (distance ** gamma)
        total_energy += E_tx
    return total_energy


def estimate_power_and_lifetime_from_digests(T, aggregator, local_data, n, d, epsilon,
                                             constant=1.0, battery_capacity=10.0,
                                             E_elec=50e-9, epsilon_amp=100e-12, gamma=2,
                                             bits_per_bucket=96):
    """
    Estimate energy consumption and lifetime for one update event using actual message sizes derived from Q-Digest sketches.
    """
    message_size = estimate_comm_overhead_from_digests(aggregator, local_data, T, bits_per_bucket)
    total_energy = compute_transmission_energy(T, message_size, E_elec, epsilon_amp, gamma)
    avg_energy_per_node = total_energy / n
    lifetime_rounds = battery_capacity / avg_energy_per_node if avg_energy_per_node > 0 else float('inf')
    return total_energy, avg_energy_per_node, lifetime_rounds


def compute_direct_transmission_energy(network, message_size, E_elec=50e-9, epsilon_amp=100e-12, gamma=2):
    """
    Compute the total transmission energy (in Joules) for the centralized setting,
    where each sensor node transmits a message of size 'message_size' bits directly to the base station.
    Here we scale the distance between a node and the base station by DISTANCE_SCALE.
    """
    total_energy = 0.0
    base_pos = np.array(network.base_station.position())
    for node in network.nodes:
        node_pos = np.array(node.position())
        distance = np.linalg.norm(node_pos - base_pos) * DISTANCE_SCALE
        E_tx = E_elec * message_size + epsilon_amp * message_size * (distance ** gamma)
        total_energy += E_tx
    return total_energy


def estimate_central_power_and_lifetime(network, n, d, constant=1.0, battery_capacity=10.0,
                                        E_elec=50e-9, epsilon_amp=100e-12, gamma=2):
    """
    Estimate energy consumption and lifetime for the centralized aggregator,
    where each sensor node transmits raw data directly to the base station.
    Message size per node is assumed to be 32 bits per coordinate.
    """
    message_size = 32 * d
    total_energy = compute_direct_transmission_energy(network, message_size, E_elec, epsilon_amp, gamma)
    avg_energy_per_node = total_energy / n
    lifetime_rounds = battery_capacity / avg_energy_per_node if avg_energy_per_node > 0 else float('inf')
    return total_energy, avg_energy_per_node, lifetime_rounds


######################################################
# Simulation Parameters and Main Loop
######################################################

nodes_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
beta_values = [0.1, 0.2, 0.3, 0.4]
epsilon_values = [0.01, 0.05, 0.1, 0.15]
adversarial_std_devs = [1.0, 10.0, 100.0, 200.0]
honest_distributions = ["correlated_gaussian", "uniform"]
delta_frac = [0.1, 0.15, 0.2, 0.25, 0.3]

# Other simulation parameters
w = 100.0  # Deployment area: 100 x 100 square
comm_radius = [30.0, 25.0, 20.0, 15.0, 10.0, 10.0, 8.0, 8.0, 8.0, 8.0]  # Communication radius
time_iterations = 10  # Number of iterations to simulate streaming data
seed = 81278181  # Random seed for reproducibility

# Domain for Q-Digest: assume sensor values are in [0, 100]
R = 100

# Monitoring threshold: for example, tau = 50.
tau = 50.0


# Checkpoint file for results
results_filename = "simulation_results.csv"
if os.path.exists(results_filename):
    df_results = pd.read_csv(results_filename)
else:
    df_results = pd.DataFrame(columns=[
        "nodes", "dimension", "beta", "epsilon", "delta_frac", "adv_std_dev", "honest_distribution", "iteration",
        "f_approx", "f_event", "f_central", "abs_error_approx", "abs_error_event",
        "comm_overhead_approx", "comm_overhead_event", "comm_overhead_central",
        "storage_approx", "storage_event", "storage_central",
        "power_tx_approx", "power_tx_event", "power_tx_central",
        "avg_energy_per_node_approx", "avg_energy_per_node_event", "avg_energy_per_node_central",
        "lifetime_rounds_approx", "lifetime_rounds_event", "lifetime_rounds_central",
        "threshold", "f_approx_exceeds_threshold", "f_event_exceeds_threshold"
    ])

random.seed(seed)
np.random.seed(seed)

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
                            adversarial_fraction = beta  # Fraction of adversarial nodes (must be <= beta)
                            network = SensorNetwork(n, w, d, cr, seed=seed,
                                                    adversarial_fraction=adversarial_fraction,
                                                    adversarial_std_dev=adv_std,
                                                    honest_distribution=honest_dist)
                            spanning_tree = network.spanning_tree

                            # Instantiate both aggregators.
                            aggregator_periodic = TrimmedMeanAggregator(spanning_tree, R, beta, epsilon)
                            aggregator_event = EventDrivenTrimmedMeanAggregator(spanning_tree, R, beta, epsilon, d_frac)

                            # Collect results for the given configuration.
                            config_results = []
                            for t in range(time_iterations):
                                raw_data_list = []
                                local_data = {}
                                for node in network.nodes:
                                    vec = node.get_data()
                                    # Map real-valued data to integers in [0, R-1] (assuming sensor values in [0, 100]).
                                    int_vec = [min(max(int(round(x)), 0), R - 1) for x in vec]
                                    local_data[node.id] = int_vec
                                    raw_data_list.append(int_vec)
                                # Include base station data.
                                base_vec = network.base_station.get_data()
                                int_base_vec = [min(max(int(round(x)), 0), R - 1) for x in base_vec]
                                local_data[network.base_station.id] = int_base_vec
                                raw_data_list.append(int_base_vec)

                                raw_data = np.array(raw_data_list)  # shape: (n+1, d)

                                # Compute aggregators.
                                approx_agg = aggregator_periodic.compute_global_aggregator(local_data)
                                event_agg = aggregator_event.compute_global_aggregator(local_data)
                                central_agg = centralized_beta_trimmed_mean(raw_data, beta)

                                f_approx = monitoring_function(approx_agg)
                                f_event = monitoring_function(event_agg)
                                f_central = monitoring_function(central_agg)
                                abs_error_approx = abs(f_approx - f_central)
                                abs_error_event = abs(f_event - f_central)

                                comm_overhead_approx = estimate_comm_overhead_from_digests(aggregator_periodic,
                                                                                           local_data, spanning_tree)
                                comm_overhead_event = estimate_comm_overhead_from_digests(aggregator_event, local_data,
                                                                                          spanning_tree)
                                comm_overhead_central = n * d * 32  # 32 bits per coordinate

                                storage_approx = estimate_storage_from_digests(aggregator_periodic, local_data)
                                storage_event = estimate_storage_from_digests(aggregator_event, local_data)
                                storage_central = n * d * 32

                                total_energy_approx, avg_energy_approx, lifetime_rounds_approx = estimate_power_and_lifetime_from_digests(
                                    spanning_tree, aggregator_periodic, local_data, n, d, epsilon,
                                    constant=1.0, battery_capacity=10.0, E_elec=50e-9, epsilon_amp=100e-12, gamma=2)
                                total_energy_event, avg_energy_event, lifetime_rounds_event = estimate_power_and_lifetime_from_digests(
                                    spanning_tree, aggregator_event, local_data, n, d, epsilon,
                                    constant=1.0, battery_capacity=10.0, E_elec=50e-9, epsilon_amp=100e-12, gamma=2)
                                total_energy_central, avg_energy_central, lifetime_rounds_central = estimate_central_power_and_lifetime(
                                    network, n, d, constant=1.0, battery_capacity=10.0, E_elec=50e-9,
                                    epsilon_amp=100e-12, gamma=2)

                                f_approx_exceeds = f_approx > tau
                                f_event_exceeds = f_event > tau

                                config_results.append({
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
                                    "f_event_exceeds_threshold": f_event_exceeds
                                })

                            df_config = pd.DataFrame(config_results)
                            df_results = pd.concat([df_results, df_config], ignore_index=True)
                            df_results.to_csv(results_filename, index=False)
                            print(f"Config {config_str} completed with {time_iterations} iterations.\n")