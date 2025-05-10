#!/usr/bin/env python3
"""
simulate_air_quality.py

Use the Air Quality & Meteorology dataset as distributed sensor streams.
Construct a geographic connectivity graph among stations, form a spanning tree,
and apply the existing aggregation algorithms (periodic Q-Digest, event-driven,
central histogram) without modifying their implementations.
Simulate a fraction of stations as Byzantine nodes producing adversarial noise.
"""
import os
import math
import random
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Import aggregation classes and data streams
from approximate_trimmed_mean import TrimmedMeanAggregator  # Q-Digest based aggregator (periodic)
from event_driven_trimmed_mean import EventDrivenTrimmedMeanAggregator  # Event-driven aggregator
from simple_list import SimpleListAggregator  # New baseline aggregator using list-of-histograms
from sensor_network import AdversarialDataStream

# Set plotting style.
plt.style.use('seaborn-whitegrid')
#rc('text', usetex=True)
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-ticks")

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
        distance = np.linalg.norm(pos_u - pos_v)
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

#----------------------------------------
# Helper: haversine distance (km)
#----------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R_earth = 6371.0  # Earth radius in km
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = (math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2)
    return 2 * R_earth * math.asin(math.sqrt(a))


#----------------------------------------
# Configuration
#----------------------------------------
def draw_network(graph, spanning_tree):
    pos = nx.get_node_attributes(graph, 'pos')
    plt.figure(figsize=(8, 8))
    sensor_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'sensor']
    base_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'base station']
    nx.draw_networkx_nodes(graph, pos, nodelist=sensor_nodes, node_color='blue', node_size=10,
                           label='Sensor Nodes')
    nx.draw_networkx_nodes(graph, pos, nodelist=base_nodes, node_color='red', node_size=15,
                           label='Base Station')
    nx.draw_networkx_edges(graph, pos, alpha=0.15)
    nx.draw_networkx_edges(spanning_tree, pos, edge_color='green', width=1, arrows=False,
                           label='Spanning Tree')
    plt.legend()
    plt.axis('equal')
    plt.savefig('aq_network.pdf', dpi=300)
    plt.show()

DATA_DIR = Path("./data")  # directory containing Station.txt and CrawledData.txt
STATION_FILE = DATA_DIR / "Station.txt"
DATA_FILE    = DATA_DIR / "CrawledData.txt"
COMM_RADIUS_KM = 20.0         # connect stations within 10 km
BYZANTINE_FRACTION = 0.1      # 10% Byzantine nodes
ADV_STD_DEV = 100.0           # adversarial noise sigma
R = 1000                      # integer domain size for all features
BETA = 0.1                    # trimming fraction
EPSILON = 0.05                # quantile sketch error
DELTA_FRAC = 0.01              # safe-interval fraction

#----------------------------------------
# Load station metadata
#----------------------------------------
stations = pd.read_csv(
    STATION_FILE,
    sep=r"\s+|,",
    names=["station_id", "station_name", "latitude", "longitude"],
    header=0,
    engine="python",
    dtype={"station_id": str},
)

#----------------------------------------
# Build geographic graph
#----------------------------------------
g = nx.Graph()
root = stations['station_id'].iloc[0]
for _, row in stations.iterrows():
    sid = row.station_id
    if sid not in [root]:
        g.add_node(sid, pos=(row.latitude, row.longitude), type='sensor')
    else:
        g.add_node(sid, pos=(row.latitude, row.longitude), type='base station')

nodes = list(g.nodes())

for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        u, v = nodes[i], nodes[j]
        lat1, lon1 = g.nodes[u]['pos']
        lat2, lon2 = g.nodes[v]['pos']
        d_km = haversine(lat1, lon1, lat2, lon2)
        if d_km <= COMM_RADIUS_KM:
            g.add_edge(u, v, weight=d_km)

# Ensure connectivity by adding MST edges if needed
if not nx.is_connected(g):
    full = nx.Graph()
    full.add_nodes_from(g.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            lat1, lon1 = g.nodes[u]['pos']
            lat2, lon2 = g.nodes[v]['pos']
            d_km = haversine(lat1, lon1, lat2, lon2)
            full.add_edge(u, v, weight=d_km)
    mst = nx.minimum_spanning_tree(full, weight='weight')
    for u, v, data_edge in mst.edges(data=True):
        if not g.has_edge(u, v):
            g.add_edge(u, v, weight=data_edge['weight'])
assert nx.is_connected(g), "Graph should now be connected"

# Compute a spanning tree via BFS

T = nx.bfs_tree(g, source=root)
for node in T.nodes():
    T.nodes[node]['pos'] = g.nodes[node]['pos']

#draw_network(g, T)
#----------------------------------------
# Load time-series data
#----------------------------------------
data = pd.read_csv(
    DATA_FILE,
    parse_dates=['time'],
    names=[
        'station_id', 'time', 'PM25', 'PM10', 'NO2',
        'temperature', 'pressure', 'humidity', 'wind', 'weather'
    ],
    header=0,
    engine='python',
    dtype={'station_id': str, 'weather': float}
)
# Fill missing weather values and convert to int (use -1 for unknown)
data['weather'] = data['weather'].fillna(-1).astype(int)

print(data.head(2))

# Build per-timestamp sensor vectors
timestamps = sorted(data['time'].unique())
records = {}
for t in timestamps:
    df_t = data[data['time'] == t]
    rec = {}
    for _, row in df_t.iterrows():
        sid = row.station_id
        vec = [
            row.PM25, row.PM10, row.NO2,
            row.temperature, row.pressure,
            row.humidity, row.wind
        ]
        rec[sid] = vec
    records[t] = rec

# Identify Byzantine nodes
# Determine feature dimension from a sample record
sample_record = next(iter(records.values()))  # dict: station_id -> vector
sample_vec = next(iter(sample_record.values()))
d = len(sample_vec)  # number of features per station
num_byz = int(BYZANTINE_FRACTION * len(nodes))
byz_ids = set(random.sample(nodes, num_byz))
# Create adversarial streams using feature dimension d
adv_streams = {
    sid: AdversarialDataStream(d=d, std_dev=ADV_STD_DEV)
    for sid in byz_ids
}

# Instantiate aggregators
aggr_periodic = TrimmedMeanAggregator(T, R, BETA, EPSILON)
aggr_event    = EventDrivenTrimmedMeanAggregator(T, R, BETA, EPSILON, DELTA_FRAC)
aggr_central  = SimpleListAggregator(T, R, BETA)

# cumulative communication counters (bits)
cum_bits_periodic = 0          # periodic Q‑Digest
cum_bits_event    = 0          # event‑driven Q‑Digest
cum_bits_central  = 0          # SimpleList baseline

# Run simulation
results = []
for t, rec in records.items():
    print(f'Simulating: {t}')
    local_data = {}
    # assemble local data for each station
    for sid in nodes:
        if sid in byz_ids:
            vec = adv_streams[sid].get_next()
        else:
            vec = rec.get(sid, [0]*d)
        # Impute any NaNs in vector with zero
        vec = [0 if pd.isna(x) else x for x in vec]
        # round & clamp to [0, R-1]
        int_vec = [min(max(int(round(x)), 0), R-1) for x in vec]
        local_data[sid] = int_vec
    # Compute aggregators
    approx = aggr_periodic.compute_global_aggregator(local_data)
    event  = aggr_event.compute_global_aggregator(local_data)
    central_trim = aggr_central.compute_global_aggregator(local_data)
    central_mean = aggr_central.compute_global_aggregator_wo_trimming(local_data)

    f_periodic = monitoring_function(approx)
    f_event = monitoring_function(event)
    f_central = monitoring_function(central_trim)

    # --- communication overhead ---
    comm_overhead_periodic = estimate_comm_overhead_from_digests(
        aggr_periodic, local_data, T)

    comm_overhead_event = estimate_comm_overhead_from_digests_event(
        aggr_event, local_data, T)

    comm_overhead_central = estimate_comm_overhead_from_simple_list(
        aggr_central, local_data, T)

    cum_bits_periodic += comm_overhead_periodic
    cum_bits_event += comm_overhead_event
    cum_bits_central += comm_overhead_central

    # --- storage (bits per node) ---
    storage_periodic = estimate_storage_from_digests(
        aggr_periodic, local_data)

    storage_event = estimate_storage_from_digests(
        aggr_event, local_data)

    storage_central = estimate_storage_from_simple_list(
        aggr_central, local_data)

    # --- power, average energy and lifetime ---
    totE_per, avgE_per, life_per = estimate_power_and_lifetime_from_digests(
        T, aggr_periodic, local_data, len(nodes), 1, EPSILON)

    totE_evt, avgE_evt, life_evt = estimate_power_and_lifetime_from_digests(
        T, aggr_event, local_data, len(nodes), 1, EPSILON)

    totE_cen, avgE_cen, life_cen = estimate_power_and_lifetime_from_simple_list(
        T, aggr_central, local_data, len(nodes), 1)

    results.append({
        'time': t,
        'f_approx': float(np.mean(approx)),
        'f_event': float(np.mean(event)),
        'f_central': float(np.mean(central_trim)),
        'f_mean': float(np.mean(central_mean)),

        'comm_bits_periodic': comm_overhead_periodic,
        'comm_bits_event': comm_overhead_event,
        'comm_bits_central': comm_overhead_central,

        'storage_periodic_bits': storage_periodic,
        'storage_event_bits': storage_event,
        'storage_central_bits': storage_central,

        'energy_periodic_J': totE_per,
        'energy_event_J': totE_evt,
        'energy_central_J': totE_cen,

        'lifetime_periodic_rounds': life_per,
        'lifetime_event_rounds': life_evt,
        'lifetime_central_rounds': life_cen,
    })
    print(results[-1])

print("\n=== Communication summary ===")
print(f"Periodic Q‑Digest : {cum_bits_periodic:,d} bits")
print(f"Event‑driven      : {cum_bits_event:,d} bits")
print(f"SimpleList        : {cum_bits_central:,d} bits")

if cum_bits_periodic:
    saving = 100.0*(1.0 - cum_bits_event / cum_bits_periodic)
    print(f"Event‑driven variant cuts message volume by "
          f"{saving:.2f}% versus periodic.")

# Save results
df = pd.DataFrame(results)
df.to_csv('air_quality_simulation_results.csv', index=False)
print("Simulation complete. Results saved to air_quality_simulation_results.csv")
