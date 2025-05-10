#!/usr/bin/env python3
"""
simulate_mawi.py

Use the MAWI Working Group Traffic Archive (Samplepoint-F) as distributed
“virtual sensor” streams. Partition the trace into /24 source subnets,
compute per‐epoch feature vectors, build a random spanning tree among
virtual sensors, and apply the existing aggregation algorithms
(periodic Q-Digest, event-driven, central histogram) without modifying
their implementations. Simulate a fraction of sensors as Byzantine nodes
producing adversarial noise, measure cumulative communication bits saved,
and plot the resulting network.
"""
import random
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv

# Import aggregation classes (assumed on PYTHONPATH)
from approximate_trimmed_mean import TrimmedMeanAggregator
from event_driven_trimmed_mean import EventDrivenTrimmedMeanAggregator
from simple_list import SimpleListAggregator

# ----------------------------------------
# Communication‐estimation helpers
# ----------------------------------------
def simplelist_serialized_size(hist, bits_per_bucket=96):
    return bits_per_bucket * len(hist)

def estimate_comm_overhead_from_simple_list(aggregator, local_data, tree, bits_per_bucket=96):
    total_bits = 0
    def traverse(node):
        nonlocal total_bits
        if tree.out_degree(node) == 0:
            leaf = aggregator.build_leaf_histogram(local_data[node])
            total_bits += sum(simplelist_serialized_size(h, bits_per_bucket) for h in leaf)
            return leaf
        children = [traverse(c) for c in tree.successors(node)]
        merged = children[0]
        for hlist in children[1:]:
            merged = [aggregator.merge_histograms(m, h) for m,h in zip(merged, hlist)]
        total_bits += sum(simplelist_serialized_size(h, bits_per_bucket) for h in merged)
        return merged
    root = next(n for n in tree.nodes() if tree.in_degree(n)==0)
    traverse(root)
    return total_bits

def qdigest_serialized_size(qd, bits_per_bucket=96):
    return bits_per_bucket * len(qd.buckets)

def estimate_comm_overhead_from_digests(aggregator, local_data, tree, bits_per_bucket=96):
    total_bits = 0
    def traverse(node):
        nonlocal total_bits
        if tree.out_degree(node) == 0:
            ld = aggregator.build_leaf_digest(local_data[node])
            total_bits += sum(qdigest_serialized_size(q, bits_per_bucket) for q in ld)
            return ld
        children = [traverse(c) for c in tree.successors(node)]
        merged = aggregator.merge_digests(children)
        local_d = aggregator.build_leaf_digest(local_data.get(node, [0]*len(merged)))
        for q_loc, q_ch in zip(local_d, merged):
            m = q_loc.merge(q_ch)
            total_bits += qdigest_serialized_size(m, bits_per_bucket)
        return merged
    root = next(n for n in tree.nodes() if tree.in_degree(n)==0)
    traverse(root)
    return total_bits

def estimate_comm_overhead_from_digests_event(aggregator, local_data, tree, bits_per_bucket=96):
    if not hasattr(aggregator, "get_triggered_nodes"):
        return estimate_comm_overhead_from_digests(aggregator, local_data, tree, bits_per_bucket)
    triggered = aggregator.get_triggered_nodes()
    if not triggered:
        return 0
    root = next(n for n in tree.nodes() if tree.in_degree(n)==0)
    edges = set()
    for v in triggered:
        cur = v
        while cur != root:
            p = next(tree.predecessors(cur))
            edges.add((p,cur))
            cur = p
    total = 0
    for _, v in edges:
        ld = aggregator.build_leaf_digest(local_data.get(v, [0]*1))
        total += sum(qdigest_serialized_size(q, bits_per_bucket) for q in ld)
    return total

# ----------------------------------------
# Configuration
# ----------------------------------------
DATA_FILE           = Path("./data/mawi_sample_small.csv")
INTERVAL_SEC        = 1
BYZANTINE_FRACTION  = 0.1
ADV_STD_DEV         = 1e6
R                   = 10000
BETA                = 0.1
EPSILON             = 0.05
DELTA_FRAC          = 0.01

# ----------------------------------------
# Monitoring function
# ----------------------------------------
def monitoring_function(v):
    return np.mean(v)

# ----------------------------------------
# Bursty adversarial stream
# ----------------------------------------
class BurstyAdversarialDataStream:
    def __init__(self, d, normal_sigma=100.0, burst_prob=0.25,
                 pareto_alpha=2.5, pareto_scale=500.0):
        self.d            = d
        self.normal_sigma = normal_sigma
        self.burst_prob   = burst_prob
        self.pareto_alpha = pareto_alpha
        self.pareto_scale = pareto_scale
    def get_next(self):
        if random.random() < self.burst_prob:
            u     = np.random.random(self.d)
            burst = self.pareto_scale * (u**(-1.0/self.pareto_alpha))
            signs = np.random.choice([1,-1], size=self.d)
            return (burst*signs).tolist()
        else:
            return np.random.normal(0, self.normal_sigma, self.d).tolist()

# ----------------------------------------
# Draw the network + spanning tree
# ----------------------------------------
def draw_network(graph, spanning_tree, filename="mawi_network.pdf"):
    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, pos, node_size=30, node_color='skyblue')
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nx.draw_networkx_edges(spanning_tree, pos, edge_color='green', width=1.0)
    plt.title("MAWI Virtual Sensor Network and Spanning Tree")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ----------------------------------------
# Load & preprocess MAWI trace
# ----------------------------------------
print('Reading data…')
df = pd.read_csv(
    DATA_FILE,
    sep=r'(?<!\\),',        # only split on commas *not* preceded by backslash
    engine='python',
    header=0,
    names=[
        'frame.time_epoch',  # TShark’s first field
        'ip.src',            # second field
        'ip.dst',            # third
        'ip.len',            # fourth
        'tcp.flags'          # fifth
    ],
    dtype=str
)

# now un‐escape any “\,” sequences that TShark encoded in‐field commas
for col in ('ip.src','ip.dst','tcp.flags'):
    df[col] = df[col].str.replace(r'\\,', ',', regex=True)

# rename into the names your downstream code expects:
df = df.rename(columns={
    'frame.time_epoch': 'time',
    'ip.src':          'srcIP',
    'ip.dst':          'dstIP',
    'ip.len':          'bytes',
    'tcp.flags':       'flags'
})

# parse the time and add a constant “1 packet per line” column
df['time'] = pd.to_datetime(df['time'].astype(float), unit='s')
df['pkts'] = 1

# derive your /24 sensor and epoch exactly as before
df['sensor'] = df['srcIP'].str.extract(r'^(\d+\.\d+\.\d+)\.')[0] + '.0/24'
df['epoch']  = df['time'].dt.floor(f'{INTERVAL_SEC}s')

# strip backslashes and commas, then replace any empty strings or NaNs with 0, then cast
df['bytes'] = (
    df['bytes']
      .str.replace(r'[\\,]', '', regex=True)   # remove backslashes & commas
      .replace('', np.nan)                     # convert any empty string to NaN
      .fillna(0)                               # fill those NaNs
      .astype(int)                             # now safe to cast
)

print('Built DataFrame with columns:', df.columns.tolist())

# ----------------------------------------
# Build per‐epoch, per‐sensor feature vectors
# ----------------------------------------
def entropy(counts):
    total = sum(counts.values())
    p = np.array(list(counts.values())) / total
    return -np.sum(p * np.log(p + 1e-12))

print('Building time series records…')
records = {}
for (ep, sen), g in df.groupby(['epoch','sensor']):
    sb    = g['bytes'].sum()
    sp    = g['pkts'].sum()
    di    = g['dstIP'].value_counts().to_dict()
    # no port field here, set zero
    H_ip  = entropy(di)
    syn   = g['flags'].str.contains('SYN').sum()
    rst   = g['flags'].str.contains('RST').sum()
    records.setdefault(ep, {})[sen] = [sb, sp, H_ip, syn, rst]
print(f'Finished building records; epochs: {len(records)}')

# list of sensors
sensors    = sorted({s for rec in records.values() for s in rec})
n_sensors  = len(sensors)
root       = sensors[0]
print(f'Found {n_sensors} virtual sensors; root={root}')

# ----------------------------------------
# Build a random spanning tree
# ----------------------------------------
print(f'Building random spanning tree over {n_sensors} sensors…')
G = nx.Graph()
G.add_nodes_from(sensors)
remaining = sensors.copy()
random.shuffle(remaining)
connected = { remaining.pop() }
while remaining:
    v = remaining.pop()
    u = random.choice(list(connected))
    G.add_edge(u, v)
    connected.add(v)
assert nx.is_connected(G)
T = nx.bfs_tree(G, source=root)
#draw_network(G, T)

# ----------------------------------------
# Byzantine streams
# ----------------------------------------
d = len(next(iter(records.values())).values().__iter__().__next__())
byz_ids = set(random.sample(sensors, int(BYZANTINE_FRACTION * n_sensors)))
adv_streams = {
    s: BurstyAdversarialDataStream(d, normal_sigma=ADV_STD_DEV)
    for s in byz_ids
}

# ----------------------------------------
# Instantiate aggregators
# ----------------------------------------
aggr_periodic = TrimmedMeanAggregator(T, R, BETA, EPSILON)
aggr_event    = EventDrivenTrimmedMeanAggregator(T, R, BETA, EPSILON, DELTA_FRAC)
aggr_central  = SimpleListAggregator(T, R, BETA)

# ----------------------------------------
# Simulation loop & communication summary
# ----------------------------------------
cum_per = cum_evt = cum_cen = 0
results = []
print('Starting simulation…')
for ep, rec in sorted(records.items()):
    local = {}
    for s in sensors:
        if s in byz_ids:
            vec = adv_streams[s].get_next()
        else:
            vec = rec.get(s, [0.0]*d)
        local[s] = [min(max(int(round(float(x))), 0), R-1) for x in vec]

    approx = aggr_periodic.compute_global_aggregator(local)
    event = aggr_event.compute_global_aggregator(local)
    central_trim = aggr_central.compute_global_aggregator(local)
    central_mean = aggr_central.compute_global_aggregator_wo_trimming(local)

    b_per = estimate_comm_overhead_from_digests(aggr_periodic, local, T)
    b_evt = estimate_comm_overhead_from_digests_event(aggr_event, local, T)
    b_cen = estimate_comm_overhead_from_simple_list(aggr_central, local, T)

    cum_per += b_per
    cum_evt += b_evt
    cum_cen += b_cen

    results.append({
        'epoch': ep,
        'f_approx': monitoring_function(approx),
        'f_event': monitoring_function(event),
        'f_central': monitoring_function(central_trim),
        'f_mean': monitoring_function(central_mean),
        'comm_bits_periodic': b_per,
        'comm_bits_event': b_evt,
        'comm_bits_central': b_cen
    })

print("\n=== Communication summary ===")
print(f"Periodic Q-Digest : {cum_per:,d} bits")
print(f"Event-driven      : {cum_evt:,d} bits")
print(f"SimpleList        : {cum_cen:,d} bits")
if cum_per:
    saving = 100 * (1 - cum_evt/cum_per)
    print(f"Event-driven cuts message volume by {saving:.2f}% vs periodic.")

# save results
pd.DataFrame(results).to_csv('mawi_simulation_results_small_high.csv', index=False)
print("MAWI simulation complete → mawi_simulation_results.csv")