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
            row.PM10
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
    results.append({
        'time': t,
        'f_approx': float(np.mean(approx)),
        'f_event': float(np.mean(event)),
        'f_central': float(np.mean(central_trim)),
        'f_mean': float(np.mean(central_mean)),
    })
    print(results[-1])

# Save results
df = pd.DataFrame(results)
df.to_csv('air_quality_simulation_results_PM10.csv', index=False)
print("Simulation complete. Results saved to air_quality_simulation_results.csv")
