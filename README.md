# Robust Distributed Monitoring Simulation

This repository contains a simulation framework for robust distributed monitoring in wireless sensor networks. The simulation evaluates decentralized robust aggregation algorithms based on the coordinate-wise β‐trimmed mean using Q‐Digest sketches. It also compares these decentralized (both periodic and event-driven) approaches with a centralized aggregator that collects all sensor data.

## Overview

In our model, a set of sensor nodes is deployed uniformly at random in a square area. Each sensor node continuously generates a $d$-dimensional data vector according to one of several distributions:
- **Honest nodes** generate data using either a temporally correlated Gaussian process (with spatial bias and an AR(1) model) or a uniform distribution.
- **Adversarial nodes** (up to a specified fraction) generate data from a Gaussian distribution with mean 0.0 and a variable standard deviation to simulate extreme conditions.

The robust aggregation is implemented using Q-Digest sketches that summarize each coordinate’s data. Two decentralized robust aggregation schemes are provided:
1. **Periodic Aggregator:** Computes a coordinate-wise β‐trimmed mean by merging Q-Digest sketches along a spanning tree.
2. **Event-Driven Aggregator:** Maintains per-node safe intervals and only triggers an update (i.e., re-aggregation) when a node’s local data falls outside its assigned interval.

A centralized aggregator is used as a baseline for comparison. The simulation framework also estimates communication overhead, storage requirements, energy consumption, and node lifetime based on a wireless transmission energy model.

## Repository Structure

- **`sensor_network.py`**  
  Implements the sensor network model. Sensor nodes are deployed in a $w \times w$ area and connected based on a communication radius. A base station is added, and a spanning tree is computed for decentralized aggregation.

- **`approximate_trimmed_mean.py`**  
  Contains the implementation of the periodic robust aggregator using Q-Digest sketches for computing the coordinate-wise β‐trimmed mean.

- **`event_driven_trimmed_mean.py`**  
  Implements the event‐driven robust aggregator that uses safe intervals to trigger updates only when needed.

- **`simulation.py`**  
  The main simulation script that runs the experiments. It supports varying parameters such as the number of nodes, data dimension, trimming fraction $\beta$, approximation error $\varepsilon$, adversarial standard deviation, and the type of distribution for honest sensors. It computes and logs metrics including the monitoring function value, absolute error between aggregators, communication overhead, storage requirements, energy consumption, and estimated lifetime.

- **`README.md`**  
  This file.

## Dependencies

The code is implemented in Python 3 and relies on the following packages:
- numpy
- networkx
- pandas
- matplotlib
- seaborn

You can install the dependencies using pip:
```bash
pip install numpy networkx pandas matplotlib seaborn
