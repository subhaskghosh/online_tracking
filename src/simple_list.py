#!/usr/bin/env python3
"""
simple_list.py

Implements a centralized aggregation algorithm that uses lossless histograms (lists of counts)
to compute the exact β-trimmed mean per coordinate. At each node in the spanning tree, the
local histogram for each coordinate (with bucket width 1) is merged with those coming from its
children. The final aggregated histogram at the root allows for exact quantile queries and trimmed mean computations.
"""

import numpy as np

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

# --- Example test harness (for standalone testing) ---
if __name__ == '__main__':
    import networkx as nx

    # Create a simple spanning tree using networkx.
    T = nx.DiGraph()
    T.add_node('root')
    T.add_node('child1')
    T.add_node('child2')
    T.add_edge('root', 'child1')
    T.add_edge('root', 'child2')
    T.add_node('leaf1')
    T.add_edge('child1', 'leaf1')
    T.add_node('leaf2')
    T.add_edge('child2', 'leaf2')

    # Define sample local data for each node; assume sensor values in 2 dimensions.
    local_data = {
        'root': [14, 24],
        'child1': [12, 22],
        'child2': [13, 23],
        'leaf1': [10, 20],
        'leaf2': [15, 25]
    }

    R = 100       # Integer domain size: values will be in [0, 99]
    beta = 0.1    # 10% trimming from each tail

    aggregator = SimpleListAggregator(T, R, beta)
    global_aggregator = aggregator.compute_global_aggregator(local_data)
    print("Exact β-Trimmed Mean per Coordinate:")
    print(global_aggregator)