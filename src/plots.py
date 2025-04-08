#!/usr/bin/env python3
"""
plot_all_results.py

This script reads the simulation results from a CSV file (e.g., simulation_results.csv)
and produces several plots to evaluate the performance of the robust aggregation methods.
All plotting routines are refactored into one single function.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_all_results(csv_filename):
    # Read CSV file into DataFrame
    df = pd.read_csv(csv_filename)

    # Set common style and font sizes.
    plt.rcParams.update({'font.size': 12})

    # -------------------------------
    # 1. Monitoring Function vs. Time
    # -------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df['iteration'], df['f_approx'], label='Periodic Aggregator', alpha=0.8)
    if 'f_event' in df.columns:
        ax1.plot(df['iteration'], df['f_event'], label='Event-Driven Aggregator', alpha=0.8)
    ax1.plot(df['iteration'], df['f_central'], label='Centralized Aggregator', alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Monitoring Function Value')
    ax1.set_title('Monitoring Function Value vs. Time')
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("monitoring_function_vs_time.png")

    # -------------------------------
    # 2. Absolute Error Comparison
    # -------------------------------
    # Line plot over time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if 'abs_error_approx' in df.columns:
        ax2.plot(df['iteration'], df['abs_error_approx'], label='Absolute Error (Periodic)', alpha=0.8)
    if 'abs_error_event' in df.columns:
        ax2.plot(df['iteration'], df['abs_error_event'], label='Absolute Error (Event-Driven)', alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Absolute Error vs. Time')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("absolute_error_vs_time.png")

    # Boxplot of error distribution (aggregated over iterations)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    error_data = []
    labels = []
    if 'abs_error_approx' in df.columns:
        error_data.append(df['abs_error_approx'].dropna())
        labels.append('Periodic')
    if 'abs_error_event' in df.columns:
        error_data.append(df['abs_error_event'].dropna())
        labels.append('Event-Driven')
    ax3.boxplot(error_data, labels=labels)
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Distribution of Absolute Errors')
    fig3.tight_layout()
    fig3.savefig("absolute_error_boxplot.png")

    # -------------------------------
    # 3. Communication Overhead vs. Number of Nodes
    # -------------------------------
    df_nodes = df.groupby('nodes').mean().reset_index()
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    if 'comm_overhead_approx' in df_nodes.columns:
        ax4.plot(df_nodes['nodes'], df_nodes['comm_overhead_approx'], marker='o', label='Periodic')
    if 'comm_overhead_event' in df_nodes.columns:
        ax4.plot(df_nodes['nodes'], df_nodes['comm_overhead_event'], marker='o', label='Event-Driven')
    if 'comm_overhead_central' in df_nodes.columns:
        ax4.plot(df_nodes['nodes'], df_nodes['comm_overhead_central'], marker='o', label='Centralized')
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Communication Overhead (bits)')
    ax4.set_title('Communication Overhead vs. Number of Nodes')
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig("comm_overhead_vs_nodes.png")

    # -------------------------------
    # 4. Energy Consumption and Estimated Lifetime vs. Number of Nodes
    # -------------------------------
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    if 'power_tx_approx' in df_nodes.columns:
        ax5.plot(df_nodes['nodes'], df_nodes['power_tx_approx'], marker='o', label='Periodic')
    if 'power_tx_event' in df_nodes.columns:
        ax5.plot(df_nodes['nodes'], df_nodes['power_tx_event'], marker='o', label='Event-Driven')
    if 'power_tx_central' in df_nodes.columns:
        ax5.plot(df_nodes['nodes'], df_nodes['power_tx_central'], marker='o', label='Centralized')
    ax5.set_xlabel('Number of Nodes')
    ax5.set_ylabel('Total Transmission Energy (Joules)')
    ax5.set_title('Energy Consumption vs. Number of Nodes')
    ax5.legend()
    fig5.tight_layout()
    fig5.savefig("energy_vs_nodes.png")

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    if 'lifetime_rounds_approx' in df_nodes.columns:
        ax6.plot(df_nodes['nodes'], df_nodes['lifetime_rounds_approx'], marker='o', label='Periodic')
    if 'lifetime_rounds_event' in df_nodes.columns:
        ax6.plot(df_nodes['nodes'], df_nodes['lifetime_rounds_event'], marker='o', label='Event-Driven')
    if 'lifetime_rounds_central' in df_nodes.columns:
        ax6.plot(df_nodes['nodes'], df_nodes['lifetime_rounds_central'], marker='o', label='Centralized')
    ax6.set_xlabel('Number of Nodes')
    ax6.set_ylabel('Estimated Lifetime (rounds)')
    ax6.set_title('Estimated Lifetime vs. Number of Nodes')
    ax6.legend()
    fig6.tight_layout()
    fig6.savefig("lifetime_vs_nodes.png")

    # -------------------------------
    # 5. Parameter Sensitivity Analysis (Heatmaps)
    # -------------------------------
    # Heatmap for absolute error (periodic) vs epsilon and beta
    pivot_approx = df.pivot_table(index='epsilon', columns='beta', values='abs_error_approx', aggfunc=np.mean)
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_approx, annot=True, fmt=".3f", cmap="viridis", ax=ax7)
    ax7.set_title("Mean Absolute Error (Periodic) vs. epsilon and beta")
    ax7.set_xlabel("beta")
    ax7.set_ylabel("epsilon")
    fig7.tight_layout()
    fig7.savefig("heatmap_abs_error_periodic.png")

    # Heatmap for absolute error (event-driven) vs epsilon and beta
    if 'abs_error_event' in df.columns:
        pivot_event = df.pivot_table(index='epsilon', columns='beta', values='abs_error_event', aggfunc=np.mean)
        fig8, ax8 = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_event, annot=True, fmt=".3f", cmap="viridis", ax=ax8)
        ax8.set_title("Mean Absolute Error (Event-Driven) vs. epsilon and beta")
        ax8.set_xlabel("beta")
        ax8.set_ylabel("epsilon")
        fig8.tight_layout()
        fig8.savefig("heatmap_abs_error_event.png")

    # -------------------------------
    # 6. Update Trigger Frequency (Event-Driven Only)
    # -------------------------------
    if 'trigger' in df.columns:
        fig9, ax9 = plt.subplots(figsize=(10, 6))
        sns.histplot(df['trigger'], bins=20, kde=False, color='skyblue', edgecolor='black', ax=ax9)
        ax9.set_xlabel('Number of Triggers per Round')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Distribution of Update Trigger Frequency (Event-Driven)')
        fig9.tight_layout()
        fig9.savefig("trigger_frequency_histogram.png")
    else:
        print("No trigger count data available in the CSV.")

    # Show all figures at the end
    plt.show()


if __name__ == '__main__':
    # Change this filename if needed.
    csv_filename = "/Users/subhas/src/papers/online_tracking/code/src/simulation_results.csv"
    plot_all_results(csv_filename)