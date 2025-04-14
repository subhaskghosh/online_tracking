import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import rc
import numpy as np

dpi = 300
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (8, 6)

# Create output directory if it doesn't exist
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

def melt_and_map(df, id_vars, value_vars, var_name, value_name, map_dict):
    melted = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    melted[var_name] = melted[var_name].map(map_dict)
    return melted

def create_facet_plot(df, row, col, x, y, hue, title, x_label, y_label, filename):
    sns.set_style("white")
    g = sns.FacetGrid(df, row=row, col=col, margin_titles=True, height=3.5, palette='Spectral')
    g.map_dataframe(sns.lineplot, x=x, y=y, hue=hue, marker='o', ci='sd')
    g.add_legend(title=hue)
    g.set_axis_labels(x_label, y_label)
    g.fig.subplots_adjust(top=0.9)
    g.set_titles(size=20)
    print(f'{title} → {filename}')
    g.savefig(os.path.join(output_dir, filename))
    plt.close(g.fig)

# Load the data
df = pd.read_csv("src/simulation_results_small.csv")

# ---------------- Plot 1: Error vs Nodes (row=beta, col=epsilon) ----------------
error_map = {'abs_error_approx': 'Approximate Q-Digest', 'abs_error_event': 'Event-driven Approximate Q-Digest'}
error_df = melt_and_map(df, ['nodes', 'beta', 'epsilon'], ['abs_error_approx', 'abs_error_event'], 'aggregator', 'abs_error', error_map)
error_df = error_df.rename(columns={"beta": r"$ \beta $", "epsilon": r"$ \varepsilon $", "nodes": r"$ n $", "abs_error": "Absolute Error"})
create_facet_plot(
    error_df, r"$ \beta $", r"$ \varepsilon $", r"$ n $", "Absolute Error", 'aggregator',
    "Absolute Error vs. Number of Nodes",
    "Number of Nodes ($n$)", "Absolute Error", "error_beta_epsilon.pdf"
)

# ---------------- Plot 2: Error vs Nodes (row=epsilon, col=beta) ----------------
create_facet_plot(
    error_df, r"$ \varepsilon $", r"$ \beta $", r"$ n $", "Absolute Error", 'aggregator',
    "Absolute Error vs. Number of Nodes",
    "Number of Nodes ($n$)", "Absolute Error", "error_epsilon_beta.pdf"
)

# ---------------- Plot 3: Error vs Nodes (row=R, col=beta) ----------------
if 'R' in df.columns:
    error_df_with_R = melt_and_map(df, ['nodes', 'beta', 'epsilon', 'R'], ['abs_error_approx', 'abs_error_event'], 'aggregator', 'abs_error', error_map)
    error_df_with_R = error_df_with_R.loc[error_df_with_R['R'].isin([100, 500, 1000])]
    error_df_with_R = error_df_with_R.rename(
        columns={"beta": r"$ \beta $", "epsilon": r"$ \varepsilon $", "nodes": r"$ n $", "abs_error": "Absolute Error", "R": r"$ R $"})

    create_facet_plot(
        error_df_with_R, r"$ R $", r"$ \beta $", r"$ n $", "Absolute Error", 'aggregator',
        "Absolute Error vs. Number of Nodes",
        "Number of Nodes ($n$)", "Absolute Error", "error_R_beta.pdf"
    )

# ---------------- Plot 4: Energy per Node ----------------
energy_map = {
    'avg_energy_per_node_approx': 'Approximate Q-Digest',
    'avg_energy_per_node_event': 'Event-driven Approximate Q-Digest',
    'avg_energy_per_node_central': 'List Summarization'
}
energy_df = melt_and_map(df, ['nodes', 'dimension', 'epsilon'], list(energy_map.keys()), 'aggregator', 'avg_energy_per_node', energy_map)
energy_df = energy_df.loc[energy_df['dimension'].isin([20,60,80])]
energy_df = energy_df.rename(
        columns={"dimension": r"$ d $", "epsilon": r"$ \varepsilon $", "nodes": r"$ n $", "avg_energy_per_node": "Avg. Energy per Node"})
create_facet_plot(
    energy_df, r"$ d $", r"$ \varepsilon $", r"$ n $", "Avg. Energy per Node", 'aggregator',
    "Average Energy per Node vs. Number of Nodes\n(Facets: Rows=Dimension, Columns=ε)",
    "Number of Nodes ($n$)", "Average Energy per Node (Joules)", "energy_per_node.pdf"
)

# ---------------- Plot 5: Communication Overhead (dimension vs epsilon) ----------------
comm_map = {
    'comm_overhead_approx': 'Approximate Q-Digest',
    'comm_overhead_event': 'Event-driven Approximate Q-Digest',
    'comm_overhead_central': 'List Summarization'
}
comm_df1 = melt_and_map(df, ['nodes', 'dimension', 'epsilon'], list(comm_map.keys()), 'aggregator', 'comm_overhead', comm_map)
comm_df1 = comm_df1.loc[comm_df1['dimension'].isin([20,60,80])]
comm_df1 = comm_df1.rename(
        columns={"dimension": r"$ d $", "epsilon": r"$ \varepsilon $", "nodes": r"$ n $", "comm_overhead": "Communication Overhead"})
create_facet_plot(
    comm_df1, r"$ d $", r"$ \varepsilon $", r"$ n $", "Communication Overhead", 'aggregator',
    "Communication Overhead vs. Number of Nodes\n(Facets: Rows=Dimension, Columns=ε)",
    "Number of Nodes ($n$)", "Communication Overhead (bits)", "comm_overhead_dimension_epsilon.pdf"
)

# ---------------- Plot 6: Communication Overhead (beta vs epsilon) ----------------
comm_df2 = melt_and_map(df, ['nodes', 'beta', 'epsilon'], list(comm_map.keys()), 'aggregator', 'comm_overhead', comm_map)
comm_df2 = comm_df2.rename(
        columns={"beta": r"$ \beta $", "epsilon": r"$ \varepsilon $", "nodes": r"$ n $", "comm_overhead": "Communication Overhead"})
create_facet_plot(
    comm_df2, r"$ \beta $", r"$ \varepsilon $", r"$ n $", "Communication Overhead", 'aggregator',
    "Communication Overhead vs. Number of Nodes",
    "Number of Nodes ($n$)", "Communication Overhead (bits)", "comm_overhead_beta_epsilon.pdf"
)

# ---------------- Plot 7: Communication Overhead (dimension vs delta frac) ----------------
comm_df3 = melt_and_map(df, ['nodes', 'dimension', 'delta_frac'], list(comm_map.keys()), 'aggregator', 'comm_overhead', comm_map)
comm_df3 = comm_df3.loc[comm_df3['dimension'].isin([20,60,80])]
comm_df3 = comm_df3.rename(
        columns={"dimension": r"$ d $", "delta_frac": r"$ \nu $", "nodes": r"$ n $", "comm_overhead": "Communication Overhead"})
create_facet_plot(
    comm_df3, r"$ d $", r"$ \nu $", r"$ n $", "Communication Overhead", 'aggregator',
    "Communication Overhead vs. Number of Nodes",
    "Number of Nodes ($n$)", "Communication Overhead (bits)", "comm_overhead_dimension_delta_frac.pdf"
)


# ---------------- Plot 8: Communication Overhead (epsilon vs delta frac) ----------------
comm_df4 = melt_and_map(df, ['nodes', 'epsilon', 'delta_frac'], list(comm_map.keys()), 'aggregator', 'comm_overhead', comm_map)
comm_df4 = comm_df4.rename(
        columns={"epsilon": r"$ \varepsilon $", "delta_frac": r"$ \nu $", "nodes": r"$ n $", "comm_overhead": "Communication Overhead"})
create_facet_plot(
    comm_df4, r"$ \varepsilon $", r"$ \nu $", r"$ n $", "Communication Overhead", 'aggregator',
    "Communication Overhead vs. Number of Nodes",
    "Number of Nodes ($n$)", "Communication Overhead (bits)", "comm_overhead_epsilon_delta_frac.pdf"
)

# ---------------- Plot 9: Monitoring Function vs Iteration by R ----------------

aggregator_map = {
    'f_approx': 'Approximate Q-Digest',
    'f_event': 'Event-driven Approximate Q-Digest',
    'f_central': 'List Summarization',
    'f_central_untrimmed_mean': 'List Summarization Untrimmed'
}


def create_facet_plot_for_R(df, R_value):
    df_R = df[df['R'] == R_value].copy()
    df_R = df_R.loc[df_R['dimension'].isin([100])]
    df_R = df_R.loc[df_R['nodes'].isin([100])]
    df_R = df_R.loc[df_R['delta_frac'].isin([0.01])]
    df_R = df_R.loc[df_R['adv_std_dev'].isin([200.0])]

    value_vars = list(aggregator_map.keys())
    melted = pd.melt(df_R, id_vars=['iteration', 'beta', 'epsilon'],
                     value_vars=value_vars, var_name='aggregator', value_name='f_value')
    melted['aggregator'] = melted['aggregator'].map(aggregator_map)

    melted = melted.rename(
        columns={"beta": r"$ \beta $", "epsilon": r"$ \varepsilon $", "f_value": r"$ f(\hat{G}) $"})

    g = sns.FacetGrid(melted, row=r"$ \beta $", col= r"$ \varepsilon $", margin_titles=True, height=3)
    g.map_dataframe(sns.lineplot, x='iteration', y=r"$ f(\hat{G}) $", hue='aggregator', style="aggregator")
    g.add_legend(title='Aggregator')
    g.set_axis_labels("Iteration", "Monitoring Function Value")
    g.fig.subplots_adjust(top=0.9)
    g.set_titles(size=20)

    outfile = os.path.join(output_dir, f"monitoring_vs_iteration_R_{R_value}.pdf")
    g.savefig(outfile)
    plt.close(g.fig)
    print(f"Saved facet plot for R = {R_value} → {outfile}")

# Load the data
df = pd.read_csv("src/simulation_results_small_itr_1000.csv")
R_values = [1000]
for R_value in R_values:
    if 'R' in df.columns and R_value in df['R'].values:
        create_facet_plot_for_R(df, R_value)