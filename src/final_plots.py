import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

def melt_and_map(df, id_vars, value_vars, var_name, value_name, map_dict):
    melted = pd.melt(df, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    melted[var_name] = melted[var_name].map(map_dict)
    return melted

def create_facet_plot(df, row, col, x, y, hue, title, x_label, y_label, filename):
    g = sns.FacetGrid(df, row=row, col=col, margin_titles=True, height=4, aspect=1.2)
    g.map_dataframe(sns.lineplot, x=x, y=y, hue=hue, marker='o')
    g.add_legend(title=hue)
    g.set_axis_labels(x_label, y_label)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title, fontsize=14)
    print(f'✅  {title} → {filename}')
    g.savefig(os.path.join(output_dir, filename))
    plt.close(g.fig)

# Load the data
df = pd.read_csv("simulation_results.csv")

# ---------------- Plot 1: Error vs Nodes (row=beta, col=epsilon) ----------------
error_map = {'abs_error_approx': 'Periodic', 'abs_error_event': 'Event-driven'}
error_df = melt_and_map(df, ['nodes', 'beta', 'epsilon'], ['abs_error_approx', 'abs_error_event'], 'aggregator', 'abs_error', error_map)
create_facet_plot(
    error_df, 'beta', 'epsilon', 'nodes', 'abs_error', 'aggregator',
    "Absolute Error vs. Number of Nodes (β vs ε)",
    "Number of Nodes", "Absolute Error", "error_beta_epsilon.pdf"
)

# ---------------- Plot 2: Error vs Nodes (row=epsilon, col=beta) ----------------
create_facet_plot(
    error_df, 'epsilon', 'beta', 'nodes', 'abs_error', 'aggregator',
    "Absolute Error vs. Number of Nodes (ε vs β)",
    "Number of Nodes", "Absolute Error", "error_epsilon_beta.pdf"
)

# ---------------- Plot 3: Error vs Nodes (row=R, col=beta) ----------------
if 'R' in df.columns:
    error_df_with_R = melt_and_map(df, ['nodes', 'beta', 'epsilon', 'R'], ['abs_error_approx', 'abs_error_event'], 'aggregator', 'abs_error', error_map)
    create_facet_plot(
        error_df_with_R, 'R', 'beta', 'nodes', 'abs_error', 'aggregator',
        "Absolute Error vs. Number of Nodes (R vs β)",
        "Number of Nodes", "Absolute Error", "error_R_beta.pdf"
    )

# ---------------- Plot 4: Energy per Node ----------------
energy_map = {
    'avg_energy_per_node_approx': 'Periodic',
    'avg_energy_per_node_event': 'Event-driven',
    'avg_energy_per_node_central': 'Central'
}
energy_df = melt_and_map(df, ['nodes', 'dimension', 'epsilon'], list(energy_map.keys()), 'aggregator', 'avg_energy_per_node', energy_map)
create_facet_plot(
    energy_df, 'dimension', 'epsilon', 'nodes', 'avg_energy_per_node', 'aggregator',
    "Average Energy per Node vs. Number of Nodes\n(Facets: Rows=Dimension, Columns=ε)",
    "Number of Nodes", "Average Energy per Node (Joules)", "energy_per_node.pdf"
)

# ---------------- Plot 5: Communication Overhead (dimension vs epsilon) ----------------
comm_map = {
    'comm_overhead_approx': 'Periodic',
    'comm_overhead_event': 'Event-driven',
    'comm_overhead_central': 'Central'
}
comm_df1 = melt_and_map(df, ['nodes', 'dimension', 'epsilon'], list(comm_map.keys()), 'aggregator', 'comm_overhead', comm_map)
create_facet_plot(
    comm_df1, 'dimension', 'epsilon', 'nodes', 'comm_overhead', 'aggregator',
    "Communication Overhead vs. Number of Nodes\n(Facets: Rows=Dimension, Columns=ε)",
    "Number of Nodes", "Communication Overhead (bits)", "comm_overhead_dimension_epsilon.pdf"
)

# ---------------- Plot 6: Communication Overhead (beta vs epsilon) ----------------
comm_df2 = melt_and_map(df, ['nodes', 'beta', 'epsilon'], list(comm_map.keys()), 'aggregator', 'comm_overhead', comm_map)
create_facet_plot(
    comm_df2, 'beta', 'epsilon', 'nodes', 'comm_overhead', 'aggregator',
    "Communication Overhead vs. Number of Nodes\n(Facets: Rows=β, Columns=ε)",
    "Number of Nodes", "Communication Overhead (bits)", "comm_overhead_beta_epsilon.pdf"
)

# ---------------- Plot 7: Communication Overhead (dimension vs delta frac) ----------------
comm_df3 = melt_and_map(df, ['nodes', 'dimension', 'delta_frac'], list(comm_map.keys()), 'aggregator', 'comm_overhead', comm_map)
create_facet_plot(
    comm_df3, 'dimension', 'delta_frac', 'nodes', 'comm_overhead', 'aggregator',
    "Communication Overhead vs. Number of Nodes\n(Facets: Rows=d, Columns=d_frac)",
    "Number of Nodes", "Communication Overhead (bits)", "comm_overhead_dimension_delta_frac.pdf"
)


# ---------------- Plot 7: Communication Overhead (epsilon vs delta frac) ----------------
comm_df4 = melt_and_map(df, ['nodes', 'epsilon', 'delta_frac'], list(comm_map.keys()), 'aggregator', 'comm_overhead', comm_map)
create_facet_plot(
    comm_df4, 'epsilon', 'delta_frac', 'nodes', 'comm_overhead', 'aggregator',
    "Communication Overhead vs. Number of Nodes\n(Facets: Rows=ε, Columns=d_frac)",
    "Number of Nodes", "Communication Overhead (bits)", "comm_overhead_epsilon_delta_frac.pdf"
)



# ---------------- Plot 8: FUNCTION VALUES (epsilon vs delta frac) ----------------
comm_df4 = melt_and_map(df, ['nodes', 'epsilon', 'delta_frac'], list(comm_map.keys()), 'aggregator', 'comm_overhead', comm_map)
create_facet_plot(
    comm_df4, 'epsilon', 'delta_frac', 'nodes', 'comm_overhead', 'aggregator',
    "Communication Overhead vs. Number of Nodes\n(Facets: Rows=ε, Columns=d_frac)",
    "Number of Nodes", "Communication Overhead (bits)", "comm_overhead_epsilon_delta_frac.pdf"
)
# ---------------- Plot 9: Monitoring Function vs Iteration by R ----------------

aggregator_map = {
    'f_approx': 'Periodic',
    'f_event': 'Event-driven',
    'f_central': 'Central',
    'f_central_untrimmed_mean': 'Central Untrimmed'
}


def create_facet_plot_for_R(df, R_value):
    df_R = df[df['R'] == R_value].copy()
    value_vars = list(aggregator_map.keys())
    melted = pd.melt(df_R, id_vars=['iteration', 'beta', 'epsilon'],
                     value_vars=value_vars, var_name='aggregator', value_name='f_value')
    melted['aggregator'] = melted['aggregator'].map(aggregator_map)

    g = sns.FacetGrid(melted, row='beta', col='epsilon', margin_titles=True, height=4, aspect=1.2, sharey=False)
    g.map_dataframe(sns.lineplot, x='iteration', y='f_value', hue='aggregator', marker='o')
    g.add_legend(title='Aggregator')
    g.set_axis_labels("Iteration", "Monitoring Function Value")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Monitoring Function Value vs. Iteration for R = {R_value}", fontsize=14)

    outfile = os.path.join(output_dir, f"monitoring_vs_iteration_R_{R_value}.pdf")
    g.savefig(outfile)
    plt.close(g.fig)
    print(f"✅ Saved facet plot for R = {R_value} → {outfile}")


R_values = [100, 500, 1000, 5000, 10000]
for R_value in R_values:
    if 'R' in df.columns and R_value in df['R'].values:
        create_facet_plot_for_R(df, R_value)
