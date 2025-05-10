import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rc
import matplotlib.dates as mdates

dpi = 300
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (8, 6)

output_dir = "../plots"
os.makedirs(output_dir, exist_ok=True)

def plot_mawi_error(csv_file, start_time=None, end_time=None):
    """
    Plot absolute percentage error (Periodic, Event‐Driven, Untrimmed)
    relative to the central trimmed mean, over time.
    """
    df = pd.read_csv(csv_file, parse_dates=['epoch'])
    df = df.sort_values('epoch')
    if start_time:
        df = df[df['epoch'] >= pd.to_datetime(start_time)]
    if end_time:
        df = df[df['epoch'] <= pd.to_datetime(end_time)]

    # compute percentage errors
    df[r'$\textsc{Approximate-Trim}$'] = df['f_approx']
    df[r'$\textsc{Event-Driven-Approximate-Trim}$'] = df['f_event']
    df[r'$\textsc{List-Summarization}$ Untrimmed'] = df['f_mean']
    df[r'$\textsc{List-Summarization}$'] = df['f_central']

    # melt to long form
    df_long = df.melt(
        id_vars=['epoch'],
        value_vars=[r'$\textsc{List-Summarization}$', r'$\textsc{Approximate-Trim}$',r'$\textsc{Event-Driven-Approximate-Trim}$',r'$\textsc{List-Summarization}$ Untrimmed'],
        var_name='Method',
        value_name='Percentage Error'
    )

    # single‐row facet (or no facet at all)
    sns.set_style("white")
    g = sns.FacetGrid(
        df_long,
        hue='Method',
        height=4, aspect=2.5,
        palette='Spectral'
    )
    g.map_dataframe(sns.lineplot, x='epoch', y='Percentage Error', marker='.')
    g.add_legend(title='')
    g.set_axis_labels('Time', 'Absolute % Error')
    g.set_titles('')
    g.fig.subplots_adjust(top=0.85)
    xformatter = mdates.DateFormatter("%H:%M")
    g.axes[0, 0].xaxis.set_major_formatter(xformatter)
    plt.xticks(rotation=30)

    out = os.path.join(output_dir, "mawi_pct_error.pdf")
    g.savefig(out, dpi=dpi)
    plt.close(g.fig)


def plot_mawi_comm(csv_file, start_time=None, end_time=None):
    """
    Plot communication overhead (Periodic, Event-Driven, List) in bits, over time.
    """
    df = pd.read_csv(csv_file, parse_dates=['epoch'])
    df = df.sort_values('epoch')
    if start_time:
        df = df[df['epoch'] >= pd.to_datetime(start_time)]
    if end_time:
        df = df[df['epoch'] <= pd.to_datetime(end_time)]

    df['Periodic Q-Digest']        = df['comm_bits_periodic']
    df['Event-Driven Q-Digest']    = df['comm_bits_event']
    df['List-Summarization']       = df['comm_bits_central']

    df_long = df.melt(
        id_vars=['epoch'],
        value_vars=['Periodic Q-Digest','Event-Driven Q-Digest','List-Summarization'],
        var_name='Method',
        value_name='Bits'
    )

    sns.set_style("white")
    g = sns.FacetGrid(
        df_long,
        hue='Method',
        height=4, aspect=2.5,
        palette=["#1B7F79","#FF4858","#FFAE00"]
    )
    g.map_dataframe(sns.lineplot, x='epoch', y='Bits', marker='.')
    g.add_legend(title='')
    g.set_axis_labels('Time', 'Communication (bits)')
    g.set_titles('')
    g.fig.subplots_adjust(top=0.85)

    xformatter = mdates.DateFormatter("%H:%M:%S")
    g.axes[0, 0].xaxis.set_major_formatter(xformatter)
    plt.xticks(rotation=30)

    out = os.path.join(output_dir, "mawi_comm_overhead.pdf")
    g.savefig(out, dpi=dpi)
    plt.close(g.fig)


# Example usage:

plot_mawi_error("mawi_simulation_results_mid.csv")
plot_mawi_comm ("mawi_simulation_results_mid.csv")