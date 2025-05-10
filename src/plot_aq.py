import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rc
import matplotlib.dates as mdates
from sympy.abc import alpha

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

def plot_diff_facet(params, csv_pattern, start_date='2013-05-01', end_date='2013-06-01'):
    """
    Plot absolute percentage error for multiple parameters using a Seaborn FacetGrid,
    with one row per parameter and shared x-axis.

    params        : List of parameter names, e.g. ['PM25','PM10','NO2','temperature']
    csv_pattern   : Pattern for CSV filenames, e.g. 'air_quality_simulation_results_{}.csv'
    start_date    : Start date for filtering (inclusive)
    end_date      : End date for filtering (inclusive)
    """
    # Collect data for all parameters
    records = []
    for p in params:
        df = pd.read_csv(csv_pattern.format(p), parse_dates=['time'])
        df = df.sort_values('time')
        # Filter date range
        mask = (df['time'] >= pd.to_datetime(start_date)) & (df['time'] <= pd.to_datetime(end_date))
        df = df.loc[mask].copy()
        # Compute absolute percentage differences
        df[r'$\textsc{Approximate-Trim}$'] = np.abs(df['f_approx'] - df['f_central']) / df['f_central'] * 100.0
        df[r'$\textsc{Event-Driven-Approximate-Trim}$']      = np.abs(df['f_event']  - df['f_central']) / df['f_central'] * 100.0
        df[r'$\textsc{List-Summarization}$ Untrimmed']    = np.abs(df['f_mean']   - df['f_central']) / df['f_central'] * 100.0
        # Melt to long form
        df_long = df.melt(
            id_vars=['time'],
            value_vars=[r'$\textsc{Event-Driven-Approximate-Trim}$', r'$\textsc{Approximate-Trim}$', r'$\textsc{List-Summarization}$ Untrimmed'],
            var_name='Method',
            value_name='Percentage Error'
        )
        if p == 'PM25':
            p = r'$ \textsc{PM}_{2.5} $'
        elif p == 'PM10':
            p = r'$ \textsc{PM}_{10} $'
        elif p == 'NO2':
            p = r'$ \textsc{NO}_{2} $'

        df_long['Parameter'] = p
        records.append(df_long)

    # Combine all parameters
    data_all = pd.concat(records, ignore_index=True)

    # Create FacetGrid: one row per parameter
    sns.set_style("white")
    g = sns.FacetGrid(
        data_all,
        row='Parameter',
        hue='Method',
        sharex=True,
        sharey=False,
        height=2.5,
        aspect=4,
        margin_titles=True,
        palette=sns.color_palette(["#1B7F79", "#FF4858", "#FFAE00"])
    )
    g.map_dataframe(sns.lineplot, x='time', y='Percentage Error', marker='.')
    g.add_legend(title='Aggregation Method')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 0.95), ncol=3, title=None, frameon=False,
    )
    g.set_axis_labels('Time ($t$)', 'Absolute Percentage Error')
    g.set_titles(row_template='{row_name}')
    xformatter = mdates.DateFormatter("%m/%d/%Y")
    g.axes[0, 0].xaxis.set_major_formatter(xformatter)
    plt.xticks(rotation=30)
    plt.subplots_adjust(top=0.9)
    plt.setp(g._legend.get_title(), fontsize=35)
    outfile = os.path.join(output_dir, f"monitoring_air_quality.pdf")
    g.savefig(outfile)
    plt.show()
    plt.close(g.fig)


def plot_comm(params, csv_pattern, start_date='2013-05-01', end_date='2013-06-01'):
    """
    Plot absolute percentage error for multiple parameters using a Seaborn FacetGrid,
    with one row per parameter and shared x-axis.

    params        : List of parameter names, e.g. ['PM25','PM10','NO2','temperature']
    csv_pattern   : Pattern for CSV filenames, e.g. 'air_quality_simulation_results_{}.csv'
    start_date    : Start date for filtering (inclusive)
    end_date      : End date for filtering (inclusive)
    """
    # Collect data for all parameters
    records = []
    for p in params:
        df = pd.read_csv(csv_pattern.format(p), parse_dates=['time'])
        df = df.sort_values('time')
        # Filter date range
        mask = (df['time'] >= pd.to_datetime(start_date)) & (df['time'] <= pd.to_datetime(end_date))
        df = df.loc[mask].copy()
        # Compute absolute percentage differences
        df[r'$\textsc{Event-Driven-Approximate-Trim}$']      = df['comm_bits_event']
        df[r'$\textsc{Approximate-Trim}$'] = df['comm_bits_periodic']
        df[r'$\textsc{List-Summarization}$']    = df['comm_bits_central']
        # Melt to long form
        df_long = df.melt(
            id_vars=['time'],
            value_vars=[r'$\textsc{Event-Driven-Approximate-Trim}$', r'$\textsc{Approximate-Trim}$', r'$\textsc{List-Summarization}$'],
            var_name='Method',
            value_name='Communication Overhead (bits)'
        )
        if p == 'PM25':
            p = r'$ \textsc{PM}_{2.5} $'
        elif p == 'PM10':
            p = r'$ \textsc{PM}_{10} $'
        elif p == 'NO2':
            p = r'$ \textsc{NO}_{2} $'

        df_long['Parameter'] = p
        records.append(df_long)

    # Combine all parameters
    data_all = pd.concat(records, ignore_index=True)

    # Create FacetGrid: one row per parameter
    sns.set_style("white")
    g = sns.FacetGrid(
        data_all,
        row='Parameter',
        hue='Method',
        sharex=True,
        sharey=False,
        height=2.5,
        aspect=4,
        margin_titles=True,
        palette=sns.color_palette(["#1B7F79", "#FF4858", "#FFAE00"])
    )

    g.map_dataframe(sns.lineplot, x='time', y='Communication Overhead (bits)', marker='.')
    g.add_legend(title='Aggregation Method')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 0.95), ncol=3, title=None, frameon=False,
    )
    g.set_axis_labels('Time ($t$)', 'Communication Overhead (bits)')
    g.set_titles(row_template='{row_name}')
    xformatter = mdates.DateFormatter("%m/%d/%Y")
    g.axes[0, 0].xaxis.set_major_formatter(xformatter)
    plt.xticks(rotation=30)
    plt.subplots_adjust(top=0.9)
    plt.setp(g._legend.get_title(), fontsize=35)
    outfile = os.path.join(output_dir, f"monitoring_air_quality_comm.pdf")
    g.savefig(outfile)
    plt.show()
    plt.close(g.fig)


# Example usage:
plot_diff_facet(
    ['PM25','PM10','NO2'],
    'air_quality_simulation_results_{}.csv'
)

plot_comm(
    ['PM25','PM10','NO2'],
    'air_quality_simulation_results_{}.csv'
)