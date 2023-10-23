'''

'''
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

sys.path.insert(0, "..")
from utils import load_alnaji2019, load_pelz2021, load_alnaji2021
from utils import get_sequence, create_nucleotide_ratio_matrix, count_direct_repeats_overall, include_correction, preprocess
from utils import SEGMENTS, RESULTSPATH, NUCLEOTIDES


def analyse_nucleotide_enrichment_over_time(dfs, x):
    '''
    
    '''
    relevant_indices = [i for i in range(1, 21)]

    ms = list()
    indices = list()
    nucs = list()
    for nuc in NUCLEOTIDES.keys():
        y = dict({i: list() for i in relevant_indices})

        for df in dfs:
            probability_matrix = create_nucleotide_ratio_matrix(df, "seq_around_deletion_junction")
            for i in relevant_indices:
                y[i].append(probability_matrix.loc[i,nuc] * 100)

        for k, v in y.items():
            # do linear regression with v
            model = LinearRegression().fit(x.reshape((-1, 1)), v)
            m = model.coef_[0]
            ms.append(m)
            indices.append(k)
            nucs.append(nuc)

    df = pd.DataFrame({"slope": ms, "position": indices, "nucleotide": nucs})

    return df


def plot_nucleotide_enrichment_over_time(dfs: list, relevant_indices: list, x: list, xlabel: str, fname: str)-> None:
    '''
        Plot heatmaps of nucleotide ratios around deletion junctions.

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.

        :return: None
    '''
    fig, axs = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)
    axs = axs.flatten()

    for i, nuc in enumerate(NUCLEOTIDES.keys()):
        y = dict({j: list() for j in relevant_indices})

        for df in dfs:
            probability_matrix = create_nucleotide_ratio_matrix(df, "seq_around_deletion_junction")
            for j in relevant_indices:
                y[j].append(probability_matrix.loc[j,nuc])

        
        for k, v in y.items():
            # map relevant index to start/end and exact position
            if k <= 10:
                pos = "Start"
            else:
                pos = "End"
                k = k - 10
            label = f"{pos} pos {k}"

#            res, pval = stats.normaltest(v)
 #           if pval < 0.005:
  #              print(f"{nuc} {label} {pval}")
            axs[i].plot(x, v, label=label)

        axs[i].set_title(f"{nuc} nucleotide ratios around deletion junction")
        axs[i].set_ylabel("relative occurrence of nucleotide")
        axs[i].set_xlabel(xlabel)

    axs[i].legend(loc="upper center", bbox_to_anchor=(0.0, -0.1), fancybox=True, shadow=True, ncol=8)

    save_path = os.path.join(RESULTSPATH, "timeseries", f"nuc_occ_{fname}")
    plt.savefig(save_path)


def plot_direct_repeats_over_time(dfs: list, x: list, xlabel: str, fname: str)-> None:
    '''Plot heatmaps of nucleotide ratios around deletion junctions.

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        col (str, optional): The column name in the DataFrames that contains the sequence segments of interest. 
                             Default is "seq_around_deletion_junction".
        height (float, optional): The height of the figure in inches. Default is 20.
        width (float, optional): The width of the figure in inches. Default is 16.
        nucleotides (list of str, optional): The nucleotides to be plotted. Default is ["A", "C", "G", "T"].

    :return: None

    '''
    fig, axs = plt.subplots(figsize=(len(dfs)/2, 6), nrows=1, ncols=1)
   
    y = dict({i: list() for i in range(6)})
    for df in dfs:
        all = 0
        co = dict()
        for s in SEGMENTS:
            df_s = df[df["Segment"] == s]
            if len(df_s) == 0:
                continue
                
            seq = get_sequence(df_s["Strain"].unique()[0], s)
            counts, _ = count_direct_repeats_overall(df_s, seq)
            counts = include_correction(counts)
            for k, v in counts.items():
                if k in co:
                    co[k] += v
                else:
                    co[k] = v
                all += v

        for i in range(6):
            y[i].append(co[i] / all)

    bar_width = 0.4
    bottom = np.zeros(len(dfs))

    for i in range(6):
        axs.bar(x, y[i], bar_width, label=i, bottom=bottom)
        bottom += y[i]

    axs.set_ylabel("relative occurrence of direct repeat length")
    axs.set_xlabel(xlabel)
    fig.tight_layout()

    box = axs.get_position()
    axs.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    handles, labels = axs.get_legend_handles_labels()
    labels[-1] = f"> {labels[-2]}"
    axs.legend(handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=8)

    save_path = os.path.join(RESULTSPATH, "timeseries", f"dir_rep_{fname}")
    plt.savefig(save_path)
    plt.close()


def plot_segment_distribution_over_time(dfs, x, xlabel, fname)-> None:
    '''Plot heatmaps of nucleotide ratios around deletion junctions.

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        col (str, optional): The column name in the DataFrames that contains the sequence segments of interest. 
                             Default is "seq_around_deletion_junction".
        height (float, optional): The height of the figure in inches. Default is 20.
        width (float, optional): The width of the figure in inches. Default is 16.
        nucleotides (list of str, optional): The nucleotides to be plotted. Default is ["A", "C", "G", "T"].

    :return: None
    '''
    fig, axs = plt.subplots(figsize=(len(dfs)/2, 6), nrows=1, ncols=1)
    cm = plt.get_cmap("viridis")
    colors = [cm(1.*i/len(SEGMENTS)) for i in range(len(SEGMENTS))]

    y = dict({s: list() for s in SEGMENTS})
    for df in dfs:
        fractions = df["Segment"].value_counts() / len(df)
        for s in SEGMENTS:
            if s not in fractions:
                y[s].append(0.0)
            else:
                y[s].append(fractions[s])

    bar_width = 0.4
    bottom = np.zeros(len(dfs))

    for i, s in enumerate(SEGMENTS):
        axs.bar(x, y[s], bar_width, color=colors[i], label=s, bottom=bottom)
        bottom += y[s]

    axs.set_ylabel("relative occurrence of segment")
    axs.set_xlabel(xlabel)
    fig.tight_layout()

    box = axs.get_position()
    axs.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    axs.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=8)

    save_path = os.path.join(RESULTSPATH, "timeseries", f"segment_dist_{fname}")
    plt.savefig(save_path)
    plt.close()


def length_over_time(dfs, x, fname):
    '''
    
    '''
    fig, axs = plt.subplots(figsize=(len(dfs)/2, 6), nrows=1, ncols=1)
    cm = plt.get_cmap("viridis")
    colors = [cm(1.*i/len(SEGMENTS)) for i in range(len(SEGMENTS))]
    
    all = list()
    long = list()
    for df in dfs:
        df["DI_length"] = df["End"] - df["Start"]
        filter_df = df[df["DI_length"] < 600]
        all.append(df.shape[0])
        long.append(filter_df.shape[0])

    axs.plot(x, all)
    axs.plot(x, long)

    plt.show()
    plt.close()

    """
    save_path = os.path.join(RESULTSPATH, "timeseries", f"length_{fname}")
    plt.savefig(save_path)
    plt.close()
    """


def run_analyses_for_dataset(dfs, dfnames, x, xlabel, fname):
    '''
    
    '''
    slope_df = analyse_nucleotide_enrichment_over_time(dfs, x)  
    relevant_indices = slope_df.sort_values(by="slope").head(2)["position"].unique().tolist()
    relevant_indices = list(set(relevant_indices))
    plot_nucleotide_enrichment_over_time(dfs, relevant_indices, x, xlabel, fname)

    plot_direct_repeats_over_time(dfs, x, xlabel, fname)
    plot_segment_distribution_over_time(dfs, x, xlabel, fname)


def compare_counts(counts_all, xs_normalised_all, labels_all):
    '''
    
    '''
    fig, axs = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
    cm = plt.get_cmap("viridis")
    colors = [cm(1.*i/len(SEGMENTS)) for i in range(len(SEGMENTS))]
    
    for c, x, l in zip(counts_all, xs_normalised_all, labels_all):
        axs.plot(x, c, label=l)

    axs.set_xlabel("Time [h]")
    axs.set_ylabel("# DI RNAs")
    plt.legend()

    plt.show()
    plt.close()
    """
    save_path = os.path.join(RESULTSPATH, "timeseries", f"length_{fname}")
    plt.savefig(save_path)
    plt.close()
    """


if __name__ == "__main__":
    cutoff = 5
    plt.style.use("seaborn")

    counts_all = list()
    xs_normalised_all = list()
    labels_all = list()

### Alnaji2019 ###
    dfs = list()
    dfnames = list()
    counts = list()
    x = np.array([1, 3, 6])
    df = load_alnaji2019("Cal07_time")
    for t in ["1", "3", "6"]:
        df_t = df[df["Passage"] == t].copy()
        counts.append(df_t.shape[0])
        dfs.append(preprocess("Cal07", df_t, cutoff))
        dfnames.append(t)
        xlabel = "# passage"
        fname = "Alanji2019_Cal07"

    counts_all.append(counts)
    xs_normalised_all.append(x*24) # one passage is assumed to be 1 day
    labels_all.append(fname)

    run_analyses_for_dataset(dfs, dfnames, x, xlabel, fname)


### Alnaji2021 ###
    df = load_alnaji2021()
    x = np.array([3, 6, 24])
    xlabel = "time [h]"
    for rep in ["A", "B", "C"]:
        dfs = list()
        dfnames = list()
        counts = list()
        fname = f"Alnaji2021_{rep}"
        df_r = df[df["Replicate"] == rep].copy()
        for t in ["3hpi", "6hpi", "24hpi"]:
            df_t = df_r[df_r["Time"] == t].copy()
            counts.append(df_t.shape[0])
            dfs.append(preprocess("PR8", df_t, cutoff))
            dfnames.append(t)

        counts_all.append(counts)
        xs_normalised_all.append(x)
        labels_all.append(fname)

        run_analyses_for_dataset(dfs, dfnames, x, xlabel, fname)

    
### Pelz2021 ###
    dfs = list()
    dfnames = list()
    counts = list()
    x = np.array([0.5,1,1.4,3.5,4,4.5,5,5.5,8,9,9.4,12.4,13,13.5,16,17,17.5,18,19.5,20,20.4,21])
    df = load_pelz2021()
    for t in ["0.5dpi","1dpi","1.4dpi","3.5dpi","4dpi","4.5dpi","5dpi","5.5dpi","8dpi","9dpi","9.4dpi","12.4dpi","13dpi","13.5dpi","16dpi","17dpi","17.5dpi","18dpi","19.5dpi","20dpi","20.4dpi","21dpi"]:
        df_t = df[df["Time"] == t].copy()
        counts.append(df_t.shape[0])
        dfs.append(preprocess("PR8", df_t, cutoff))
        dfnames.append(t)
        xlabel = "time [d]"
        fname = "Pelz2021"

    counts_all.append(counts)
    xs_normalised_all.append(x*24)
    labels_all.append(fname)

    run_analyses_for_dataset(dfs, dfnames, x, xlabel, fname)
    


### overall comparision ###
    compare_counts(counts_all, xs_normalised_all, labels_all)