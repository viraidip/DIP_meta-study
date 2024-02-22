'''
    Performs an analysis of the metadata of the datasets
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_all_mapped_reads, load_mapped_reads, load_all, get_dataset_names
from utils import SEGMENTS, DATAPATH, RESULTSPATH, ACCNUMDICT, CMAP, CUTOFF


def load_all_metadata(dfnames: list)-> list:
    '''
        loads the metadata file for each given dataset.
        :param dfnames: The names of the datasets

        :return: the metadata as DataFrame, in the same order as dfnames
    '''
    dfs = list()
    for dfname in dfnames:
        file = os.path.join(DATAPATH, "metadata", f"{dfname}.csv")
        data_df = pd.read_csv(file)
        acc_nums = ACCNUMDICT[dfname].keys()
        df = data_df[data_df["Run"].isin(acc_nums)].copy()
        dfs.append(df)

    return dfs


def analyse_metadata(dfs: list, dfnames: list)-> None:
    '''
        analyse the metadata of each dataset and write results into a csv.
        :param dfs: The list of DataFrames containing the metadata
        :param dfnames: The names associated with each DataFrame in `dfs`

        :return: None        
    '''
    results = dict({"names": dfnames, "Reads mean": list(), "Reads sum": list(), "AvgSpotLen": list(), "considered datasets": list()})

    for df, dfname in zip(dfs, dfnames):
        results["considered datasets"].append(len(ACCNUMDICT[dfname]))
        if "AvgSpotLen" not in df.columns:
            df["AvgSpotLen"] = df["Bases"] / df["Reads"]
        results["AvgSpotLen"].append(df["AvgSpotLen"].mean())
        if "Reads" not in df.columns:
            df["Reads"] = df["Bases"] / df["AvgSpotLen"]
        results["Reads mean"].append(df["Reads"].mean())
        results["Reads sum"].append(df["Reads"].sum())

    for header in ["system type", "LibraryLayout", "LibrarySelection", "LibrarySource", "strain", "subtype"]:
            #  in vivo human, in vivo mouse , vitro,                                  
        results[header] = list()
        for df, dfname in zip(dfs, dfnames):
            if header in df.columns:
                results[header].append(" & ".join(df[header].unique()))
            else:
                print(f"{dfname}: {header} not given")
                results[header].append(np.nan)
        
    result_df = pd.DataFrame(results)
    pd.set_option('display.float_format', '{:.1f}'.format)
    save_path = os.path.join(RESULTSPATH, "metadata")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_df.to_csv(os.path.join(save_path, "metadata.csv"), float_format="%.2f", index=False)



def dataset_distributions(dfs: list, dfnames: list)-> None:
    '''
        calcualte statistical parameters for the datasets.
        :param dfs: The list of DataFrames containing the DelVG data
        :param dfnames: The names of the datasets

        :return: None
    '''
    stats = dict({"Dataset": dfnames,
                  "Size": list(),
                  "Mean": list(),
                  "Median": list(),
                  "Std. dev.": list(),
                  "Max": list()})
    plot_data = list()
    for df, dfname in zip(dfs, dfnames):
        counts = df["NGS_read_count"]
        plot_data.append(counts)
        stats["Size"].append(df.shape[0])
        stats["Mean"].append(counts.mean())
        stats["Median"].append(counts.median())
        stats["Std. dev."].append(counts.std())
        stats["Max"].append(counts.max())
    
    labels = [f"{name} ({n})" for name, n in zip(dfnames, stats["Size"])]
    plt.figure(figsize=(8, 6), tight_layout=True)
    plt.boxplot(plot_data, labels=labels, vert=False)
    plt.xscale("log")
    plt.xlabel("NGS read count (log scale)")
    plt.ylabel("Datasets")
    save_path = os.path.join(RESULTSPATH, "metadata")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"dataset_distribution_{CUTOFF}.png"))
    plt.close()
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(save_path, f"dataset_stats_{CUTOFF}.csv"), float_format="%.2f", index=False)


if __name__ == "__main__":
    plt.style.use("seaborn")

    dfnames = get_dataset_names()
    meta_dfs = load_all_metadata(dfnames)

    analyse_metadata(meta_dfs, dfnames)

    
    dfs, _ = load_all(dfnames)
    dataset_distributions(dfs, dfnames)
