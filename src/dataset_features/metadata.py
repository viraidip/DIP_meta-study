'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, "..")
from utils import load_all_mapped_reads, load_mapped_reads, load_all
from utils import SEGMENTS, DATAPATH, RESULTSPATH, ACCNUMDICT, DATASET_STRAIN_DICT, CMAP


def load_all_metadata(dfnames):
    '''
    
    '''
    dfs = list()
    for dfname in dfnames:
        file = os.path.join(DATAPATH, "metadata", f"{dfname}.csv")
        data_df = pd.read_csv(file)
        acc_nums = ACCNUMDICT[dfname].keys()
        df = data_df[data_df["Run"].isin(acc_nums)].copy()
        dfs.append(df)

    return dfs


def analyse_metadata(dfs, dfnames, mr_dfs)-> None:
    '''
    
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

    for header in ["Assay Type", "Instrument", "Organism", "Host", "system type", "LibraryLayout", "LibrarySelection", "LibrarySource", "strain", "subtype"]:
        results[header] = list()
        for df, dfname in zip(dfs, dfnames):
            if header in df.columns:
                results[header].append(" & ".join(df[header].unique()))
            else:
                print(f"{dfname}: {header} not given")
                results[header].append(np.nan)
        
    results["mapped reads"] = list()
    for mr_df in mr_dfs:
        results["mapped reads"].append(mr_df["counts"].sum())

    result_df = pd.DataFrame(results)
    pd.set_option('display.float_format', '{:.1f}'.format)
    path = os.path.join(RESULTSPATH, "metadata", "metadata.csv")
    result_df.to_csv(path, float_format="%.2f", index=False)


def mapped_reads_distribution(dfs: list, dfnames: list)-> None:
    '''
    
    '''
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    x = 0
    bar_width = 0.7
    cm = plt.get_cmap(CMAP)
    colors = [cm(1.*i/8) for i in range(8)]

    for df in dfs:
        sum_counts = df.groupby("segment")["counts"].sum().reset_index()
        total_counts = sum_counts["counts"].sum()
        sum_counts["fraction"] = sum_counts["counts"] / total_counts
        bottom = np.zeros(len(dfs))

        for i, s in enumerate(SEGMENTS):
            v = sum_counts[sum_counts['segment'] == s]['fraction'].values[0]
            axs.bar(x, v, bar_width, color=colors[i], label=s, bottom=bottom)
            bottom += v
        x += 1

    axs.set_xticks(range(len(dfnames)))
    axs.set_xticklabels(dfnames, rotation=45)
    axs.set_xlabel('Segment')
    axs.set_ylabel('Fraction')
    axs.set_title('Fraction of reads from different segments')

    handles, labels = axs.get_legend_handles_labels()
    unique_handles_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_handles_labels:
            unique_handles_labels[label] = handle

    box = axs.get_position()
    axs.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    axs.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=8)

    plt.show()


def dataset_distributions(dfs: list, dfnames: list)-> None:
    '''
    
    '''
    stats = dict({"Dataset": dfnames,
                  "Size": list(),
                  "Mean": list(),
                  "Median": list(),
                  "Std. dev.": list(),
                  "Max": list(),
                  "Mapped reads": list(),
                  "NGS/MR": list()})
    plot_data = list()
    
    for df, dfname in zip(dfs, dfnames):
        counts = df["NGS_read_count"]
        plot_data.append(counts)
        stats["Size"].append(df.shape[0])
        stats["Mean"].append(counts.mean())
        stats["Median"].append(counts.median())
        stats["Std. dev."].append(counts.std())
        stats["Max"].append(counts.max())

        # load mapped reads
        mr_df = load_mapped_reads(dfname)
        mr_sum = mr_df["counts"].sum()

        stats["Mapped reads"].append(mr_sum)
        stats["NGS/MR"].append(counts.sum() / mr_sum)

    labels = [f"{name} ({n})" for name, n in zip(dfnames, stats["Size"])]
    plt.figure(figsize=(8, 6), tight_layout=True)
    plt.boxplot(plot_data, labels=labels, vert=False, notch=True)
    plt.xscale("log")
    plt.xlabel("NGS read count (log scale)")
    plt.ylabel("Datasets")

    save_path = os.path.join(RESULTSPATH, "metadata")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "dataset_distribution.png"))
    plt.close()

    stats_df = pd.DataFrame(stats)

    stats_df.to_csv(os.path.join(save_path, "dataset_stats.csv"), float_format="%.2f", index=False)


if __name__ == "__main__":
    plt.style.use("seaborn")

    dfnames = list(DATASET_STRAIN_DICT.keys())
    meta_dfs = load_all_metadata(dfnames)
    mr_dfs = load_all_mapped_reads(dfnames)

    analyse_metadata(meta_dfs, dfnames, mr_dfs)
    mapped_reads_distribution(mr_dfs, dfnames)
    
    dfs, _ = load_all(dfnames)
    
    dataset_distributions(dfs, dfnames)
