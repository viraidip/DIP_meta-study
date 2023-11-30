'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, "..")
from utils import load_all_mapped_reads
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
    results = dict({"names": dfnames, "Reads mean": list(), "Reads sum": list()})

    results["AvgSpotLen"] = list()
    results["considered datasets"] = list()
    for df, dfname in zip(dfs, dfnames):
        results["considered datasets"].append(len(ACCNUMDICT[dfname]))

        if "AvgSpotLen" in df.columns:
            results["AvgSpotLen"].append(df["AvgSpotLen"].mean())
        else:
            print(f"{dfname}: spot length not given")
            results["AvgSpotLen"].append(np.nan)

        if "Reads" not in df.columns:
            df["Reads"] = df["Bases"] / df["AvgSpotLen"]
        results["Reads mean"].append(df["Reads"].mean())
        results["Reads sum"].append(df["Reads"].sum())

    for header in ["Assay Type", "Instrument", "Organism", "Host", "LibraryLayout", "LibrarySelection", "LibrarySource", "strain"]:
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

if __name__ == "__main__":
    plt.style.use("seaborn")

    dfnames = DATASET_STRAIN_DICT.keys()
    dfs = load_all_metadata(dfnames)
    mr_dfs = load_all_mapped_reads(dfnames)

    analyse_metadata(dfs, dfnames, mr_dfs)
    mapped_reads_distribution(mr_dfs, dfnames)
