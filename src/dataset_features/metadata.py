'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, "..")
from utils import load_mapped_reads
from utils import SEGMENTS, DATAPATH, RESULTSPATH, CMAP, ACCNUMDICT


def analyse_metadata(dfnames)-> None:
    '''
    
    '''
    avgspotlenl = list()
    readsl = list()
    reads_suml = list()
    for name in dfnames:
        dir = os.path.join(DATAPATH, "metadata", f"{name}.csv")
        data_df = pd.read_csv(dir)

        if name == "Alnaji2019":
            acc_nums = list()
            for strain in ["Cal07", "NC", "Perth", "BLEE"]:
                acc_nums.extend(ACCNUMDICT[f"Alnaji2019_{strain}"].keys())
        else:
            acc_nums = ACCNUMDICT[name].keys()
        
        df = data_df[data_df["Run"].isin(acc_nums)].copy()

        
        if "AvgSpotLen" in df.columns:
            avgspotlen = df["AvgSpotLen"].mean()
        if "Reads" not in df.columns:
            df["Reads"] = df["Bases"] / df["AvgSpotLen"]
        reads = df["Reads"].mean()
        reads_sum = df["Reads"].sum()

        print(name)
        print(avgspotlen)
        print(reads)

        avgspotlenl.append(avgspotlen)
        readsl.append(reads)
        reads_suml.append(reads_sum)

    result_df = pd.DataFrame({"name": dfnames, "avgspotlen": avgspotlenl, "reads": readsl, "reads_sum": reads_suml})
    pd.set_option('display.float_format', '{:.1f}'.format)
    print(result_df)


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
    axs.set_title('Fraction of Different Segments')

    handles, labels = axs.get_legend_handles_labels()
    unique_handles_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_handles_labels:
            unique_handles_labels[label] = handle

    axs.legend(unique_handles_labels.values(), unique_handles_labels.keys())

    plt.show()


if __name__ == "__main__":
    plt.style.use("seaborn")
    dfnames = list()
    
    dfnames.append(f"Alnaji2021")
    dfnames.append(f"Pelz2021")
    dfnames.append(f"Wang2023")
    dfnames.append(f"Wang2020")
    dfnames.append(f"Kupke2020")
    dfnames.append(f"Alnaji2019")
    dfnames.append(f"Penn2022")
    dfnames.append(f"Lui2019")
    dfnames.append(f"Mendes2021")

    analyse_metadata(dfnames)

