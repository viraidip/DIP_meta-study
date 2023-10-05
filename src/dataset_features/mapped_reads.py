'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats import chi2_contingency

sys.path.insert(0, "..")
from utils import load_mapped_reads
from utils import SEGMENTS, RESULTSPATH



def mapped_reads_distribution(dfs: list, dfnames: list)-> None:
    '''
    
    '''
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    x = 0
    bar_width = 0.7
    cm = plt.get_cmap('tab10')
    colors = [cm(1.*i/10) for i in range(10)]

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
    dfs = list()
    dfnames = list()
    
    
    experiment = "Alnaji2021"
    df = load_mapped_reads(experiment, "Alnaji2021")
   # for t in df["Time"].unique():
    #    df_t = df[df["Time"] == t].copy()
    dfs.append(df)
    dfnames.append(f"Alnaji2021")

    experiment = "Pelz2021"
    df = load_mapped_reads(experiment, "Pelz2021")
    dfs.append(df)
    dfnames.append(f"Pelz2021")

    experiment = "Wang2023"
    df = load_mapped_reads(experiment, "Wang2023")
    dfs.append(df)
    dfnames.append(f"Wang2023")

    experiment = "Wang2020"
    df = load_mapped_reads(experiment, "Wang2020")
    dfs.append(df)
    dfnames.append(f"Wang2020")

    experiment = "Kupke2020"
    df = load_mapped_reads(experiment, "Kupke2020")
    df_t = df[df["Time"] == "post"].copy()
    dfs.append(df_t)
    dfnames.append(f"Kupke2020")

    experiment = "Alnaji2019"
    for strain, p in [("Cal07", "6"), ("NC", "1"), ("Perth", "4") , ("BLEE", "7")]:
        df = load_mapped_reads(experiment, strain)
        df = df[df["Passage"] == p].copy()
        dfs.append(df)
        dfnames.append(f"Alnaji2019 {strain}")

    experiment = "Penn2022"
    df = load_mapped_reads(experiment, "Penn2022")
    dfs.append(df)
    dfnames.append(f"Penn2022")

    experiment = "Lui2019"
    df = load_mapped_reads(experiment, "Lui2019")
    dfs.append(df)
    dfnames.append(f"Lui2019")

    experiment = "Mendes2021"
    df = load_mapped_reads(experiment, "Mendes2021")
    dfs.append(df)
    dfnames.append(f"Mendes2021")

    mapped_reads_distribution(dfs, dfnames)

