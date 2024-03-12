'''
    Does a linear and exponential regression for data from Schwartz 2016 and 
    Alnaji 2019. Data is normalized by sum of y values for all data sets.
    Expected value is calculated by dividing length of each segment with sum of
    the length of all segements.

    Also creates a model for all three IAV strains together.
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import RESULTSPATH, NUCLEOTIDES
from utils import load_all, get_dataset_names, create_nucleotide_ratio_matrix


def nucleotide_enrichment_overview(dfs):
    fig, axs = plt.subplots(figsize=(10, 3), tight_layout=True)
    colors = ["green", "orange", "blue", "red"]
    for df in dfs:
        probability_matrix = create_nucleotide_ratio_matrix(df, "seq_around_deletion_junction")
        if "overall_m" in locals():
            overall_m += probability_matrix
        else:
            overall_m = probability_matrix

    norm_m = overall_m/len(dfs)
    bottom = np.zeros(len(norm_m.index))
    for i, c in enumerate(norm_m.columns):
        axs.bar(norm_m.index, norm_m[c], label=c, color=colors[i], bottom=bottom)
        bottom += norm_m[c]

    quarter = len(norm_m.index) // 4
    axs.set_xticks(norm_m.index, list(range(1,11))+list(range(1,11)))
    xlabels = axs.get_xticklabels()
    for x_idx, xlabel in enumerate(xlabels):
        if x_idx < quarter or x_idx >= quarter * 3:
            xlabel.set_color("black")
            xlabel.set_fontweight("bold")
        else:
            xlabel.set_color("grey")
    
    pos = 0
    for i, n in enumerate(NUCLEOTIDES):
        new_pos = norm_m.iloc[0, i]
        axs.text(0, pos+new_pos/2, n, color=colors[i], fontweight="bold", fontsize=20, ha="center", va="center")
        pos += new_pos

    axs.set_xlabel("nucleotide position")
    axs.set_ylabel("relative occurrence")
          
    #fig.subplots_adjust(top=0.9)
    save_path = os.path.join(RESULTSPATH, "additional_analyses")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "overall_nuc_occ.png"))
    plt.close()


if __name__ == "__main__":
    RESULTSPATH = os.path.dirname(RESULTSPATH)
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    dfnames = get_dataset_names(cutoff=50)
    dfs, _ = load_all(dfnames)
    nucleotide_enrichment_overview(dfs)
