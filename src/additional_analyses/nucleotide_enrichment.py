'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import RESULTSPATH, NUCLEOTIDES, DATASET_STRAIN_DICT, SEGMENTS
from utils import load_all, get_dataset_names, create_nucleotide_ratio_matrix, get_sequence


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

    axs.set_xlabel("Nucleotide position")
    axs.set_ylabel("Relative occurrence")
          
    save_path = os.path.join(RESULTSPATH, "additional_analyses")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "overall_nuc_occ.png"))
    plt.close()


def analyze_adenin_distribution_datasets(dfs: list, dfnames: list):
    a_fracs = dict()
    a_max = ("strain", "segment", 0)
    a_min = ("strain", "segment", 1)
    for df, dfname in zip(dfs, dfnames):
        st = DATASET_STRAIN_DICT[dfname]
        a_fracs[st] = list()
        for seg in SEGMENTS:
            df_s = df.loc[df["Segment"] == seg]
            if len(df_s) == 0:
                continue
            seq = get_sequence(st, seg)
            start = int(df_s["Start"].mean())
            end = int(df_s["End"].mean())
            s = (max(start-200, 50), start+200)
            e = (end-200, min(end+200, len(seq)-50))
            
            # skip if there is no range given this would lead to oversampling of a single position
            if s[0] == s[1] or e[0] == e[1]:
                continue
            # positions are overlapping
            if s[1] > e[0]:
                continue

            seq = seq[s[0]:s[1]] + seq[e[0]:e[1]]

            a_counts = seq.count("A")
            assert a_counts != 0, f"No Adenines found in {st} {seg}: {seq}"
            
            a_frac = a_counts/len(seq)
            if a_frac > a_max[2]:
                a_max = (st, seg, a_frac)
            if a_frac < a_min[2]:
                a_min = (st, seg, a_frac)

            a_fracs[st].append(a_frac)

    print(f"Max. occurence of a: {a_max[0]}\t{a_max[1]}\t{a_max[2]}")
    print(f"Min. occurence of a: {a_min[0]}\t{a_min[1]}\t{a_min[2]}")

    a_occ = list()
    for st, values in a_fracs.items():
        for v in values:
            a_occ.append(v*v)    

    print(max(a_occ))
    print(min(a_occ))


if __name__ == "__main__":
    RESULTSPATH = os.path.dirname(RESULTSPATH)
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    dfnames = get_dataset_names(cutoff=40)
    dfs, _ = load_all(dfnames)
    nucleotide_enrichment_overview(dfs)

    analyze_adenin_distribution_datasets(dfs, dfnames)
    