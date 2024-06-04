'''

'''
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import RESULTSPATH, NUCLEOTIDES, DATASET_STRAIN_DICT, SEGMENTS, CMAP
from utils import load_all, get_dataset_names, create_nucleotide_ratio_matrix, get_sequence


def nucleotide_enrichment_overview_expected(df, exp_df):
    fig, axs = plt.subplots(figsize=(10, 3), tight_layout=True)
    colors = ["darkred", "firebrick", "indianred", "lightcoral"]
    shift = -0.2
    bars = list()
    for df in [df, exp_df]:
        m = create_nucleotide_ratio_matrix(df, "seq_around_deletion_junction")
        bottom = np.zeros(len(m.index))
        for i, c in enumerate(m.columns):
            b = axs.bar(m.index+shift, m[c], width=0.3, label=c, color=colors[i], bottom=bottom, edgecolor="black")#, alpha=1-i*0.2)
            if c == "C":
                bars.append(b)
            bottom += m[c]
        shift = 0.2
        colors = ["navy", "royalblue", "cornflowerblue", "lightblue"]

    quarter = len(m.index) // 4
    axs.set_xticks(m.index, list(range(1,11))+list(range(1,11)))
    xlabels = axs.get_xticklabels()
    for x_idx, xlabel in enumerate(xlabels):
        if x_idx < quarter or x_idx >= quarter * 3:
            xlabel.set_color("black")
            xlabel.set_fontweight("bold")
        else:
            xlabel.set_color("grey")
    
    plt.axvline(x=10.5, color="grey", linewidth=4)
    pos = 0
    for i, n in enumerate(NUCLEOTIDES):
        new_pos = m.iloc[0, i]
        axs.text(0, pos+new_pos/2, n, color="black", fontweight="bold", fontsize=20, ha="center", va="center")
        pos += new_pos

    axs.set_xlabel("Nucleotide position")
    axs.set_ylabel("Relative occurrence")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=2, handles=bars, labels=["observed", "expected"])
          
    save_path = os.path.join(RESULTSPATH, "additional_analyses")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "exp_nuc_occ.png"))
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

    print(f"{max(a_occ)*100:.1f}")
    print(f"{min(a_occ)*100:.1f}")


if __name__ == "__main__":
    RESULTSPATH = os.path.dirname(RESULTSPATH)
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    dfnames = get_dataset_names(cutoff=40)
    dfs, _ = load_all(dfnames)
    analyze_adenin_distribution_datasets(dfs, dfnames)

    df, exp_df = load_all(["Alnaji2021"], True)
    nucleotide_enrichment_overview_expected(df[0], exp_df[0])
    