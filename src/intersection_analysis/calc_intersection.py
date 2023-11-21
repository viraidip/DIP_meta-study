'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

sys.path.insert(0, "..")
from utils import join_data, load_alnaji2021, load_pelz2021, load_wang2023, load_wang2020, load_kupke2020
from utils import preprocess, calculate_direct_repeat, get_sequence
from utils import CUTOFF, RESULTSPATH


def generate_overlap_matrix_plot(dfs: list, dfnames: list):
    '''
    
    '''
    # initialize an empty matrix
    matrix_size = len(dfs)
    matrix = [[0] * matrix_size for _ in range(matrix_size)]

    # calculate the differences and populate the matrix
    labels = list ()
    for i in range(matrix_size):
        set1 = set(dfs[i]["key"])
        labels.append(f"{dfnames[i]} (n={len(set1)})")
        for j in range(matrix_size):
            set2 = set(dfs[j]["key"])

            matrix[i][j] = len(set1 & set2) / (max(len(set1), len(set2)))

            if i == j:
                text = f"{matrix[i][j]:.1f}"
                color = "black"
            else:
                text = f"{matrix[i][j]:.3f}"
                color = "white"

            plt.annotate(text, xy=(j, i), color=color,
                        ha='center', va='center', fontsize=12, fontweight='bold')

    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.xticks(np.arange(len(dfnames)), labels, rotation=30)
    plt.yticks(np.arange(len(dfnames)), dfnames)
    plt.tight_layout()
    plt.grid(False)

    save_path = os.path.join(RESULTSPATH, "intersection_analysis", f"intersection_matrix_PR8.png")
    plt.savefig(save_path)
    plt.close()


def generate_maximum_overlap_candidates(dfs: list, dfnames: list):
    '''
    
    '''
    all_candidates = list()
    for df in dfs:
        all_candidates.extend(df["key"].tolist())

    candidate_counts = Counter(all_candidates)

    candidates = list()
    counts = list()
    dir_reps = list()
    print("candidate\tcount\tdir_rep")
    for cand, count in candidate_counts.items():
        if count >= 3:
            seg, s, e = cand.split("_")
            seq = get_sequence("PR8", seg)
            dir_rep, _ = calculate_direct_repeat(seq, int(s), int(e), w_len=10)
            print(f"{cand}:\t{count}\t{dir_rep}")

            candidates.append(cand)
            counts.append(count)
            dir_reps.append(dir_rep)


    d = dict({"DI": candidates, "counts": counts, "dir_reps": dir_reps})
    df = pd.DataFrame(d)
    
    df[["Segment", "Start", "End"]] = df["DI"].str.split("_", expand=True)
    print(df.groupby(["Segment"]).size().reset_index(name='count'))
    
    for c in df[df["counts"] >= 4]["DI"].tolist():
        for df, dfname in zip(dfs, dfnames):
            if c not in df["key"].tolist():
                print(f"{c} not in {dfname}")


if __name__ == "__main__":
    plt.style.use("seaborn")
    dfs = list()
    dfnames = list()

    ### Alnaji2021 ###
    strain = "PR8"
    df = join_data(load_alnaji2021())
    dfs.append(preprocess(strain, df, CUTOFF))
    dfnames.append("Alnaji2021")

    ### Pelz2021 ###
    strain = "PR8"
    df = join_data(load_pelz2021())
    dfs.append(preprocess(strain, df, CUTOFF))
    dfnames.append("Pelz2021")

    ### Wang2023 ###
    strain = "PR8"
    df = join_data(load_wang2023())
    dfs.append(preprocess(strain, df, CUTOFF))
    dfnames.append(f"Wang2023")

    ### Wang2020 ###
    strain = "PR8"
    df = load_wang2020()
    dfs.append(preprocess(strain, join_data(df), CUTOFF))
    dfnames.append(f"Wang2020")
        
    ### Kupke2020 ###
    strain = "PR8"
    df = join_data(load_kupke2020())
    dfs.append(preprocess(strain, df, 5))
    dfnames.append("Kupke2020")


    generate_overlap_matrix_plot(dfs, dfnames)
    generate_maximum_overlap_candidates(dfs, dfnames)