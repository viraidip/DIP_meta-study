'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from scipy.stats import percentileofscore

sys.path.insert(0, "..")
from utils import join_data, load_dataset, load_single_dataset
from utils import preprocess, calculate_direct_repeat, get_sequence
from utils import CUTOFF, RESULTSPATH, SEGMENT_DICTS, ACCNUMDICT, DATASET_STRAIN_DICT


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


def generate_max_overlap_candidates(dfs: list, dfnames: list):
    '''
    
    '''
    all_candidates = list()
    for df in dfs:
        all_candidates.extend(df["key"].tolist())

    candidate_counts = Counter(all_candidates)
    max_count = max(candidate_counts.values())

    candidates = list()
    counts = list()
    dir_reps = list()
    print("candidate\tcount\tdir_rep")
    for cand, count in candidate_counts.items():
        if count >= max_count-1:
            seg, s, e = cand.split("_")
            seq = get_sequence("PR8", seg)
            dir_rep, _ = calculate_direct_repeat(seq, int(s), int(e), w_len=10)
            print(f"{cand}:\t{count}\t{dir_rep}")

            candidates.append(cand)
            counts.append(count)
            dir_reps.append(dir_rep)

    count_df = pd.DataFrame(dict({"DI": candidates, "counts": counts, "dir_reps": dir_reps}))

    count_df[["Segment", "Start", "End"]] = count_df["DI"].str.split("_", expand=True)
    print(df.groupby(["Segment"]).size().reset_index(name='count'))

    return count_df


def analyze_max_overlap_candidates(dfs, dfnames, count_df):
    '''
    
    '''
    plt.figure(figsize=(8, 6), tight_layout=True)
    plot_data = list()
    labels = [f"{name} ({df.shape[0]})" for name, df in zip(dfnames, dfs)]
    for df in dfs:
        plot_data.append(df["NGS_read_count"])
    plt.boxplot(plot_data, labels=labels)
    
    x_p = range(1, len(dfs)+1)
    max_count = count_df["counts"].max()
    for c in count_df[count_df["counts"] >= max_count]["DI"].tolist():
        print(f"### {c} ###")
        y_p = list()
        for df, dfname in zip(dfs, dfnames):
            if c not in df["key"].tolist():
                print(f"{c} not in {dfname}")
                y_p.append(0)
            else:
                ngs_count = df[df["key"] == c]["NGS_read_count"].values[0]
                y_p.append(ngs_count)
                percentile = percentileofscore(df["NGS_read_count"], ngs_count)
                print(f"{dfname}\t{ngs_count}\t{percentile:.1f}")

        plt.scatter(x_p, y_p, marker="x", label=c, zorder=100)

    plt.yscale("log")
    plt.xticks(rotation=45) 
    plt.xlabel("Datasets")
    plt.ylabel("NGS read count (log scale)")
    plt.legend()

    save_path = os.path.join(RESULTSPATH, "intersection_analysis", "ngs_counts.png")
    plt.savefig(save_path)
    plt.close()

    # compare to the labels of Pelz et al.
    for c in count_df[count_df["counts"] >= max_count]["DI"].tolist():
        print(f"### {c} ###")  
        counts = list()  
        for accnum, meta in ACCNUMDICT["Pelz2021"].items():
            t_df = load_single_dataset("Pelz2021", accnum, SEGMENT_DICTS["PR8"])
            t_df["key"] = t_df["Segment"] + "_" + t_df["Start"].map(str) + "_" + t_df["End"].map(str)
            
            count = t_df[t_df["key"] == c]["NGS_read_count"]

            if count.empty:
                counts.append(0)
            else:
                counts.append(count.values[0])

        print(counts)

        t1 = "de novo " if counts[0] == 0 else ""
        t2 = "gain" if counts[0] < counts[-1] else "loss"

        print(f"{t1}{t2}")
    
    
if __name__ == "__main__":
    plt.style.use("seaborn")
    
    dfs = list()
    dfnames = ["Alnaji2021", "Pelz2021", "Wang2023", "Wang2020", "Kupke2020", "EBI2020"]
    for dataset in dfnames:
        strain = DATASET_STRAIN_DICT[dataset]
        df = join_data(load_dataset(dataset))
        dfs.append(preprocess(strain, df, CUTOFF))

    generate_overlap_matrix_plot(dfs, dfnames)
    count_df = generate_max_overlap_candidates(dfs, dfnames)
    print(count_df)
    # TODO: only select PB2 candidates
    analyze_max_overlap_candidates(dfs, dfnames, count_df)