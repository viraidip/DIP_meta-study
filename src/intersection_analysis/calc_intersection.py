'''
    Analyses the intersection of DelVG candidates between the given PR8
    datasets.
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
from utils import preprocess, calculate_direct_repeat, get_sequence, generate_sampling_data
from utils import RESULTSPATH, SEGMENT_DICTS, ACCNUMDICT


def generate_overlap_matrix_plot(dfs: list, dfnames: list, name: str=""):
    '''
        plot a matrix that shows how big the overlap of DelVGs between the
        given datasets is.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param name: string to define result and name of resultsfile

        :return: None
    '''
    plt.figure(figsize=(5, 4))
    plt.rc("font", size=20)
    # initialize an empty matrix
    matrix_size = len(dfs)
    matrix = [[0] * matrix_size for _ in range(matrix_size)]
    # calculate the differences and populate the matrix
    for i in range(matrix_size):
        set1 = set(dfs[i]["key"])
        for j in range(matrix_size):
            set2 = set(dfs[j]["key"])
            matrix[i][j] = len(set1 & set2) / (max(len(set1), len(set2), 1)) * 100
            if i == j:
                text = f"{matrix[i][j]:.0f}"
                color = "black"
            else:
                text = f"{matrix[i][j]:.1f}"
                color = "white"
            plt.annotate(text, xy=(j, i), color=color, ha='center', va='center', fontsize=12, fontweight='bold')

    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04, label="Intersecting DelVGs [%]")
    plt.xticks(np.arange(len(dfnames)), dfnames, rotation=90)
    plt.yticks(np.arange(len(dfnames)), dfnames)
    plt.tight_layout()
    plt.grid(False)

    if name != "":
        filename = f"{name}_intersection_matrix_PR8.png"
    else:
        filename = "intersection_matrix_PR8.png"
    save_path = os.path.join(RESULTSPATH, "intersection_analysis")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, filename))
    plt.close()


def generate_max_overlap_candidates(dfs: list, thresh: int=0)-> pd.DataFrame:
    '''
        analyses for all DelVGs in the given datasets how often they occur in
        the datasets and resturns this in a DataFrame.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param thresh: threshold for occurrence of a DelVG to be considered

        :return: DataFrame with all DelVGs above or equal to given threshold
    '''
    all_candidates = list()
    for df in dfs:
        all_candidates.extend(df["key"].tolist())
    candidate_counts = Counter(all_candidates)
    if thresh == 0:
        thresh = max(candidate_counts.values())-1
    candidates = list()
    counts = list()
    dir_reps = list()
    s_motifs = list()
    e_motifs = list()
    print("candidate\tcount\tdir_rep")
    for cand, count in candidate_counts.items():
        if count >= thresh:
            seg, s, e = cand.split("_")
            s = int(s)
            e = int(e)
            seq = get_sequence("PR8", seg)
            dir_rep, _ = calculate_direct_repeat(seq, s, e, w_len=10)
            w_len = 2
            s_motif = seq[s-w_len:s]
            e_motif = seq[e-(w_len+1):e-1]

            candidates.append(cand)
            counts.append(count)
            dir_reps.append(dir_rep)
            s_motifs.append(s_motif)
            e_motifs.append(e_motif)

    count_df = pd.DataFrame(dict({"DI": candidates, "counts": counts, "dir_reps": dir_reps}))
    count_df[["Segment", "Start", "End"]] = count_df["DI"].str.split("_", expand=True)
    print(count_df.groupby(["Segment"]).size().reset_index(name='count'))
    return count_df


def analyze_max_overlap_candidates(dfs: list, dfnames: list, count_df: pd.DataFrame, name: str="")-> None:
    '''
        check the NGS counts of the identified DelVGs with high occurrence for
        all provided datasets. Also checks label given in Pelz et al. 2021.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param count_df: DataFrame with all DelVGs with high occurrence across the
            datasets
        :param name: string to define result and name of resultsfile

        :return: None
    '''
    plt.figure(figsize=(5, 8), tight_layout=True)
    plot_data = list()
    labels = [f"{name} ({df.shape[0]})" for name, df in zip(dfnames, dfs)]
    for df in dfs:
        plot_data.append(df["NGS_read_count"])
    plt.boxplot(plot_data, labels=labels)

    # mark identified DelVGs in boxplot
    x_p = np.arange(1, len(dfs)+1)
    counts_list = count_df["DI"].tolist()
    for i, c in enumerate(counts_list):
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

        shift = i / ((len(counts_list)-1)*2) - 0.25
        plt.scatter(x_p+shift, y_p, marker="x", label=c, zorder=100)
    plt.yscale("log")
    plt.xticks(rotation=90)
    plt.ylabel("NGS read count (log scale)")
    plt.legend(loc="upper right", ncol=1, frameon=True, shadow=True, facecolor="white", edgecolor="black")
    if name != "":
        filename = f"{name}_ngs_counts.png"
    else:
        filename = "ngs_counts.png"
    save_path = os.path.join(RESULTSPATH, "intersection_analysis")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

    # compare to the labels of Pelz et al.
    for c in counts_list:
        print(f"### {c} ###")  
        counts = list()  
        for accnum in ACCNUMDICT["Pelz2021"].keys():
            t_df = load_single_dataset("Pelz2021", accnum, SEGMENT_DICTS["PR8"])
            t_df["key"] = t_df["Segment"] + "_" + t_df["Start"].map(str) + "_" + t_df["End"].map(str)
            count = t_df[t_df["key"] == c]["NGS_read_count"]
            if count.empty:
                counts.append(0)
            else:
                counts.append(count.values[0])
        t1 = "de novo " if counts[0] == 0 else ""
        t2 = "gain" if counts[0] < counts[-1] else "loss"
        print(f"{t1}{t2}")


if __name__ == "__main__":
    plt.style.use("seaborn")
    
    dfs = list()
    dfnames = ["Alnaji2021", "Pelz2021", "Wang2023", "Wang2020", "Kupke2020", "EBI2020", "IRC2015"]
    strain = "PR8"
    for dataset in dfnames:
        df = join_data(load_dataset(dataset))
        #df = df[df["Segment"].isin(["PB2", "PB1", "PA"])]
        dfs.append(preprocess(strain, df, 1))    
        
    original_stdout = sys.stdout
    save_path = os.path.join(RESULTSPATH, "intersection_analysis")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "intersection_analysis.log"), 'w') as file:
        sys.stdout = file
        generate_overlap_matrix_plot(dfs, dfnames)
        count_df = generate_max_overlap_candidates(dfs)
        analyze_max_overlap_candidates(dfs, dfnames, count_df)
        count_df = count_df[count_df["DI"].str.startswith("PB2")]
        print("### only PB2 Candidates ###")
        analyze_max_overlap_candidates(dfs, dfnames, count_df, name="only_PB2")
    sys.stdout = original_stdout


    exit()
    ### make validation with sampling data ###
    original_stdout = sys.stdout
    file_path = os.path.join(RESULTSPATH, "intersection_analysis", "control_analysis.log")
    file = open(file_path, 'w')
    sys.stdout = file
    for i in range(1):
        sampl_dfs = list()
        print(f"##### round {i} #####")
        for df in dfs:
            for seg in ["PB2", "PB1", "PA"]:
                s_df = df.loc[df["Segment"] == seg]
                n = s_df.shape[0]
                if n == 0:
                    continue
                seq = get_sequence(strain, seg)
                start = int(s_df["Start"].mean())
                end = int(s_df["End"].mean())
                s = (max(start-200, 50), start+200)
                e = (end-200, min(end+200, len(seq)-50))
                assert s[1] < e[0], "Sampling: start and end positions are overlapping!"
                #skip if there is no range given
                #this would lead to oversampling of a single position
                if s[0] == s[1] or e[0] == e[1]:
                    continue
                if "samp_df" in locals():
                    temp_df = generate_sampling_data(seq, s, e, n)
                    temp_df["Segment"] = seg
                    samp_df = pd.concat([samp_df, temp_df], ignore_index=True)
                else:
                    samp_df = generate_sampling_data(seq, s, e, n)
                    samp_df["Segment"] = seg
            samp_df["NGS_read_count"] = 1
            sampl_dfs.append(preprocess(strain, samp_df, 1))
            del samp_df
        generate_overlap_matrix_plot(sampl_dfs, dfnames, name="testing")
        count_df = generate_max_overlap_candidates(sampl_dfs, thresh=3)
    sys.stdout = original_stdout
    file.close()
