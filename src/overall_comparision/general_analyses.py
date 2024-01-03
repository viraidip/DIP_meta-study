'''
    Performs general analyses of the datasets
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from typing import Tuple
from collections import Counter

sys.path.insert(0, "..")
from utils import load_all, get_sequence, get_seq_len, get_p_value_symbol, plot_heatmap, create_nucleotide_ratio_matrix, count_direct_repeats_overall, get_dataset_names, sort_datasets_by_type
from utils import SEGMENTS, RESULTSPATH, DATASET_STRAIN_DICT, CMAP, NUCLEOTIDES, CUTOFF


def plot_distribution_over_segments(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        creates a plot that shows how the DelVGs are distributed over the
        segments for a given list of datasets.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None
    '''
    fig, axs = plt.subplots(figsize=(5, 5))
    cm = plt.get_cmap(CMAP)
    colors = [cm(1.*i/len(SEGMENTS)) for i in range(len(SEGMENTS))]

    pvalues = list()
    x = np.arange(0, len(dfs))
    y = dict({s: list() for s in SEGMENTS})
    for df, dfname in zip(dfs, dfnames):
        fractions = df["Segment"].value_counts() / len(df) * 100
        for s in SEGMENTS:
            if s not in fractions:
                fractions[s] = 0.0
            y[s].append(fractions[s])
        
        f_obs = fractions
        full_seqs = np.array([get_seq_len(DATASET_STRAIN_DICT[dfname], seg) for seg in SEGMENTS])
        f_exp = full_seqs / sum(full_seqs) * 100
        _, pvalue = stats.chisquare(f_obs, f_exp)
        pvalues.append(pvalue)

    bar_width = 0.7
    bottom = np.zeros(len(dfs))

    for i, s in enumerate(SEGMENTS):
        axs.barh(x, y[s], bar_width, color=colors[i], label=s, left=bottom)
        for j, text in enumerate(y[s]):
            if text > 10:
                axs.text(bottom[j] + text/2, j, str(round(text, 1)), ha="center", va="center", fontsize=6)
        bottom += y[s]
    
    axs.set_xlabel("segment occurrence [%]")
    axs.set_ylabel("dataset")
    plt.yticks(range(len(dfnames)), [f"{dfname} (n={len(df)}) {get_p_value_symbol(p)}" for dfname, df, p in zip(dfnames, dfs, pvalues)])
    axs.legend(loc="upper center", bbox_to_anchor=(0.3, 1.15), fancybox=True, shadow=True, ncol=4)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "fraction_segments.png"))
    plt.close()

    frac_df = pd.DataFrame(y)
    frac_df["name"] = dfnames
    frac_df.to_csv(os.path.join(save_path, "fraction_segments.csv"))


def calculate_deletion_shifts(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        creates a plot that shows the deletion shifts of the DelVGs.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None
    '''
    fig, axs = plt.subplots(figsize=(5, 5))
    cm = plt.get_cmap(CMAP)
    colors = [cm(1.*i/3) for i in range(3)]

    pvalues = list()
    x = np.arange(0, len(dfs))
    y = dict({n: list() for n in [0, 1, 2]})
    for df in dfs:
        df["length"] = df["deleted_sequence"].apply(len)
        df["shift"] = df["length"] % 3
        shifts = df["shift"].value_counts()
        n = df.shape[0]
        for idx in [0, 1, 2]:
            if idx not in shifts.index:
                shifts.loc[idx] = 0    
            y[idx].append(shifts.loc[idx] / n * 100)

        f_obs = shifts / sum(shifts) * 100
        f_exp = [33.3333333, 33.3333333, 33.3333333]
        _, pvalue = stats.chisquare(f_obs, f_exp)
        pvalues.append(pvalue)

    bar_width = 0.7
    bottom = np.zeros(len(dfs))
    labels = list(["in-frame", "shift +1", "shift -1"])
    for i, label in enumerate(labels):
        axs.barh(x, y[i], bar_width, color=colors[i], label=label, left=bottom)
        for j, text in enumerate(y[i]):
            axs.text(bottom[j] + text/2, j, str(round(text, 1)), ha="center", va="center", fontsize=6)
        bottom += y[i]
    
    axs.set_xlabel("deletion shift [%]")
    axs.set_ylabel("dataset")
    plt.yticks(range(len(dfnames)), [f"{dfname} (n={len(df)}, p.={p:.2})" for dfname, df, p in zip(dfnames, dfs, pvalues)])
    axs.legend(loc="upper center", bbox_to_anchor=(0.3, 1.1), fancybox=True, shadow=True, ncol=3)

    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "deletion_shifts.png"))
    plt.close()
 

def calc_DI_lengths(dfs: list, dfnames: list)-> dict:
    '''
        counts the length of the DelVGs for each segment independently.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        
        :return: dictionary with the DelVG lengths per segment for each dataset
    '''
    overall_count_dict = dict()
    for df, dfname in zip(dfs, dfnames):
        # create a dict for each segment including the NGS read count
        count_dict = dict()
        for s in SEGMENTS:
            count_dict[s] = dict()

        for _, r in df.iterrows():
            DVG_Length = len(r["seq"])-len(r["deleted_sequence"])
            if DVG_Length in count_dict[r["Segment"]]:
                count_dict[r["Segment"]][DVG_Length] += 1
            else:
                count_dict[r["Segment"]][DVG_Length] = 1

        overall_count_dict[dfname] = count_dict
    
    return overall_count_dict
    

def length_distribution_histrogram(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        creates a histogram that shows the length distribution of the DelVGs.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None
    '''
    plt.rc("font", size=16)
    overall_count_dict = calc_DI_lengths(dfs, dfnames)

    for s in SEGMENTS:
        fig, axs = plt.subplots(len(dfnames), 1, figsize=(10, len(dfnames)*1.5), tight_layout=True)
        for i, dfname in enumerate(dfnames):
            count_dict = overall_count_dict[dfname]
            if len(count_dict[s].keys()) > 1:
                counts_list = list()
                for length, count in count_dict[s].items():
                    counts_list.extend([length] * count)

                m = round(np.mean(list(counts_list)), 2)                
                axs[i].hist(count_dict[s].keys(), weights=count_dict[s].values(), bins=100, label=f"{dfname} (µ={m})", alpha=0.3)
                axs[i].set_xlim(left=0)
                axs[i].set_xlabel("sequence length")
                axs[i].set_ylabel("occurrences")
                axs[i].legend()
            else:
                axs[i].set_visible(False)
                m = 0

        save_path = os.path.join(RESULTSPATH, folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"{s}_length_del_hist.png"))
        plt.close()


def length_distribution_violinplot(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        creates a violinplot that shows the length distribution of the DelVGs.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None    
    '''
    dfs, dfnames = sort_datasets_by_type(dfs, dfnames, cutoff=50)
    plt.rc("font", size=16)
    overall_count_dict = calc_DI_lengths(dfs, dfnames)

    for s in SEGMENTS:
        fig, axs = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
        plot_list = list()
        position_list = list()
        labels = list()

        for i, dfname in enumerate(dfnames):
            n_counts = len(overall_count_dict[dfname][s].keys())
            if n_counts >= 1:
                counts_list = list()
                for length, count in overall_count_dict[dfname][s].items():
                    counts_list.extend([length] * count)
                plot_list.append(counts_list)
                position_list.append(i+1)            
            labels.append(f"{dfname} (n={n_counts})")
        
        axs.violinplot(plot_list, position_list, points=1000, showmedians=True)
        axs.set_xticks(range(1, len(dfnames)+1))
        axs.set_xticklabels(labels, rotation=90)
        axs.set_xlabel("Dataset")
        axs.set_ylabel("DVG sequence length")

        save_path = os.path.join(RESULTSPATH, folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"{s}_length_del_violinplot.png"))
        plt.close()


def start_vs_end_lengths(dfs: list, dfnames: list, limit: int=0, folder: str="general_analysis")-> None:
    '''
        plots the length of the start against the length of the end of the
        DelVG RNA sequences as a scatter plot (5' = start, 3' = end).
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param limit: defines where to set the x- and y-axis limits
        :param folder: defines where to save the results
    
        :return: None  
    '''
    for df, dfname in zip(dfs, dfnames):
        fig, axs = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
        j = 0
        for i, s in enumerate(SEGMENTS):
            df_s = df[df["Segment"] == s].copy()
            if df_s.shape[0] > 1:
                df_s["End_L"] = df_s["full_seq"].str.len() - df_s["End"]
                axs[j,i%4].scatter(df_s["Start"], df_s["End_L"], s=1.0)
                axs[j,i%4].plot([0, 1], [0, 1], transform=axs[j,i%4].transAxes, c="r", linewidth=0.5, linestyle="--")
                if limit == 0:
                    max_p = max(df_s["Start"].max(), df_s["End_L"].max())
                else:
                    max_p = limit
                    axs[j,i%4].set_xlim(0, max_p)
                    axs[j,i%4].set_ylim(0, max_p)
                    df_s = df_s[(df_s["Start"] <= max_p) & (df_s["End_L"] <= max_p)]
                if df_s.shape[0] > 1:
                    pearson = stats.pearsonr(df_s["Start"], df_s["End_L"])
                else:
                    pearson = ["NA", "NA"]
                axs[j,i%4].set_xticks([0, max_p/2, max_p])
                axs[j,i%4].set_yticks([0, max_p/2, max_p])
                axs[j,i%4].set_aspect("equal", "box")

            else:
                axs[j,i%4].set_visible(False)

            axs[j,i%4].set_title(f"{s} (r={pearson[0]:.2})")
            axs[j,i%4].set_xlabel("3' end")
            axs[j,i%4].set_ylabel("5' end")

            if i == 3:
                j = 1

        if limit == 0:
            filename = f"{dfname}_length_start_end.png"
        else:
            filename = f"{dfname}_length_start_end_{limit}.png"

        save_path = os.path.join(RESULTSPATH, folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, filename))
        plt.close()


def calc_start_end_lengths(dfs: list, dfnames: list, thresh: int=300)-> Tuple[list, list]:
    '''
        calcualtes the difference of the start and end lengths of the DelVG RNA
        sequences (5' = start, 3' = end).
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param thresh: defines to which difference DelVGs should be included,
            allows to exclude long DelVGs
    
        :return: Tuple
            List of difference between start and end lengths
            List of labels for the plot
    '''
    plot_list = list()
    labels = list()
    for df, dfname in zip(dfs, dfnames):
        df["End_L"] = df["full_seq"].str.len() - df["End"]
        l = (df["Start"] - df["End_L"]).to_list()
        
        l = [x for x in l if x <= thresh]
        l = [x for x in l if x >= -thresh]
        plot_list.append(l)
        labels.append(f"{dfname} (n={df.shape[0]})")

    return plot_list, labels


def diff_start_end_lengths(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        plots the difference of the start and end lengths of the DelVG RNA
        sequences as a violinplot (5' = start, 3' = end).
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None  
    '''
    fig, axs = plt.subplots(1, 1, figsize=(7, 5), tight_layout=True)
    thresh = 300
    plot_list, labels = calc_start_end_lengths(dfs, dfnames, thresh)

    position_list = np.arange(0, len(dfs))
    axs.violinplot(plot_list, position_list, points=1000, showmedians=True)
    axs.set_xticks(position_list)
    axs.set_xticklabels(labels, rotation=90)
    axs.set_ylabel("5'-end length - 3'-end length")

    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "diff_start_end_violinplot.png"))
    plt.close()


def plot_nucleotide_ratio_around_deletion_junction_heatmaps(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        plot heatmaps of nucleotide ratios around deletion junctions.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None
    '''
    fig, axs = plt.subplots(figsize=(13, len(dfs)), nrows=2, ncols=2)
    axs = axs.flatten()
    for i, nuc in enumerate(NUCLEOTIDES.keys()):
        x = list()
        y = list()
        vals = list()
        for dfname, df in zip(dfnames, dfs):
            probability_matrix = create_nucleotide_ratio_matrix(df, "seq_around_deletion_junction")
            for j in probability_matrix.index:
                x.append(j)
                y.append(dfname)
                vals.append(probability_matrix.loc[j, nuc] * 100)
                
        axs[i] = plot_heatmap(x,y,vals, axs[i], vmin=min(vals), vmax=max(vals), cbar=True, format=".0f")
        for val_label in axs[i].texts:
            val_label.set_size(8)
        axs[i].set_title(f"{NUCLEOTIDES[nuc]}")
        axs[i].set_ylabel("")
        axs[i].set_yticks([ytick + 0.5 for ytick in range(len(dfnames))])
        axs[i].set_xlabel("position")  
        axs[i].set_xticks([xtick - 0.5 for xtick in probability_matrix.index])
        
        quarter = len(probability_matrix.index) // 4
        indexes = [pos for pos in range(1, quarter * 2 + 1)]
        if i % 2 == 0:
            axs[i].set_yticklabels([f"{dfname} ({len(df)})" for dfname, df in zip(dfnames, dfs)])
        else:
            axs[i].set_yticklabels([])
        if i < 2:
            axs[i].xaxis.set_ticks_position("top")
            axs[i].xaxis.set_label_position("top")

        axs[i].set_xticklabels(indexes + indexes)
        xlabels = axs[i].get_xticklabels()
        for x_idx, xlabel in enumerate(xlabels):
            if x_idx < quarter or x_idx >= quarter * 3:
                xlabel.set_color("black")
                xlabel.set_fontweight("bold")
            else:
                xlabel.set_color("grey") 
          
    fig.subplots_adjust(top=0.9)
    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "nuc_occ.png"))
    plt.close()


def plot_direct_repeat_ratio_heatmaps(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        plot heatmaps of nucleotide ratios around deletion junctions.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None
    '''
    fig, axs = plt.subplots(figsize=(10, len(dfs)/2))
    x = list()
    y = list()
    vals = list()
    for dfname, df in zip(dfnames, dfs):
        final_d = dict()
        for s in SEGMENTS:
            df_s = df[df["Segment"] == s]
            if len(df_s) == 0:
                continue
            seq = get_sequence(df_s["Strain"].unique()[0], s)
            counts, _ = count_direct_repeats_overall(df_s, seq)
            for k, v in counts.items():
                if k in final_d:
                    final_d[k] += v
                else:
                    final_d[k] = v

        x.extend(final_d.keys())
        y.extend([f"{dfname} ({len(df)})" for _ in range(6)])
        final = np.array(list(final_d.values()))
        vals.extend(final/final.sum())

    axs = plot_heatmap(x,y,vals, axs, vmin=0, vmax=1, cbar=True, format=".5f")
    axs.set_title("direct repeat ratios around deletion junction")
    axs.set_ylabel("")
    axs.set_xlabel("direct repeat length")

    x_ticks = axs.get_xticklabels()
    label = x_ticks[-2].get_text()
    x_ticks[-1].set_text(f"> {label}")
    axs.set_xticklabels(x_ticks)
    fig.tight_layout()

    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "dir_rep.png"))
    plt.close()


def deletion_site_motifs(dfs: list, dfnames: list, m_len: int, folder: str="general_analysis")-> None:
    '''
        calcualte the motifs of specified length before start and end of
        deletion site.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param m_len: lenght of the motif to consider
        :param folder: defines where to save the results
    
        :return: None
    '''
    results = dict({
        "name": dfnames,
        "start": list(),
        "start prct.": list(),
        "end": list(),
        "end prct.": list()
    })
    for df in dfs:
        s_motifs = list()
        e_motifs = list()
        for _, r in df.iterrows():
            seq = r["full_seq"]
            s = r["Start"]
            e = r["End"]
            s_motif = seq[s-m_len:s+m_len]
            e_motif = seq[e-(m_len+1):e+(m_len-1)]
            s_motifs.append(s_motif[:m_len])
            e_motifs.append(e_motif[:m_len])

        s_motifs_counts = Counter(s_motifs)
        start_motif = s_motifs_counts.most_common(1)[0]
        e_motifs_counts = Counter(e_motifs)
        end_motif = e_motifs_counts.most_common(1)[0]

        results["start"].append(start_motif[0])
        results["start prct."].append(round(start_motif[1]/df.shape[0] * 100, 1))
        results["end"].append(end_motif[0])
        results["end prct."].append(round(end_motif[1]/df.shape[0] * 100, 1))

    results_df = pd.DataFrame(results)

    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = (os.path.join(save_path, "deletion_site_motif.txt"))
    with open(path , "w") as f:
        print(results_df, file=f)
        print(Counter(results["start"]), file=f)
        print(Counter(results["end"]), file=f)


if __name__ == "__main__":
    plt.style.use("seaborn")
    
    dfnames = get_dataset_names(cutoff=50)
    dfs, _ = load_all(dfnames)

    plot_distribution_over_segments(dfs, dfnames)
    calculate_deletion_shifts(dfs, dfnames)
    length_distribution_histrogram(dfs, dfnames)
    length_distribution_violinplot(dfs, dfnames)
    plot_nucleotide_ratio_around_deletion_junction_heatmaps(dfs, dfnames)
    plot_direct_repeat_ratio_heatmaps(dfs, dfnames)
    start_vs_end_lengths(dfs, dfnames, limit=600)
    diff_start_end_lengths(dfs, dfnames)
    deletion_site_motifs(dfs, dfnames, w_len=2)
    