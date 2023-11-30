'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy import stats
from scipy.stats import chi2_contingency

sys.path.insert(0, "..")
from utils import load_all, get_sequence, plot_heatmap, create_nucleotide_ratio_matrix, count_direct_repeats_overall
from utils import SEGMENTS, RESULTSPATH, DATASET_STRAIN_DICT, CMAP, NUCLEOTIDES


def plot_distribution_over_segments(dfs: list, dfnames: list, name: str="")-> None:
    '''

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        col (str, optional): The column name in the DataFrames that contains the sequence segments of interest. 
                             Default is "seq_around_deletion_junction".

    :return: None
    '''
    fig, axs = plt.subplots(figsize=(len(dfs), 5), nrows=1, ncols=1)
    cm = plt.get_cmap(CMAP)
    colors = [cm(1.*i/len(SEGMENTS)) for i in range(len(SEGMENTS))]

    x = np.arange(0, len(dfs))

    y = dict({s: list() for s in SEGMENTS})
    for df in dfs:
        fractions = df["Segment"].value_counts() / len(df)
        for s in SEGMENTS:
            if s not in fractions:
                y[s].append(0.0)
            else:
                y[s].append(fractions[s])

    bar_width = 0.7
    bottom = np.zeros(len(dfs))

    for i, s in enumerate(SEGMENTS):
        axs.bar(x, y[s], bar_width, color=colors[i], label=s, bottom=bottom)
        bottom += y[s]
    
    axs.set_ylabel("relative occurrence of segment")
    axs.set_xlabel("dataset")
    plt.xticks(range(len(dfnames)), dfnames, rotation=25)

    box = axs.get_position()
    axs.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    axs.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=8)
    
    if name != "":
        filename = f"fraction_segments_{name}.png"
    else:
        filename = "fraction_segments.png"

    save_path = os.path.join(RESULTSPATH, "general_analysis", filename)
    plt.savefig(save_path)
    plt.close()


def calculate_deletion_shifts(dfs: list, dfnames: list)-> None:
    '''

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        col (str, optional): The column name in the DataFrames that contains the sequence segments of interest. 
                             Default is "seq_around_deletion_junction".
    :return: None
    '''
    fig, axs = plt.subplots(figsize=(12, 12), nrows=5, ncols=5)
    cm = plt.get_cmap(CMAP)
    colors = [cm(1.*i/3) for i in range(3)]

    i = 0
    j = 0
    overall = np.array([0, 0, 0])
    n = 0
    li = list()

    for df, dfname in zip(dfs, dfnames):
        df["length"] = df["deleted_sequence"].apply(len)
        df["shift"] = df["length"] % 3
        shifts = df["shift"].value_counts()
        for idx in [0, 1, 2]:
            if idx not in shifts.index:
                shifts.loc[idx] = 0

        sorted_shifts = shifts.loc[[0, 1, 2]]
        overall += sorted_shifts
        n += len(df)
        li.append(shifts)
        shifts = shifts / len(df)

        axs[i,j].set_title(dfname)
        labels = list(["in-frame", "shift +1", "shift -1"])
        axs[i,j].pie(sorted_shifts, labels=labels, autopct="%1.1f%%", colors=colors, textprops={"size": 14})

        j += 1
        if j == 5:
            i += 1
            j = 0

    table = np.array(li)
 #   statistic, pvalue, dof, expected_freq = chi2_contingency(table)
  #  print(statistic)
   # print(pvalue)

    print(f"mean distribution:\n\t{overall/n}")

    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, "general_analysis", "deletion_shifts.png")
    plt.savefig(save_path)
    plt.close()
 

def length_distribution_heatmap(dfs: list, dfnames: list)-> None:
    '''
    
    '''
    plt.rc("font", size=16)
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

    for s in SEGMENTS:
        fig, axs = plt.subplots(len(dfnames), 1, figsize=(10, 15), tight_layout=True)
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


        save_path = os.path.join(RESULTSPATH, "general_analysis", f"{s}_length_del_hist.png")
        plt.savefig(save_path)
        plt.close()


def length_distribution_violinplot(dfs: list, dfnames: list)-> None:
    '''
    
    '''
    plt.rc("font", size=16)
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

    for s in SEGMENTS:
        fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
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
        axs.set_xticklabels(labels, rotation=45)
        axs.set_xlabel("Dataset")
        axs.set_ylabel("DVG sequence length")

        save_path = os.path.join(RESULTSPATH, "general_analysis", f"{s}_length_del_violinplot.png")
        plt.savefig(save_path)
        plt.close()


def start_vs_end_lengths(dfs, dfnames, limit: int=0)-> None:
    '''
        Plots the length of the start against the length of the end of the DI
        RNA sequences as a scatter plot.

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

        save_path = os.path.join(RESULTSPATH, "general_analysis", filename)
        plt.savefig(save_path)
        plt.close()


def diff_start_end_lengths(dfs, dfnames, name: str="")-> None:
    '''

    '''
    fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    plot_list = list()
    position_list = np.arange(0, len(dfs))
    labels = list()

    for df, dfname in zip(dfs, dfnames):
        df["End_L"] = df["full_seq"].str.len() - df["End"]
        l = (df["Start"] - df["End_L"]).to_list()
        thresh = 300
        l = [x for x in l if x <= thresh]
        l = [x for x in l if x >= -thresh]
        plot_list.append(l)
        labels.append(f"{dfname} (n={df.shape[0]})")

    axs.violinplot(plot_list, position_list, points=1000, showmedians=True)
    axs.set_xticks(position_list)
    axs.set_xticklabels(labels, rotation=25)
    axs.set_xlabel("Dataset")
    axs.set_ylabel("Start-End sequence lengths")
    axs.set_title(f"Difference of start to end sequence lengths (threshold={thresh})")

    if name != "":
        filename = f"diff_start_end_violinplot_{name}.png"
    else:
        filename = "diff_start_end_violinplot.png"

    save_path = os.path.join(RESULTSPATH, "general_analysis", filename)
    plt.savefig(save_path)
    plt.close()


def dataset_distributions(dfs: list, dfnames: list)-> None:
    '''
    
    '''
#TODO: add here the number of mapped reads
#      also add the fraction of DIPs found/ number of mapped reads

    ns = list()
    plot_data = list()
    means = list()
    medians = list()
    stddevs = list()
    maxs = list()
    
    for df in dfs:
        ns.append(df.shape[0])
        counts = df["NGS_read_count"]
        plot_data.append(counts)
        means.append(counts.mean())
        medians.append(counts.median())
        stddevs.append(counts.std())
        maxs.append(counts.max())

    labels = [f"{name} ({n})" for name, n in zip(dfnames, ns)]
    plt.figure(figsize=(8, 6), tight_layout=True)
    plt.boxplot(plot_data, labels=labels)
    plt.yscale("log")
    plt.xticks(rotation=45) 
    plt.xlabel("Datasets")
    plt.ylabel("NGS read count (log scale)")

    save_path = os.path.join(RESULTSPATH, "general_analysis", "ngs_count_distribution.png")
    plt.savefig(save_path)
    plt.close()

    stats_df = pd.DataFrame({"Dataset": dfnames,
                             "Size": ns,
                             "Mean": means,
                             "Median": medians,
                             "Std. dev.": stddevs,
                             "Max": maxs})

    save_path = os.path.join(RESULTSPATH, "general_analysis", "ngs_count_stats.csv")
    stats_df.to_csv(save_path, index=False)


def plot_nucleotide_ratio_around_deletion_junction_heatmaps(dfs, dfnames):
    '''
        Plot heatmaps of nucleotide ratios around deletion junctions.

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        height (float, optional): The height of the figure in inches. Default is 20.
        width (float, optional): The width of the figure in inches. Default is 16.
        nucleotides (list of str, optional): The nucleotides to be plotted. Default is ["A", "C", "G", "T"].

    Returns:
        tuple: A tuple containing the figure and the axes of the subplots.
            - fig (matplotlib.figure.Figure): The generated figure.
            - axs (numpy.ndarray of matplotlib.axes.Axes): The axes of the subplots.

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

    save_path = os.path.join(RESULTSPATH, "general_analysis", "nuc_occ.png")
    plt.savefig(save_path)
    plt.close()


def plot_direct_repeat_ratio_heatmaps(dfs: list, dfnames: list)-> None:
    '''
        Plot heatmaps of nucleotide ratios around deletion junctions.

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        col (str, optional): The column name in the DataFrames that contains the sequence segments of interest. 
                             Default is "seq_around_deletion_junction".
        height (float, optional): The height of the figure in inches. Default is 20.
        width (float, optional): The width of the figure in inches. Default is 16.
        nucleotides (list of str, optional): The nucleotides to be plotted. Default is ["A", "C", "G", "T"].

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

    save_path = os.path.join(RESULTSPATH, "general_analysis", "dir_rep.png")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")
    dfnames = DATASET_STRAIN_DICT.keys()
    dfs, expected_dfs = load_all(dfnames)

    dataset_distributions(dfs, dfnames)
    plot_distribution_over_segments(dfs, dfnames)
    calculate_deletion_shifts(dfs, dfnames)
    length_distribution_heatmap(dfs, dfnames)
    length_distribution_violinplot(dfs, dfnames)
    plot_nucleotide_ratio_around_deletion_junction_heatmaps(dfs, dfnames)
    plot_direct_repeat_ratio_heatmaps(dfs, dfnames)
    start_vs_end_lengths(dfs, dfnames, limit=600)
    diff_start_end_lengths(dfs, dfnames)