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
from utils import SEGMENTS, RESULTSPATH, DATASET_STRAIN_DICT, CMAP, NUCLEOTIDES


def get_overall_counts(dfs):
    overall_counts = {}
    for df in dfs:
        counts = df["Segment"].value_counts()
        for segment, count in counts.items():
            if segment in overall_counts:
                overall_counts[segment] += count
            else:
                overall_counts[segment] = count
    overall_counts_df = pd.DataFrame(list(overall_counts.items()), columns=["Segment", "Count"])
    return overall_counts_df


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
    fig, axs = plt.subplots(figsize=(5, 6))
    cm = plt.get_cmap(CMAP)
    colors = [cm(1.*i/len(SEGMENTS)) for i in range(len(SEGMENTS))]
    cramers_vs = list()
    counts = [len(df) for df in dfs]
    y = dict({s: list() for s in SEGMENTS})

    # generate data
    plot_exp = list()
    for df, dfname in zip(dfs, dfnames):
        f_obs = df["Segment"].value_counts()
        fractions = f_obs / len(df) * 100
        for s in SEGMENTS:
            if s not in f_obs:
                f_obs[s] = 0
            if s not in fractions:
                fractions[s] = 0.0
            y[s].append(fractions[s])
        
        full_seqs = np.array([get_seq_len(DATASET_STRAIN_DICT[dfname], seg) for seg in SEGMENTS])
        f_exp = full_seqs / sum(full_seqs) * f_obs.sum()
        plot_exp.append(full_seqs)

        r, pvalue = stats.chisquare(f_obs, f_exp)
        if pvalue < 0.05:
            v_data = np.stack((np.rint(f_obs.to_numpy()), np.rint(f_exp)), axis=1).astype(int)
            cramers_v = stats.contingency.association(v_data)
            cramers_v = round(cramers_v, 2)
        else:
            cramers_v = "n.a."
        cramers_vs.append(cramers_v)

    # add data for IAV overall
    iav_overall = get_overall_counts(dfs[:13])
    iav_overall["Perc"] = iav_overall["Count"] / sum(iav_overall["Count"]) * 100
    iav_exp = np.sum(np.array(plot_exp[:13]), axis=0)
    iav_exp = iav_exp / np.sum(iav_exp) * sum(iav_overall["Count"])
    dfnames.insert(13, "IAV overall")
    r, pvalue = stats.chisquare(iav_overall["Count"], iav_exp)
    if pvalue < 0.05:
        v_data = np.stack((np.rint(iav_overall["Count"].to_numpy()), np.rint(iav_exp)), axis=1).astype(int)
        cramers_v = stats.contingency.association(v_data)
        cramers_v = round(cramers_v, 2)
    else:
        cramers_v = "n.a."
    cramers_vs.insert(13, cramers_v)
    counts.insert(13, sum(iav_overall["Count"]))
    for s in SEGMENTS:
        y[s].insert(13, iav_overall[iav_overall["Segment"] == s]["Perc"].values[0])

    # add data for IBV overall
    ibv_overall = get_overall_counts(dfs[14:])
    ibv_overall["Perc"] = ibv_overall["Count"] / sum(ibv_overall["Count"]) * 100
    ibv_exp = np.sum(np.array(plot_exp[14:]), axis=0)
    ibv_exp = ibv_exp / np.sum(ibv_exp) * sum(ibv_overall["Count"])
    dfnames.append("IBV overall")
    r, pvalue = stats.chisquare(ibv_overall["Count"], ibv_exp)
    if pvalue < 0.05:
        v_data = np.stack((np.rint(ibv_overall["Count"].to_numpy()), np.rint(ibv_exp)), axis=1).astype(int)
        cramers_v = stats.contingency.association(v_data)
        cramers_v = round(cramers_v, 2)
    else:
        cramers_v = "n.a."    
    cramers_vs.append(cramers_v)
    counts.append(sum(ibv_overall["Count"]))
    for s in SEGMENTS:
        y[s].insert(14, ibv_overall[ibv_overall["Segment"] == s]["Perc"].values[0])

    # add data for expected by length (reference)
    labels = [f"{dfname} (n={count}, V={p})  " for dfname, count, p in zip(dfnames, counts, cramers_vs)]
    dfnames.append("Expected by length")
    counts.append(0)
    plot_exp = np.sum(np.array(plot_exp), axis=0)
    plot_exp = plot_exp / sum(plot_exp) * 100
    for idx, s in enumerate(SEGMENTS):
        y[s].append(plot_exp[idx])
    labels.append("Expected by length            ")

    x = np.arange(0, len(dfnames))
    bar_width = 0.7
    bottom = np.zeros(len(dfnames))

    for i, s in enumerate(SEGMENTS):
        axs.barh(x, y[s], bar_width, color=colors[i], label=s, left=bottom, edgecolor="black")
        for j, text in enumerate(y[s]):
            if text > 10:
                axs.text(bottom[j] + text/2, j-0.2, str(round(text, 1)), ha="center", fontsize=9)
        bottom += y[s]
    
    axs.set_xlabel("Fraction of DelVGs per segment [%]")
    plt.yticks(range(len(dfnames)), labels)
    plt.gca().get_yticklabels()[-1].set_fontweight("bold")
    plt.gca().get_yticklabels()[-2].set_fontweight("bold")
    plt.gca().get_yticklabels()[-10].set_fontweight("bold")

    axs.legend(loc="upper center", bbox_to_anchor=(0.3, 1.1), fancybox=True, shadow=True, ncol=4)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "fraction_segments.png"), dpi=300)
    plt.close()

    frac_df = pd.DataFrame(y)
    frac_df["name"] = dfnames
    frac_df.to_csv(os.path.join(save_path, "fraction_segments.csv"), index=False)


def calculate_deletion_shifts(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        creates a plot that shows the deletion shifts of the DelVGs.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None
    '''
    fig, axs = plt.subplots(figsize=(5, 6))
    cm = plt.get_cmap(CMAP)
    colors = [cm(0/8), cm(3/8), cm(1/8)]

    symbols = list()
    x = np.arange(0, len(dfs)+3)
    y = dict({n: list() for n in [2, 0, 1]})
    iav_shifts = list()
    ibv_shifts = list()
    f_exp = np.array([33.3333333, 33.3333333, 33.3333333])
    for df in dfs:
        df["length"] = df["deleted_sequence"].apply(len)
        df["shift"] = df["length"] % 3
        shifts = df["shift"].value_counts()
        if df["Strain"].unique()[0] in ["BLEE", "Victoria", "Brisbane", "Yamagata"]:
            ibv_shifts.append(shifts)
        else:
            iav_shifts.append(shifts)
        n = df.shape[0]
        for idx in [2, 0, 1]:
            y[idx].append(shifts.loc[idx] / n * 100)

        f_exp_copy = (f_exp / 100) * n
        r, pvalue = stats.chisquare(shifts, f_exp_copy)
        if pvalue < 0.05:
            v_data = np.stack((shifts.to_numpy(), np.rint(f_exp_copy)), axis=1).astype(int)
            cramers_v = stats.contingency.association(v_data)
            cramers_v = round(cramers_v, 2)
        else:
            cramers_v = "n.a."
        symbols.append(cramers_v)

    labels = [f"{dfname} (n={len(df)}, V={s})  " for dfname, df, s in zip(dfnames, dfs, symbols)]
    iav_shifts = sum(iav_shifts)
    ibv_shifts = sum(ibv_shifts)
    for idx in [2, 0, 1]:
        y[idx].insert(13, iav_shifts.loc[idx] / sum(iav_shifts) * 100)
        y[idx].append(ibv_shifts.loc[idx] / sum(ibv_shifts) * 100)
        y[idx].append(33.3333333)

    # add IAV data
    f_exp_copy = (f_exp / 100) * iav_shifts.sum()
    r, pvalue = stats.chisquare(iav_shifts, f_exp_copy)
    if pvalue < 0.05:
        v_data = np.stack((iav_shifts.to_numpy(), np.rint(f_exp_copy)), axis=1).astype(int)
        cramers_v = stats.contingency.association(v_data)
        cramers_v = round(cramers_v, 2)
    else:
        cramers_v = "n.a." 
    labels.insert(13, f"IAV overall (n={sum(iav_shifts)}, V={cramers_v})  ")
    
    # add IBV data
    f_exp_copy = (f_exp / 100) * ibv_shifts.sum()
    r, pvalue = stats.chisquare(ibv_shifts, f_exp_copy)
    if pvalue < 0.05:
        v_data = np.stack((ibv_shifts.to_numpy(), np.rint(f_exp_copy)), axis=1).astype(int)
        cramers_v = stats.contingency.association(v_data)
        cramers_v = round(cramers_v, 2)
    else:
        cramers_v = "n.a."
    labels.append(f"IBV overall (n={sum(ibv_shifts)}, V={cramers_v})  ")
    labels.append("Expected by chance  ")

    bar_width = 0.7
    bottom = np.zeros(len(dfs) + 3)
    shift_labels = dict({"shift -1": 2, "in-frame": 0, "shift +1": 1})
    for i, (label, idx) in enumerate(shift_labels.items()):
        axs.barh(x, y[idx], bar_width, color=colors[i], label=label, left=bottom, edgecolor="black")
        for j, text in enumerate(y[idx]):
            axs.text(bottom[j] + text/2, j-0.2, str(round(text, 1)), ha="center", fontsize=9)
        bottom += y[idx]
    
    axs.set_xlim(right=100)
    axs.set_xlabel("Fraction of deletion shift [%]")
    plt.yticks(range(len(labels)), labels)
    plt.gca().get_yticklabels()[-1].set_fontweight("bold")
    plt.gca().get_yticklabels()[-2].set_fontweight("bold")
    plt.gca().get_yticklabels()[-10].set_fontweight("bold")
    axs.legend(loc="upper center", bbox_to_anchor=(0.3, 1.1), fancybox=True, shadow=True, ncol=3)

    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "deletion_shifts.png"), dpi=300)
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
            DVG_Length = len(r["full_seq"])-len(r["deleted_sequence"])
            if DVG_Length in count_dict[r["Segment"]]:
                count_dict[r["Segment"]][DVG_Length] += 1
            else:
                count_dict[r["Segment"]][DVG_Length] = 1

        overall_count_dict[dfname] = count_dict

    calc_means = False
    if calc_means:
        in_vivo = list()
        in_vitro = list()
        min_median = (2000, "dataset name")
        max_median = (0, "dataset name")
        for k, v in overall_count_dict.items():
            print(k)
            l = list()
            for element in v.values():
                for key, value in element.items():
                    l.extend([key] * value)

            median = np.median(l)
            print(f"\t{median}")
            if median < min_median[0]:
                min_median = (median, k)
            if median > max_median[0]:
                max_median = (median, k)

            if k in ["Wang2023", "Penn2022", "Lui2019", "Berry2021_A", "Berry2021_B", "Valesano2020_Vic", "826.9078947368421", "Berry2021_B_Yam", "Southgate2019", "Valesano2020_Yam"]:
                in_vivo.extend(l)
            else:
                in_vitro.extend(l)
        
        print("")
        print(f"in vivo:\t{np.mean(in_vivo)}")
        print(f"in vitro:\t{np.mean(in_vitro)}")
        print(np.mean(in_vivo) - np.mean(in_vitro))
        print("")

        print(f"Min median: {min_median[1]}\t{min_median[0]}")
        print(f"Max median: {max_median[1]}\t{max_median[0]}")

    return overall_count_dict
    

def length_distribution_violinplot(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        creates a violinplot that shows the length distribution of the DelVGs.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None    
    '''
    dfs, dfnames = sort_datasets_by_type(dfs, dfnames, cutoff=40)
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
        
        # also add the individual points as a scatter plot
        for i, d in enumerate(plot_list):
            y_p = np.random.uniform(i+1-0.3, i+1+0.3, len(d))
            plt.scatter(y_p, d, c="darkgrey", s=2, zorder=0)

        # plot data
        axs.violinplot(plot_list, position_list, points=1000, showextrema=False, showmedians=True)
        axs.set_xticks(range(1, len(dfnames)+1))
        axs.set_xticklabels(labels, rotation=90)
        axs.set_ylim(bottom=0, top=2500)
        axs.set_ylabel("DelVG sequence length (nt)")

        save_path = os.path.join(RESULTSPATH, folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"{s}_length_del_violinplot.png"), dpi=300)
        plt.close()


def calc_start_end_lengths(dfs: list, dfnames: list, thresh: int=300)-> Tuple[list, list]:
    '''
        calculates the difference of the start and end lengths of the DelVG RNA
        sequences (3' = start, 5' = end).
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
        sequences as a violinplot (3' = start, 5' = end).
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param folder: defines where to save the results
    
        :return: None  
    '''
    fig, axs = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
    thresh = 300
    plot_list, labels = calc_start_end_lengths(dfs, dfnames, thresh)

    position_list = np.arange(0, len(dfs))
    violin_parts = axs.violinplot(plot_list, position_list, showextrema=False, points=1000, showmeans=True, vert=True)
    for pc in violin_parts["bodies"]:
        pc.set_edgecolor("black")

    for i, d in enumerate(plot_list):
        y_p = np.random.uniform(i-0.3, i+0.3, len(d))
        plt.scatter(y_p, d, c="darkgrey", s=2, zorder=0)

    axs.set_xticks(position_list)
    labels = [f"{l}   " for l in labels]
    axs.set_xticklabels(labels, rotation=90)
    axs.set_xlim(left=-0.5, right=len(dfs)-0.5)
    axs.set_yticks(range(-300, 301, 150))
    axs.set_ylabel("3'-end length - 5'-end length              ")

    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "diff_start_end_violinplot.png"), dpi=300)
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


def nucleotide_pair_table(dfs: list, dfnames: list, folder: str="general_analysis")-> None:
    '''
        calcualte the motifs of specified length before start and end of
        deletion site.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
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
            s_motif = seq[s-2:s+2]
            e_motif = seq[e-(2+1):e+(2-1)]
            s_motifs.append(s_motif[:2])
            e_motifs.append(e_motif[:2])

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
    results_df.to_csv(os.path.join(save_path, "deletion_site_motif.csv"), index=False)


if __name__ == "__main__":
    plt.style.use("seaborn")
    
    dfnames = get_dataset_names(cutoff=40)
    dfs, _ = load_all(dfnames)

    plot_distribution_over_segments(dfs, dfnames)
    calculate_deletion_shifts(dfs, dfnames)
    length_distribution_violinplot(dfs, dfnames)
    plot_nucleotide_ratio_around_deletion_junction_heatmaps(dfs, dfnames)
    plot_direct_repeat_ratio_heatmaps(dfs, dfnames)
    diff_start_end_lengths(dfs, dfnames)
    nucleotide_pair_table(dfs, dfnames)
    