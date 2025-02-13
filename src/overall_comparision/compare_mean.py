'''
    Performs analyses of direct repeats and nucleotide enrichment of the
    datasets while comparing the results to the mean.
'''
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_all
from utils import get_sequence, count_direct_repeats_overall, create_nucleotide_ratio_matrix, plot_heatmap, get_dataset_names, get_eta_squared
from utils import SEGMENTS, RESULTSPATH, NUCLEOTIDES


def plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs: list, dfnames: list, compared: str, folder: str="compare_expected")-> None:
    '''
        plot difference of expected vs observed nucleotide enrichment around
        deletion junctions as heatmap.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param expected_dfs: The list of DataFrames containing the expected
            data, preprocessed with sequence_df(df)
        :param compared: defines in title what data is compared
        :param folder: defines where to save the results
    
        :return: None
    '''
    fig, axs = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)
    axs = axs.flatten()

    mean_matrix = pd.DataFrame(columns=NUCLEOTIDES.keys())
    count_matrizes_rel = list()
    for df in dfs:
        df = df.reset_index()
        count_matrix = create_nucleotide_ratio_matrix(df, "seq_around_deletion_junction")
        if len(mean_matrix) == 0:
            mean_matrix = count_matrix.copy()
        else:
            mean_matrix += count_matrix
        count_matrizes_rel.append(count_matrix)

    mean_matrix_rel = mean_matrix / len(dfs)

    for i, nuc in enumerate(NUCLEOTIDES.keys()):
        x = list()
        y = list()
        vals = list()
        val_labels = list()
        for dfname, df, count_matrix_rel in zip(dfnames, dfs, count_matrizes_rel):
            n_samples = len(df)

            for j in count_matrix_rel.index:
                x.append(j)
                y.append(dfname)

                p1 = count_matrix_rel.loc[j,nuc]
                p2 = mean_matrix_rel.loc[j,nuc]
                vals.append(p1 - p2)

                test_array = np.concatenate((np.ones(int(n_samples * p1)), np.zeros(int(n_samples - n_samples * p1))))
                test_array2 = np.concatenate((np.ones(int(n_samples * p2)), np.zeros(int(n_samples - n_samples * p2))))
                # perform an ANOVA as done in Alaji2021
                pval = stats.f_oneway(test_array, test_array2).pvalue
                res = stats.kruskal(test_array, test_array2)
                pval = res.pvalue

                # calcualte effect size (eta squared)
                # from https://rpkgs.datanovia.com/rstatix/reference/kruskal_effsize.html
                # "0.01- < 0.06 (small effect), 0.06 - < 0.14 (moderate effect) and >= 0.14 (large effect)."
                if pval < 0.05:
                    eta = get_eta_squared(res.statistic, 2, n_samples)
                    if eta > 0.06:
                        text = f"{eta:.2f}"
                        text = text[1:]
                        
                    else:
                        text = ""
                else:
                    text = ""

                val_labels.append(text)

        if len(vals) != 0:        
            m = abs(min(vals)) if abs(min(vals)) > max(vals) else max(vals)
        else:
            m = 0
        axs[i] = plot_heatmap(x,y,vals, axs[i], format=".1e", cbar=True, vmin=-m, vmax=m, cbar_kws={"pad": 0.01})
        thres = 0.2 if i in [0, 2] else 0.15
        for v_idx, val_label in enumerate(axs[i].texts):
            val_label.set_text(val_labels[v_idx])
            val_label.set_size(8)
            if abs(vals[v_idx]) > abs(thres):
                val_label.set_color("white")
            else:
                val_label.set_color("black")
        axs[i].set_title(f"{NUCLEOTIDES[nuc]}")
        axs[i].set_ylabel("")
        axs[i].set_yticks([ytick + 0.5 for ytick in range(len(dfnames))])
        axs[i].set_xlabel("")  
        axs[i].set_xticks([xtick - 0.5 for xtick in mean_matrix_rel.index])
        
        quarter = len(mean_matrix_rel.index) // 4
        indexes = [pos for pos in range(1, quarter * 2 + 1)]
        if i % 2 == 0:
            axs[i].set_yticklabels([f"{dfname} ({len(df)})" for dfname,df in zip(dfnames,dfs)])
        else:
            axs[i].set_yticklabels([])
        if i < 2:
            axs[i].xaxis.set_ticks_position("top")
            axs[i].xaxis.set_label_position("top")
        axs[i].set_xticklabels(indexes + indexes, rotation=0)
        xlabels = axs[i].get_xticklabels()
        for x_idx, xlabel in enumerate(xlabels):
            if x_idx < quarter or x_idx >= quarter * 3:
                xlabel.set_color("black")
                xlabel.set_fontweight("bold")
            else:
                xlabel.set_color("grey")   

    fig.suptitle("Enriched (red) and depleted (blue) nucleotides")
    fig.subplots_adjust(top=0.9)
    fig.tight_layout()
    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "nuc_occ_mean.png"), dpi=300)
    plt.close()


def plot_expected_vs_observed_direct_repeat_heatmaps(dfs: list, dfnames: list, compared: str, folder: str="compare_expected")-> None:
    '''
        plot difference of expected vs observed direct repeat ratios around
        deletion junctions as heatmap.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        :param expected_dfs: The list of DataFrames containing the expected
            data, preprocessed with sequence_df(df)
        :param compared: defines in title what data is compared
        :param folder: defines where to save the results
    
        :return: None
    '''
    mean_data = 0
    dir_rep_counts = list()
    # calculate direct repeats
    for dfname, df, in zip(dfnames, dfs):
        final_d = dict()
        for s in SEGMENTS:
            df_s = df[df["Segment"] == s]
            n_samples = len(df_s)
            if n_samples == 0:
                continue
            seq = get_sequence(df_s["Strain"].unique()[0], s)            
            counts, _ = count_direct_repeats_overall(df_s, seq)
            for k, v in counts.items():
                if k in final_d:
                    final_d[k] += v
                else:
                    final_d[k] = v

        final = np.array(list(final_d.values()))
        dir_rep_counts.append(final)

        final_rel = final/final.sum()
        if type(mean_data) == int:
            mean_data = final_rel.copy()
        else:
            mean_data += final_rel

    mean_data = mean_data / len(dfnames)

    # plot results
    fig, axs = plt.subplots(figsize=(10, 7))
    x = list()
    y = list()
    vals = list()
    f_exp = np.array(list(mean_data))
    for dfname, df, counts in zip(dfnames, dfs, dir_rep_counts):
        final = np.array(counts)
        f_obs = final/final.sum()
    
        # Perform chi-squared test and if sign. p value calculate Cramer's V
        r, pvalue = stats.chisquare(final, f_exp * final.sum()) # as in Boussier et al. 2020
        if pvalue < 0.05:
            v_data = np.stack((final, np.rint(f_exp * final.sum())), axis=1).astype(int)
            v_data = v_data[~np.all(v_data == [0, 0], axis=1)]
            cramers_v = stats.contingency.association(v_data)
            cramers_v = round(cramers_v, 2)
        else:
            cramers_v = "n.a."

        x.extend(final_d.keys())
        y.extend([f"{dfname} (n={len(df)}) V={cramers_v}" for _ in range(6)])
        vals.extend((f_obs - f_exp) * 100)

    m = abs(min(vals)) if abs(min(vals)) > max(vals) else max(vals)
    axs = plot_heatmap(x,y,vals, axs, vmin=-m, vmax=m, cbar=True, format=".1f")
    axs.set_title(f"Difference in direct repeat distribution ({compared})")
    axs.set_ylabel("")
    axs.set_xlabel("Direct repeat length")
    for v_idx, val_label in enumerate(axs.texts):
        val_label.set_text(f"{val_label.get_text()}")
    x_ticks = axs.get_xticklabels()
    label = x_ticks[-2].get_text()
    x_ticks[-1].set_text(f"> {label}")
    axs.set_xticklabels(x_ticks)
    fig.tight_layout()
    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "dir_rep_mean.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")

    dfnames = get_dataset_names(cutoff=40)
    dfs, _ = load_all(dfnames, expected=False)

    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs, dfnames, "observed-mean")
    plot_expected_vs_observed_direct_repeat_heatmaps(dfs, dfnames, "observed-mean")
