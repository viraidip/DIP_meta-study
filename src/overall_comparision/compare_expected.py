'''

'''
import os
import sys

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from matplotlib.ticker import FixedLocator, FixedFormatter

sys.path.insert(0, "..")
from utils import load_all
from utils import get_sequence, count_direct_repeats_overall, include_correction, get_p_value_symbol, create_nucleotide_ratio_matrix, plot_heatmap
from utils import SEGMENTS, RESULTSPATH, NUCLEOTIDES, CMAP


def plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs, dfnames, expected_dfs, name: str=""):
    '''
        Plot difference of expected vs observed nucleotide enrichment around deletion junctions as heatmap.

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data.
                                        data should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        expected_df (pandas.DataFrame): The DataFrame containing the expected data.
                                        data should be preprocessed with sequence_df(df)
        col (str, optional): The column name in the DataFrames that contains the sequences of interest. 
                             Default is "seq_around_deletion_junction".
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
        val_labels = list()
        for dfname, df, expected_df in zip(dfnames, dfs, expected_dfs):
            df = df.reset_index()
            probability_matrix = create_nucleotide_ratio_matrix(df, "seq_around_deletion_junction")
            n_samples = len(df)
            expected_probability_matrix = create_nucleotide_ratio_matrix(expected_df, "seq_around_deletion_junction")
            n_samples2 = len(expected_df)
            for j in probability_matrix.index:
                x.append(j)
                y.append(dfname)
                
                p1 = probability_matrix.loc[j,nuc]
                p2 = expected_probability_matrix.loc[j,nuc]
                
                if n_samples < n_samples2:
                    n_s = n_samples
                else:
                #    p1, p2 = p2, p1
                    n_s = n_samples2

                n_s = min(n_s, 1000)
                n_samples2 = min(n_samples2, 1000)

                test_array = np.concatenate((np.ones(int(n_s * p1)), np.zeros(int(n_s - n_s * p1))))
                test_array2 = np.concatenate((np.ones(int(n_samples2 * p2)), np.zeros(int(n_samples2 - n_samples2 * p2))))
                # perform an ANOVA as done in Alaji2021
                pval =  stats.f_oneway(test_array, test_array2).pvalue

                diff = p1 - p2
                vals.append(diff)
                if pval < 0.00001:
                    pval_symbol = "**"
                elif pval < 0.0001:
                    pval_symbol = "*"
                else:
                    pval_symbol = ""
                val_labels.append(pval_symbol)
                
        m = abs(min(vals)) if abs(min(vals)) > max(vals) else max(vals)
        axs[i] = plot_heatmap(x,y,vals, axs[i], format=".1e", cbar=True, vmin=-m, vmax=m, cbar_kws={"pad": 0.01})
        for v_idx, val_label in enumerate(axs[i].texts):
            val_label.set_text(val_labels[v_idx])
            val_label.set_size(10)
        axs[i].set_title(f"{NUCLEOTIDES[nuc]}")
        axs[i].set_ylabel("")
        axs[i].set_yticks([ytick + 0.5 for ytick in range(len(dfnames))])
        axs[i].set_xlabel("position")  
        axs[i].set_xticks([xtick - 0.5 for xtick in probability_matrix.index])
        
        quarter = len(probability_matrix.index) // 4
        indexes = [pos for pos in range(1, quarter * 2 + 1)]
        if i % 2 == 0:
            axs[i].set_yticklabels([f"{dfname} ({len(df)})" for dfname,df in zip(dfnames,dfs)])
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

    if name != "":
        filename = f"nuc_occ_diff_{name}.png"
    else:
        filename = f"nuc_occ_diff.png"

    save_path = os.path.join(RESULTSPATH, "compare_expected", filename)
    plt.savefig(save_path)
    plt.close()


def plot_expected_vs_observed_direct_repeat_heatmaps(dfs: list, dfnames: list, expected_dfs: list, compared: str, name: str="")-> None:
    '''
        Plot difference of expected vs observed nucleotide enrichment around deletion junctions as heatmap.

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data.
                                        data should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        expected_df (pandas.DataFrame): The DataFrame containing the expected data.
                                        data should be preprocessed with sequence_df(df)
        col (str, optional): The column name in the DataFrames that contains the sequences of interest. 
                             Default is "seq_around_deletion_junction".

    :return: None
    '''
    fig, axs = plt.subplots(figsize=(10, len(dfs)/2))

    x = list()
    y = list()
    vals = list()
    val_labels = list()
        
    # calculate direct repeats
    for dfname, df, expected_df in zip(dfnames, dfs, expected_dfs):
        final_d = dict()
        expected_final_d = dict()

        for s in SEGMENTS:
            df_s = df[df["Segment"] == s]
            expected_df_s = expected_df[expected_df["Segment"] == s]
            n_samples = len(df_s)
            if n_samples == 0:
                continue

            seq = get_sequence(df_s["Strain"].unique()[0], s)
            
            counts, _ = count_direct_repeats_overall(df_s, seq)
            if name == "":
                counts = include_correction(counts)
            
            for k, v in counts.items():
                if k in final_d:
                    final_d[k] += v
                else:
                    final_d[k] = v

            expected_counts, _ = count_direct_repeats_overall(expected_df_s, seq)
            for k, v in expected_counts.items():
                if k in expected_final_d:
                    expected_final_d[k] += v
                else:
                    expected_final_d[k] = v

        final = np.array(list(final_d.values()))
        expected_final = np.array(list(expected_final_d.values()))
        f_obs = final/final.sum()
        f_exp = expected_final/expected_final.sum()
        _, pvalue = stats.chisquare(f_obs, f_exp)

        symbol = get_p_value_symbol(pvalue)
        x.extend(final_d.keys())
        y.extend([f"{dfname} ({len(df)}) {symbol}" for _ in range(6)])
        vals.extend(final/final.sum() - expected_final/expected_final.sum())

        for f_ob, f_ex, n_samples in zip(f_obs, f_exp, final_d.values()):
            if int(n_samples) == 0:
                pval_symbol = ""
            else:
                pvalue = stats.binomtest(int(n_samples * f_ob), int(n_samples), f_ex).pvalue
                if pvalue < 0.00001:
                    pval_symbol = "**"
                elif pvalue < 0.0001:
                    pval_symbol = "*"
                else:
                    pval_symbol = ""
            val_labels.append(pval_symbol)

    m = abs(min(vals)) if abs(min(vals)) > max(vals) else max(vals)
    axs = plot_heatmap(x,y,vals, axs, vmin=-m, vmax=m, cbar=True, format=".5f")
    axs.set_title(f"direct repeat ratio difference ({compared})")
    axs.set_ylabel("")
    axs.set_xlabel("direct repeat length")

    for v_idx, val_label in enumerate(axs.texts):
        val_label.set_text(f"{val_label.get_text()}\n{val_labels[v_idx]}")

    x_ticks = axs.get_xticklabels()
    label = x_ticks[-2].get_text()
    x_ticks[-1].set_text(f"> {label}")
    axs.set_xticklabels(x_ticks)
    fig.tight_layout()

    if name != "":
        filename = f"dir_rep_diff_{name}.png"
    else:
        filename = f"dir_rep_diff.png"

    save_path = os.path.join(RESULTSPATH, "compare_expected", filename)
    plt.savefig(save_path)
    plt.close()


def direct_repeat_composition(dfs: list, dfnames: list, expected_dfs: list, name: str=""):
    '''
    
    '''
    plt.set_cmap(CMAP) 
    # calculate direct repeats
    for dfname, df, expected_df in zip(dfnames, dfs, expected_dfs):
        final_d = dict()
        expected_final_d = dict()
        strain = df["Strain"].unique()[0]

        for s in SEGMENTS:
            df_s = df[df["Segment"] == s]
            expected_df_s = expected_df[expected_df["Segment"] == s]
            n_samples = len(df_s)
            if n_samples == 0:
                continue

            seq = get_sequence(strain, s)
            
            # TODO: maybe include a correction of the direct repeat sequences ???
            _ , dr_sequences = count_direct_repeats_overall(df_s, seq)          
            if "_" in dr_sequences:
                del dr_sequences["_"]
            final_d[s] = dr_sequences

            _ , exp_dr_sequences = count_direct_repeats_overall(expected_df_s, seq)
            if "_" in exp_dr_sequences:
                del exp_dr_sequences["_"]
            expected_final_d[s] = exp_dr_sequences

        # make a bar plot with direct repeats lengths 1, 2, ,3 ,4, >4
        fig, axs = plt.subplots(figsize=(5, 5))

        nuc_counts_d = dict({"A": dict({1: 0, 2: 0, 3: 0, 4: 0, 5: 0}),
                            "C": dict({1: 0, 2: 0, 3: 0, 4: 0, 5: 0}),
                            "G": dict({1: 0, 2: 0, 3: 0, 4: 0, 5: 0}),
                            "U": dict({1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
                            })
        sum_counts_d = dict({1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
        for seq_dict in final_d.values():
            for k, v in seq_dict.items():
                if len(k) > 4:
                    l = 5
                else:
                    l = len(k)
                
                for n in NUCLEOTIDES.keys():
                    nuc_counts_d[n][l] += k.count(n) * v
                sum_counts_d[l] += l * v

        bar_width = 0.7
        bottom = np.zeros(5)

        for n in NUCLEOTIDES:
            y = np.array(list(nuc_counts_d[n].values())) / np.array(list(sum_counts_d.values()))
            axs.bar(nuc_counts_d[n].keys(), y, bar_width, label=n, bottom=bottom)
            bottom += y

        axs.set_title("nucleotide distribution in direct repeat sequences")
        axs.set_ylabel("frequency")
        axs.set_xlabel("direct repeat length")
        plt.gca().xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5]))
        sum_counts_d[">4"] = sum_counts_d.pop(5)
        labels = [f"{k} (n={v})" for k, v in sum_counts_d.items()]
        plt.gca().xaxis.set_major_formatter(FixedFormatter(labels))
        fig.tight_layout()

        if name != "":
            filename = f"nucleotides_direct_repeats_{name}_{dfname}.png"
        else:
            filename = f"nucleotides_direct_repeats_{dfname}.png"

        save_path = os.path.join(RESULTSPATH, "compare_expected", filename)
        plt.savefig(save_path)
        plt.close()

        # make a bar plot to compare orig. data, expected data, composition of full sequence (so I have two references)
        fig, axs = plt.subplots(figsize=(5, 5))

        comp_nuc_counts_d = dict({"A": dict({"orig": 0, "exp": 0, "full": 0}),
                                  "C": dict({"orig": 0, "exp": 0, "full": 0}),
                                  "G": dict({"orig": 0, "exp": 0, "full": 0}),
                                  "U": dict({"orig": 0, "exp": 0, "full": 0})
                                  })
        comp_sum_counts_d = dict({"orig": 0, "exp": 0, "full": 0})

        # observed data
        for n in NUCLEOTIDES:
            comp_nuc_counts_d[n]["orig"] = sum(nuc_counts_d[n].values())
            comp_sum_counts_d["orig"] += comp_nuc_counts_d[n]["orig"]

        # expected data
        for seq_dict in expected_final_d.values():
            for k, v in seq_dict.items():
                for n in NUCLEOTIDES:
                    comp_nuc_counts_d[n]["exp"] += k.count(n) * v
        for n in NUCLEOTIDES:
            comp_sum_counts_d["exp"] += comp_nuc_counts_d[n]["exp"]

        # full seq as reference    
        for s in SEGMENTS:
            seq = get_sequence(strain, s)
            for n in NUCLEOTIDES:
                comp_nuc_counts_d[n]["full"] += seq.count(n)
        for n in NUCLEOTIDES:
            comp_sum_counts_d["full"] += comp_nuc_counts_d[n]["full"]

        bar_width = 0.7
        bottom = np.zeros(3)
        for n in NUCLEOTIDES:
            y = np.array(list(comp_nuc_counts_d[n].values())) / np.array(list(comp_sum_counts_d.values()))
            axs.bar(comp_nuc_counts_d[n].keys(), y, bar_width, label=n, bottom=bottom)
            bottom += y
        
        axs.set_title(f"nucleotide distribution in direct repeat sequences (n={comp_sum_counts_d['orig']})")
        axs.set_ylabel("frequency")
        axs.set_xlabel("data")
        fig.tight_layout()

        if name != "":
            filename = f"nucleotides_direct_repeats_compared_{name}_{dfname}.png"
        else:
            filename = f"nucleotides_direct_repeats_compared_{dfname}.png"

        save_path = os.path.join(RESULTSPATH, "compare_expected", filename)
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")
    dfs, dfnames, expected_dfs = load_all(expected=True)

 #   plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs, dfnames, expected_dfs)
  #  plot_expected_vs_observed_direct_repeat_heatmaps(dfs, dfnames, expected_dfs "observed-expected")
    direct_repeat_composition(dfs, dfnames, expected_dfs)