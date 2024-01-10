'''
    Compare the three datasets by Berry et al. 2021 based on their difference
    in 3' and 5' lengths.
'''
import os
import sys
import random

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_all, get_p_value_symbol, get_dataset_names
from utils import RESULTSPATH
from overall_comparision.general_analyses import calc_start_end_lengths


def compare_3_5_ends(dfs1: list, dfnames1: list, dfs2: list, dfnames2: list, categories: list)-> None:
    '''
        compare the given datasets in their difference of start and end
        sequence length.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`

        :return: None
    '''
    data1, _ = calc_start_end_lengths(dfs1, dfnames1)
    data2, _ = calc_start_end_lengths(dfs2, dfnames2)
    list1 = [item for sublist in data1 for item in sublist]
    list2 = [item for sublist in data2 for item in sublist]
    
    _, pvalue = stats.f_oneway(*data1)
    lab1 = f"IAV (pval. = {pvalue:.2e})"
    _, pvalue = stats.f_oneway(*data2)
    lab2 = f"IBV (p-value = {pvalue:.2e})"

    fig, axs = plt.subplots(1, 1, figsize=(2, 5), tight_layout=True)
    position_list = np.arange(0, 2)
    axs.violinplot([list1, list2], position_list, points=1000, showmedians=True)
    axs.set_xticks(position_list)
    axs.set_xticklabels([lab1, lab2], rotation=90)
    axs.set_ylim(top=430)
    axs.set_ylabel("5'-end length - 3'-end length")
    
    def add_significance(l1, axs, start, end, height):
        _, pvalue = stats.f_oneway(*l1)
        symbol = get_p_value_symbol(pvalue)
        if symbol != "":
            axs.plot([start, end], [height, height], lw=2, color='black')
            axs.text((start + end) / 2, height, f"{pvalue:.2e}", ha='center', va='bottom', color='black', fontsize=8)

    add_significance([list1, list2], axs, 0, 1, 310)

    save_path = os.path.join(RESULTSPATH, "datasplits")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "IAV_IBV_3_5_ends.png"))
    plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")

    IAV_dfnames = get_dataset_names(cutoff=0, selection="IAV")
    IAV_dfs, _ = load_all(IAV_dfnames)
    IBV_dfnames = get_dataset_names(cutoff=0, selection="IBV")
    IBV_dfs, _ = load_all(IBV_dfnames)
    categories = list(["IAV", "IBA"])

    compare_3_5_ends(IAV_dfs, IAV_dfnames, IBV_dfs, IBV_dfnames, categories)
