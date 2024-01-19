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
from matplotlib.colors import LogNorm

sys.path.insert(0, "..")
from utils import load_all, get_p_value_symbol, get_dataset_names
from utils import RESULTSPATH
from overall_comparision.general_analyses import calc_start_end_lengths


def compare_iav_ibv(dfs1: list, dfnames1: list, dfs2: list, dfnames2: list, categories: list)-> None:
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


def compare_berry(dfs: list, dfnames: list)-> None:
    '''
        compare the given datasets in their difference of start and end
        sequence length.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`

        :return: None
    '''
    data, labels = calc_start_end_lengths(dfs, dfnames)
    fig, axs = plt.subplots(1, 1, figsize=(2, 5), tight_layout=True)
    position_list = np.arange(0, 3)
    axs.violinplot(data, position_list, points=1000, showmedians=True)
    axs.set_xticks(position_list)
    axs.set_xticklabels(labels, rotation=90)
    axs.set_ylim(top=430)
    axs.set_ylabel("5'-end length - 3'-end length")
    
    def add_significance(l1, l2, axs, start, end, height):
        _, pvalue = stats.f_oneway(l1, l2)
        print(pvalue)
        symbol = get_p_value_symbol(pvalue)
        if symbol != "":
            axs.plot([start, end], [height, height], lw=2, color='black')
            axs.text((start + end) / 2, height, f"{pvalue:.2e}", ha='center', va='bottom', color='black', fontsize=8)

    add_significance(data[0], data[1], axs, 0, 1, 310)
    add_significance(data[0], data[2], axs, 0, 2, 350)
    add_significance(data[1], data[2], axs, 1, 2, 390)

    save_path = os.path.join(RESULTSPATH, "datasplits")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "Berry2021_3_5_ends.png"))
    plt.close()


def create_comparision_matrix(dfs: list, dfnames: list):
    '''
        compare the given datasets pairwise in their difference of start and
        end sequence length by creating a matrix.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`

        :return: None
    '''
    plot_list, labels = calc_start_end_lengths(dfs, dfnames)

    plt.figure(figsize=(6, 7))
    plt.rc("font", size=20)
    # initialize an empty matrix
    matrix_size = len(plot_list)
    matrix = [[0] * matrix_size for _ in range(matrix_size)]
    # calculate the differences and populate the matrix
    for i, d1 in enumerate(plot_list):
        for j, d2 in enumerate(plot_list):
            if i == j:
                matrix[i][j] = np.nan
            else:
                _, pvalue = stats.f_oneway(d1, d2)
                matrix[i][j] = max(pvalue, 0.0000000001)
                text = get_p_value_symbol(pvalue)
                color = "black" if pvalue > 0.000001 else "white"
                plt.annotate(text, xy=(j, i), color=color, ha='center', va='center', fontsize=6, fontweight='bold')

    plt.imshow(matrix, cmap="viridis", interpolation="nearest", norm=LogNorm())
    plt.colorbar(fraction=0.046, pad=0.04, orientation="horizontal", location="top", label="p-value (logarithmic scale)")
    plt.xticks(np.arange(len(dfnames)), dfnames, rotation=90)
    plt.yticks(np.arange(len(dfnames)), dfnames)
    plt.tight_layout()
    plt.grid(False)

    save_path = os.path.join(RESULTSPATH, "datasplits")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "pvalue_matrix.png"))
    plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")

    ''' this is old
    IAV_dfnames = get_dataset_names(cutoff=0, selection="IAV")
    IAV_dfs, _ = load_all(IAV_dfnames)
    IBV_dfnames = get_dataset_names(cutoff=0, selection="IBV")
    IBV_dfs, _ = load_all(IBV_dfnames)
    categories = list(["IAV", "IBA"])
    compare_iav_ibv(IAV_dfs, IAV_dfnames, IBV_dfs, IBV_dfnames, categories)

    dfnames = ["Berry2021_A", "Berry2021_B", "Berry2021_B_Yam"]
    dfs, _ = load_all(dfnames)
    compare_berry(dfs, dfnames)
    '''

    dfnames = get_dataset_names(cutoff=50)
    dfs, _ = load_all(dfnames)
    create_comparision_matrix(dfs, dfnames)
