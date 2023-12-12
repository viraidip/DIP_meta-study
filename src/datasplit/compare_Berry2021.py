'''

'''
import os
import sys

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


sys.path.insert(0, "..")
from utils import load_all, get_p_value_symbol
from utils import RESULTSPATH
from overall_comparision.general_analyses import calc_start_end_lengths


def compare_3_5_ends(dfs, dfnames):
    '''
    
    '''
    data = list()
    labels = list()
    
    data, labels = calc_start_end_lengths(dfs, dfnames)
        

    fig, axs = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    position_list = np.arange(0, 3)
    axs.violinplot(data, position_list, points=1000, showmedians=True)
    axs.set_xticks(position_list)
    axs.set_xticklabels(labels, rotation=90)
    axs.set_ylim(top=400)
    axs.set_xlabel("Dataset")
    axs.set_ylabel("Start-End sequence lengths")
    axs.set_title(f"Difference of start to end sequence lengths")
    
    def add_significance(l1, l2, axs, start, end, height):
        _, pvalue = stats.f_oneway(l1, l2)
        symbol = get_p_value_symbol(pvalue)
        if symbol != "":
            axs.plot([start, end], [height, height], lw=2, color='black')
            axs.text((start + end) / 2, height + 0.01, symbol, ha='center', va='bottom', color='black')

    add_significance(data[0], data[1], axs, 0, 1, 310)
    add_significance(data[0], data[2], axs, 0, 2, 340)
    add_significance(data[1], data[2], axs, 1, 2, 370)

    save_path = os.path.join(RESULTSPATH, "datasplits")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, "Berry2021_3_5_ends.png"))
    plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")

    dfnames = ["Berry2021_A", "Berry2021_B", "Berry2021_B_yamagata"]
    dfs, _ = load_all(dfnames)

    compare_3_5_ends(dfs, dfnames)