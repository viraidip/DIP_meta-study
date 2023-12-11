'''

'''
import os
import sys

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from collections import Counter

sys.path.insert(0, "..")
from utils import load_all, get_dataset_names, get_p_value_symbol
from utils import SEGMENTS
from overall_comparision.general_analyses import calc_DI_lengths, calc_start_end_lengths


def compare_DI_lengths(a_dfs, a_dfnames, a_label, b_dfs, b_dfnames, b_label):
    '''
    
    '''
    a_lengths_dict = calc_DI_lengths(a_dfs, a_dfnames)
    a_dict = dict({"PB2": Counter(), "PB1": Counter(), "PA": Counter(), "HA": Counter(), "NP": Counter(), "NA": Counter(), "M": Counter(), "NS": Counter()})
    for d in a_lengths_dict.values():
        for s in a_dict.keys():
            a_dict[s] += Counter(d[s])

    b_lengths_dict = calc_DI_lengths(b_dfs, b_dfnames)   
    b_dict = dict({"PB2": Counter(), "PB1": Counter(), "PA": Counter(), "HA": Counter(), "NP": Counter(), "NA": Counter(), "M": Counter(), "NS": Counter()})
    for d in b_lengths_dict.values():
        for s in b_dict.keys():
            b_dict[s] += Counter(d[s])

    for s in SEGMENTS:
        x_a = [key for key, value in a_dict[s].items() for _ in range(value)]
        x_b = [key for key, value in b_dict[s].items() for _ in range(value)]

        _, pvalue = stats.f_oneway(x_a, x_b)
        print(pvalue)
        symbol = get_p_value_symbol(pvalue)

        plt.hist(x_a, alpha=0.5, label="in vitro", bins=20, density=True)
        plt.hist(x_b, alpha=0.5, label="in vivo", bins=20, density=True)
        plt.legend()
        plt.title(f"{s} {symbol}")

        plt.show()
        

def compare_3_5_ends(a_dfs, a_dfnames, b_dfs, b_dfnames, labels):
    '''
    
    '''
    a_list, a_labels = calc_start_end_lengths(a_dfs, a_dfnames)
    b_list, b_labels = calc_start_end_lengths(b_dfs, b_dfnames)

    fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    position_list = np.arange(0, len(a_dfs) + len(b_dfs))
    axs.violinplot(a_list+b_list, position_list, points=1000, showmedians=True)
    axs.set_xticks(position_list)
    axs.set_xticklabels(a_labels+b_labels, rotation=90)
    axs.set_xlabel("Dataset")
    axs.set_ylabel("Start-End sequence lengths")
    axs.set_title(f"Difference of start to end sequence lengths")

    plt.show()
    plt.close()

    a_full_list = [item for sublist in a_list for item in sublist]
    b_full_list = [item for sublist in b_list for item in sublist]

    _, pvalue = stats.f_oneway(a_full_list, b_full_list)
    print(pvalue)
    symbol = get_p_value_symbol(pvalue)

    fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    position_list = np.arange(0, 2)
    axs.violinplot([a_full_list, b_full_list], position_list, points=1000, showmedians=True)
    axs.set_xticks(position_list)
    axs.set_xticklabels(labels, rotation=90)
    axs.set_xlabel("Dataset")
    axs.set_ylabel("Start-End sequence lengths")
    axs.set_title(f"Difference of start to end sequence lengths ({symbol})")
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")

    vitro_dfnames = get_dataset_names(cutoff=0, selection="in vitro")
    vitro_dfs, _ = load_all(vitro_dfnames)

    vivo_dfnames = get_dataset_names(cutoff=0, selection="in vivo mouse")
    vivo_dfs, _ = load_all(vivo_dfnames)
    
    patient_dfnames = get_dataset_names(cutoff=0, selection="in vivo human")
    patient_dfs, _ = load_all(patient_dfnames)

    compare_DI_lengths(vitro_dfs, vitro_dfnames, "in vitro", vivo_dfs, vivo_dfnames, "in vivo human")
    compare_3_5_ends(vitro_dfs, vitro_dfnames, patient_dfs, patient_dfnames, ["in vitro", "in vivo humans"])
