'''
    Compares the lengths of the DelVGs between the cultivation types.
'''
import os
import sys

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from collections import Counter

sys.path.insert(0, "..")
from utils import load_all, get_dataset_names, get_p_value_symbol
from utils import SEGMENTS, RESULTSPATH
from overall_comparision.general_analyses import calc_DI_lengths


def compare_DI_lengths(a_dfs: list, a_dfnames: list, a_label: str, b_dfs: list, b_dfnames: list, b_label: str, c_dfs: list, c_dfnames: list, c_label: str)-> None:
    '''
        compares the lengths of the DelVGs between three classes.
        :param a_dfs: list of datasets for class a
        :param a_dfnames: list of dataset names for class a
        :param a_label: label of class a
        :param b_dfs: list of datasets for class b
        :param b_dfnames: list of dataset names for class b
        :param b_label: label of class b
        :param c_dfs: list of datasets for class c
        :param c_dfnames: list of dataset names for class c
        :param c_label: label of class c

        :return: None
    '''
    def process_data(dfs, dfnames):
        lengths_dict = calc_DI_lengths(dfs, dfnames)
        final_d = dict({"PB2": Counter(), "PB1": Counter(), "PA": Counter(), "HA": Counter(), "NP": Counter(), "NA": Counter(), "M": Counter(), "NS": Counter()})
        for d in lengths_dict.values():
            for s in d.keys():
                final_d[s] += Counter(d[s])
        return final_d
    a_dict = process_data(a_dfs, a_dfnames)
    b_dict = process_data(b_dfs, b_dfnames)
    c_dict = process_data(c_dfs, c_dfnames)
    
    bins = 30
    for s in SEGMENTS:
        x_a = [key for key, value in a_dict[s].items() for _ in range(value)]
        x_b = [key for key, value in b_dict[s].items() for _ in range(value)]
        x_c = [key for key, value in c_dict[s].items() for _ in range(value)]
        if len(x_a) < 50 or len(x_b) < 50 or len(x_c) < 50:
            continue

        plt.figure(figsize=(6, 2))
        plt.hist(x_a, alpha=0.5, label=a_label, bins=bins, density=True)
        plt.hist(x_b, alpha=0.5, label=b_label, bins=bins, density=True)
        plt.hist(x_c, alpha=0.5, label=c_label, bins=bins, density=True)

        def calc_anova(x_1, x_2, s, e, h):
            data1, _ = np.histogram(x_1, bins=bins)
            data2, _ = np.histogram(x_2, bins=bins)
            _, pvalue = stats.f_oneway(data1, data2)
            symbol = get_p_value_symbol(pvalue)
            if symbol != "":
                plt.plot([s, e], [h, h], lw=1, color='black')
                plt.plot([s, s], [h, h-0.00005], lw=1, color='black')
                plt.plot([e, e], [h, h-0.00005], lw=1, color='black')
                plt.text((s + e) / 2, h-0.0001, symbol, ha='center', va='bottom', color='black', fontsize=8)

            return
        
        calc_anova(x_a, x_b, 550, 1000, 0.0041)
        calc_anova(x_a, x_c, 550, 1900, 0.0042)
        calc_anova(x_b, x_c, 1200, 1900, 0.0043)


        plt.ylim(0, 0.0045)
        plt.xlabel("DelVG sequence length")
        plt.ylabel("probability density")
        plt.legend(loc="upper center", ncol=3)

        save_path = os.path.join(RESULTSPATH, "datasplits")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{s}_vivo_vitro.png"))
        plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")

    vitro_dfnames = get_dataset_names(cutoff=0, selection="in vitro")
    vitro_dfs, _ = load_all(vitro_dfnames)
    vivo_dfnames = get_dataset_names(cutoff=0, selection="in vivo mouse")
    vivo_dfs, _ = load_all(vivo_dfnames)
    patient_dfnames = get_dataset_names(cutoff=0, selection="in vivo human")
    patient_dfs, _ = load_all(patient_dfnames)

    compare_DI_lengths(vitro_dfs, vitro_dfnames, "in vitro", vivo_dfs, vivo_dfnames, "in vivo mouse", patient_dfs, patient_dfnames, "in vivo human")

