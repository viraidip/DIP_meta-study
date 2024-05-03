'''
    Compares the lengths of the DelVGs between the cultivation types.
'''
import os
import sys

import matplotlib.pyplot as plt

from collections import Counter

sys.path.insert(0, "..")
from utils import load_all, get_dataset_names, calc_cliffs_d
from utils import RESULTSPATH, CMAP
from overall_comparision.general_analyses import calc_DI_lengths


def compare_DI_lengths(dfs: list, dfnames: list, labels: str, analysis: str="")-> None:
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

    def calc_stats(x_1, x_2, s, e, h):
        cliffs_d = calc_cliffs_d(x_1, x_2)
        plt.plot([s, e], [h, h], lw=1, color='black')
        plt.plot([s, s], [h, h+0.0002], lw=1, color='black')
        plt.plot([e, e], [h, h+0.0002], lw=1, color='black')
        plt.text((s + e) / 2, h-0.0007, f"{cliffs_d:.2f}", ha='center', va='bottom', color='black')
        return
    
    dicts = [process_data(df, name) for df, name in zip(dfs, dfnames)]
    figsize = (6, 2) if analysis == "vivo_vitro" else (5, 2)
    cm = plt.get_cmap(CMAP)
    colors = [cm(0/8), cm(3/8), cm(1/8)]
    bins = 30
    for s in ["PB2", "PB1", "PA"]:
        lists = [[k for k, v in d[s].items() for _ in range(v)] for d in dicts]
        
        skip = False
        for l in lists:
            if len(l) < 1:
                skip = True
        if skip == True:
            continue
        
        plt.figure(figsize=figsize, tight_layout=True)
        for i, l in enumerate(lists):
            plt.hist(l, alpha=0.5, label=labels[i], bins=bins, density=True, color=colors[i])

        if len(lists) == 2:
            start = 1100 if analysis == "IAV_IBV" else 800
            calc_stats(lists[0], lists[1], start, 1650, 0.0038)

        elif len(lists) == 3:
            calc_stats(lists[0], lists[1], 550, 1100, 0.0025)
            calc_stats(lists[0], lists[2], 550, 1900, 0.0031)
            calc_stats(lists[1], lists[2], 1100, 1900, 0.0038)
        
        plt.ylim(0, 0.005)
        plt.yticks([0, 0.0025, 0.005])
        plt.xticks([0, 500, 1000, 1500, 2000, 2500])
        plt.xlabel(f"DelVG sequence length for {s} (nts.)")
        plt.ylabel("Probability density         ")
        plt.legend(loc="upper center", ncol=3)

        save_path = os.path.join(RESULTSPATH, "datasplits")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{s}_{analysis}.png"))
        plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")

# in vitro against in vivo datasets
    vitro_dfnames = get_dataset_names(cutoff=40, selection="in vitro")
    vitro_dfs, _ = load_all(vitro_dfnames)
    vivo_dfnames = get_dataset_names(cutoff=40, selection="in vivo mouse")
    vivo_dfs, _ = load_all(vivo_dfnames)
    human_dfnames = get_dataset_names(cutoff=40, selection="in vivo human")
    human_dfs, _ = load_all(human_dfnames)

    dfs = [vitro_dfs, vivo_dfs, human_dfs]
    dfnames = [vitro_dfnames, vivo_dfnames, human_dfnames]
    labels = ["in vitro", "in vivo mouse", "in vivo human"]
    compare_DI_lengths(dfs, dfnames, labels, analysis="vivo_vitro")

# in vitro against in vivo human
    dfs = [vitro_dfs, human_dfs]
    dfnames = [vitro_dfnames, human_dfnames]
    labels = ["in vitro", "in vivo human"]
    compare_DI_lengths(dfs, dfnames, labels, analysis="vivo_vitrohuman")

# all IAV against all IBV datasets
    IAV_dfnames = get_dataset_names(cutoff=40, selection="IAV")
    IAV_dfs, _ = load_all(IAV_dfnames)
    IBV_dfnames = get_dataset_names(cutoff=40, selection="IBV")
    IBV_dfs, _ = load_all(IBV_dfnames)

    dfs = [IAV_dfs, IBV_dfs]
    dfnames = [IAV_dfnames, IBV_dfnames]
    labels = ["IAV", "IBV"]
    compare_DI_lengths(dfs, dfnames, labels, analysis="IAV_IBV")

# in vitro all IAV against BLEE and Sheng
    vitro_iav_dfs = list()
    vitro_iav_dfnames = list()
    vitro_ibv_dfs = list()
    vitro_ibv_dfnames = list()
    for df, dfname in zip(vitro_dfs, vitro_dfnames):
        if dfname in ["Alnaji2019_BLEE", "Sheng2018"]:
            vitro_ibv_dfs.append(df)
            vitro_ibv_dfnames.append(dfname)
        else:
            vitro_iav_dfs.append(df)
            vitro_iav_dfnames.append(dfname)

    dfs = [vitro_iav_dfs, vitro_ibv_dfs]
    dfnames = [vitro_iav_dfnames, vitro_ibv_dfnames]
    labels = ["IAV in vitro", "IBV in vitro"]
    compare_DI_lengths(dfs, dfnames, labels, analysis="vitro_IAV")

# in vivo human all IBV against Berry A
    vivo_iav_dfs = list()
    vivo_iav_dfnames = list()
    vivo_ibv_dfs = list()
    vivo_ibv_dfnames = list()
    for df, dfname in zip(human_dfs, human_dfnames):
        if dfname == "Berry2021_A":
            vivo_iav_dfs.append(df)
            vivo_iav_dfnames.append(dfname)
        else:
            vivo_ibv_dfs.append(df)
            vivo_ibv_dfnames.append(dfname)

    dfs = [vivo_iav_dfs, vivo_ibv_dfs]
    dfnames = [vivo_iav_dfnames, vivo_ibv_dfnames]
    labels = ["IAV in vivo human", "IBV in vivo human"]
    compare_DI_lengths(dfs, dfnames, labels, analysis="vivo_IBV")

# IAV vitro vs vivo human
    dfs = [vitro_iav_dfs, vivo_iav_dfs]
    dfnames = [vitro_iav_dfnames, vitro_ibv_dfnames]
    labels = ["IAV in vitro", "IAV in vivo human"]
    compare_DI_lengths(dfs, dfnames, labels, analysis="IAV_vitro_vivo")

# IBV vitro vs vivo human
    dfs = [vitro_ibv_dfs, vivo_ibv_dfs]
    dfnames = [vivo_iav_dfnames, vivo_ibv_dfnames]
    labels = ["IBV in vitro", "IBV in vivo human"]
    compare_DI_lengths(dfs, dfnames ,labels, analysis="IBV_vitro_vivo")
