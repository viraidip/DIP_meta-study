'''
    Compare the three datasets by Berry et al. 2021 based on their difference
    in 3' and 5' lengths.
'''
import os
import sys
import warnings
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_all, get_dataset_names, calc_cliffs_d
from utils import RESULTSPATH
from overall_comparision.general_analyses import calc_start_end_lengths


def compare_iav_ibv(dfs1: list, dfnames1: list, dfs2: list, dfnames2: list, categories: list, analysis: str)-> None:
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
    plot_list = [list1, list2]
    
    fig, axs = plt.subplots(1, 1, figsize=(5, 1.5), tight_layout=True)
    position_list = np.arange(0, 2)
    violin_parts = axs.violinplot(plot_list, position_list, showextrema=False, points=1000, showmeans=True, vert=False)
    for pc in violin_parts["bodies"]:
        pc.set_edgecolor("black")

    for i, d in enumerate(plot_list):
        y_p = np.random.uniform(i-0.3, i+0.3, len(d))
        plt.scatter(d, y_p, c="darkgrey", s=2, zorder=0)

    axs.set_yticks(position_list)
    if analysis == "IAV_IBV":
        step = (22, 22)
    elif analysis == "vivo_vitro":
        step = (6, 6)
    elif analysis == "vitro_IAV":
        step = (12, 12)
    elif analysis == "vivo_IBV":
        step = (0, 0)
    elif analysis == "IAV_vitro_vivo":
        step = (2, 2)
    elif analysis == "IBV_vitro_vivo":
        step = (0, 0)

    axs.set_yticklabels([f"{' '*step[0]}{categories[0]} (n={len(plot_list[0])})", f"{' '*step[0]}{categories[1]} (n={len(plot_list[1])})"])

    axs.set_xlabel("5'-end length - 3'-end length")
    axs.set_xlim(right=340)
    
    cliffs_d = calc_cliffs_d(*plot_list)
    axs.plot([300, 300], [0, 1], lw=2, color='black')
    axs.text(315, 0.5, f"{cliffs_d:.2f}", ha='center', va='center', color='black', fontsize=8, rotation=270)
    axs.set_xticks([-300, -200, -100, 0, 100, 200, 300])

    save_path = os.path.join(RESULTSPATH, "datasplits")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{analysis}_3_5_ends.png"))
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
    plot_list, _ = calc_start_end_lengths(dfs, dfnames)

    plt.figure(figsize=(10, 9))
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
                cliffs_d = calc_cliffs_d(d1, d2)
                matrix[i][j] = abs(cliffs_d)
                color = "black" if abs(cliffs_d) > 0.4 else "white"
                plt.annotate(f"{cliffs_d:.2f}", xy=(j, i), color=color, ha='center', va='center', fontsize=10, fontweight='bold')

    warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
    plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04, location="right", label="Cliff's $\it{d}$ (absolute values)")
    plt.xticks(np.arange(len(dfnames)), [f"{n}    " for n in dfnames], rotation=90)
    plt.yticks(np.arange(len(dfnames)), [f"{n}    " for n in dfnames])
    plt.tight_layout()
    plt.grid(False)

    save_path = os.path.join(RESULTSPATH, "datasplits")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "cliffs_d_comparision_matrix.png"))
    plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")

    IAV_dfnames = get_dataset_names(cutoff=40, selection="IAV")
    IAV_dfs, _ = load_all(IAV_dfnames)
    IBV_dfnames = get_dataset_names(cutoff=40, selection="IBV")
    IBV_dfs, _ = load_all(IBV_dfnames)
    categories = ["IAV", "IBA"]
    compare_iav_ibv(IAV_dfs, IAV_dfnames, IBV_dfs, IBV_dfnames, categories, analysis="IAV_IBV")

####### further analysis
# in vitro against in vivo datasets
    vitro_dfnames = get_dataset_names(cutoff=40, selection="in vitro")
    vitro_dfs, _ = load_all(vitro_dfnames)
    human_dfnames = get_dataset_names(cutoff=40, selection="in vivo human")
    human_dfs, _ = load_all(human_dfnames)

    categories = ["in vitro", "in vivo human"]
    compare_iav_ibv(vitro_dfs, vitro_dfnames, human_dfs, human_dfnames, categories, analysis="vivo_vitro")

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

    categories = ["IAV in vitro", "IBV in vitro"]
    compare_iav_ibv(vitro_iav_dfs, vitro_iav_dfnames, vitro_ibv_dfs, vitro_ibv_dfnames, categories, analysis="vitro_IAV")

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

    categories = ["IAV in vivo human", "IBV in vivo human"]
    compare_iav_ibv(vivo_iav_dfs, vivo_iav_dfnames, vivo_ibv_dfs, vivo_ibv_dfnames, categories, analysis="vivo_IBV")

# IAV vitro vs vivo human
    categories = ["IAV in vitro", "IAV in vivo human"]
    compare_iav_ibv(vitro_iav_dfs, vitro_iav_dfnames, vivo_iav_dfs, vitro_ibv_dfnames, categories, analysis="IAV_vitro_vivo")

# IBV vitro vs vivo human
    categories = ["IBV in vitro", "IBV in vivo human"]
    compare_iav_ibv(vitro_ibv_dfs, vivo_iav_dfnames, vivo_ibv_dfs, vivo_ibv_dfnames, categories, analysis="IBV_vitro_vivo")


### comparision matrix for all datasets together
    dfnames = get_dataset_names(cutoff=40)
    dfs, _ = load_all(dfnames)
    create_comparision_matrix(dfs, dfnames)