'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_single_dataset, preprocess
from utils import SEGMENTS, RESULTSPATH, DATAPATH, CUTOFF, CMAP, SEGMENT_DICTS


def load_mendes2021_rsc(name: str)-> dict:
    '''
        :param de_novo: if True only de novo candidates are taken
        :param long_dirna: if True loads data set that includes long DI RNA
                           candidates
        :param by_time: if True loads the dataset split up by timepoints

        :return: dictionary with one key, value pair
    '''
    if name == "v12enriched":
        filename = "Virus-1-2_enriched_junctions.tsv"
    elif name == "v21depleted":
        filename = "Virus-2-1_depleted_junctions.tsv"
    
    file_path = os.path.join(DATAPATH, "RSC_estimation", filename)
    data = pd.read_csv(file_path,
                            header=0,
                            na_values=["", "None"],
                            keep_default_na=False,
                            sep="\t")
    
    return data


def diff_start_end_lengths(dfs, dfnames, folder: str="validation_estimation")-> None:
    '''

    '''
    fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    plot_list = list()
    position_list = np.arange(0, len(dfs))
    labels = list()

    for df, dfname in zip(dfs, dfnames):
        df["End_L"] = df["full_seq"].str.len() - df["End"]
        l = (df["Start"] - df["End_L"]).to_list()
        thresh = 300
        l = [x for x in l if x <= thresh]
        l = [x for x in l if x >= -thresh]
        plot_list.append(l)
        labels.append(f"{dfname} (n={df.shape[0]})")

    axs.violinplot(plot_list, position_list, points=1000, showmedians=True)
    axs.set_xticks(position_list)
    axs.set_xticklabels(labels, rotation=90)
    axs.set_xlabel("Dataset")
    axs.set_ylabel("Start-End sequence lengths")
    axs.set_title(f"Difference of start to end sequence lengths (threshold={thresh})")

    save_path = os.path.join(RESULTSPATH, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "mendes_diff_start_end_violinplot.png"))
    plt.close()


if __name__ == "__main__":
    plt.style.use("seaborn")
    RESULTSPATH = os.path.dirname(RESULTSPATH)
    dfs = list()
    dfnames = list()

    strain = "WSN"
    ### Mendes 2021 ###
    v12enriched = load_single_dataset("Mendes2021", "SRR15720521", dict({s: s for s in SEGMENTS}))
    dfs.append(preprocess(strain, v12enriched, CUTOFF))
    dfnames.append("v12enriched")

    orig_mendes = load_mendes2021_rsc("v12enriched")
    dfs.append(preprocess(strain, orig_mendes, CUTOFF))
    dfnames.append("orig. v12enriched") 

    v21depleted = load_single_dataset("Mendes2021", "SRR15720526", dict({s: s for s in SEGMENTS}))
    dfs.append(preprocess(strain, v21depleted, CUTOFF))
    dfnames.append("v21depl")
  
    orig_mendes = load_mendes2021_rsc("v21depleted")
    dfs.append(preprocess(strain, orig_mendes, CUTOFF))
    dfnames.append("orig. v21depl")

    diff_start_end_lengths(dfs, dfnames)
