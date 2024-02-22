'''
    Estimate which RSC is sufficient.
    Is done by comparing generated data to original data of the publications.
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple

sys.path.insert(0, "..")
from utils import load_single_dataset, load_dataset, join_data
from utils import RESULTSPATH, DATAPATH, SEGMENT_DICTS


def load_pelz2021_rsc()-> dict:
    '''
        Loads the data from Pelz et al. 2021 publication.

        :return: dictionary with strain name as key and data frame as value
    '''
    filename = "pelz_2021.xlsx"
    file_path = os.path.join(DATAPATH, "RSC_estimation", filename)
    data_dict = pd.read_excel(io=file_path,
                              sheet_name=None,
                              header=0,
                              na_values=["", "None"],
                              keep_default_na=False)
    return join_data(data_dict["PR8"])


def load_alnaji2021_rsc()-> dict:
    '''
        Loads the data set of Alnaji et al. 2021.

        :return: dictionary with strain name as key and data frame as value
    '''
    path = os.path.join(DATAPATH, "RSC_estimation", "Alnaji2021.xlsx")
    data = pd.read_excel(path, na_values=["", "None"], keep_default_na=False)
    return join_data(data)


def load_alnaji2019_rsc(strain)-> dict:
    '''
        Loads the data set of Alnaji et al. 2019.

        :return: dictionary with strain name as key and data frame as value
    '''
    file_path = os.path.join(DATAPATH, "RSC_estimation", f"Alnaji2019_{strain}.xlsx")
    data_dict = pd.read_excel(io=file_path,
                              sheet_name=None,
                              header=0,
                              na_values=["", "None"],
                              keep_default_na=False)
    df = data_dict[strain]
    return join_data(df)


def load_mendes2021_rsc()-> dict:
    '''
        Loads the data set of Mendes et al. 2021.
        :param name: indicates which dataset to load

        :return: dictionary with strain name as key and data frame as value
    '''
    file_path = os.path.join(DATAPATH, "RSC_estimation", "Mendes2021.tsv")
    data = pd.read_csv(file_path,
                            header=0,
                            na_values=["", "None"],
                            keep_default_na=False,
                            sep="\t")
    
    return join_data(data)


def load_lui2019_rsc(name: str)-> dict:
    '''
        Loads the data set of Lui et al. 2019.
        :param name: indicates which dataset to load

        :return: dictionary with strain name as key and data frame as value
    '''
    if name == "SMRT":
        filename = ""
    elif name == "illumina":
        filename = "Lui2019_illumina.csv"
    
    file_path = os.path.join(DATAPATH, "RSC_estimation", filename)
    data = pd.read_csv(file_path,
                            header=0,
                            na_values=["", "None"],
                            keep_default_na=False)
    
    data = join_data(data)
    return data


def compare_datasets(d1: pd.DataFrame, d2: pd.DataFrame, thresh: int=1)-> Tuple[float, int, int]:
    '''
        Calculate the intersection of two given datasets.
        :param d1: dataset 1
        :param d2: dataset 2
        :param thresh: Threshold for min number of count for each DelVG
        
        :return: Tuple
            fraction of intersecting DelVGs
            number of DelVGs in dataset 1
            number of DelVGs in dataset 2
    '''
    d1 = d1[d1["NGS_read_count"] >= thresh]
    d2 = d2[d2["NGS_read_count"] >= thresh]
    DI_sets = [set(d["Segment"] + "_" + d["Start"].astype(str) + "_" + d["End"].astype(str)) for d in [d1, d2]]
    n_intersect = len(DI_sets[0] & DI_sets[1])
    return n_intersect, len(DI_sets[0]), len(DI_sets[1])


def loop_threshs(d1: pd.DataFrame, d2: pd.DataFrame, name: str)-> int:
    '''
        Loops over different thresholds for the RSC and calcualtes the
        intersection of two given datasets.
        :param d1: dataset 1
        :param d2: dataset 2
        :param name: name of the experiment
    '''
    threshs = np.arange(50)
    fracs = list()
    ns_new = list()
    ns_orig = list()
    above_thresh = False
    rsc = np.nan
    for t in threshs:
        n_inter, n_new, n_orig = compare_datasets(d1, d2, t)
        frac = 2 * n_inter / (n_new + n_orig)
        fracs.append(frac)
        ns_new.append(n_new)
        ns_orig.append(n_orig)

        if frac > 0.75 and not above_thresh:
            rsc = t
            above_thresh = True

    # plot fraction of intersecting DelVGs between the two datasets
    plt.plot(threshs, fracs)
    if above_thresh:
        plt.axvline(x=rsc, color='r', linestyle='--')
        plt.text(rsc, 0.5, f'rsc={rsc}', ha='center')
    plt.ylim(0, 1.1)
    plt.ylabel("ratio of common DelVGs")
    plt.xlabel("cutoff value")
    save_path = os.path.join(RESULTSPATH, "validation_estimation")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{name}_common_DelVGs.png"))
    plt.close()

    # plot number of unique DelVGs for the two datasets
    plt.plot(threshs, ns_new, label="selfm")
    plt.plot(threshs, ns_orig, label="orig")
    if above_thresh:
        plt.axvline(x=rsc, color='r', linestyle='--')
        plt.text(rsc, n_orig, f'x={rsc}', ha='center')
    plt.legend()
    plt.ylabel("number of unique DelVGs")
    plt.xlabel("cutoff value")
    plt.savefig(os.path.join(save_path, f"{name}_unique_DelVGs.png"))
    plt.close()

    return rsc
 

if __name__ == "__main__":
    plt.style.use("seaborn")
    RESULTSPATH = os.path.dirname(RESULTSPATH)
    rscs = list()
    ns = list()
    
    ### Pelz ###
    name = "pelz"
    print(f"### {name} ###")
    pelz = load_dataset("Pelz2021")
    ns.append(pelz.shape[0])
    orig_pelz = load_pelz2021_rsc()
    rscs.append(loop_threshs(pelz, orig_pelz, name))

    ### Alnaji2021 ###
    name = "alnaji2021"
    print(f"### {name} ###")
    alnaji2021 = load_dataset("Alnaji2021")
    ns.append(alnaji2021.shape[0])
    orig_alnaji2021 = load_alnaji2021_rsc()
    rscs.append(loop_threshs(alnaji2021, orig_alnaji2021, name))

    ### Alnaji2019 Cal07 ###
    strain = "Cal07"
    print(f"### {strain} ###")
    df = load_dataset("Alnaji2019_Cal07")
    ns.append(df.shape[0])
    orig = load_alnaji2019_rsc(strain)
    rscs.append(loop_threshs(df, orig, f"alnaji2019_{strain}"))

    ### Alnaji2019 NC ###
    strain = "NC"
    print(f"### {strain} ###")
    df = load_dataset("Alnaji2019_NC")
    ns.append(df.shape[0])
    orig = load_alnaji2019_rsc(strain)
    rscs.append(loop_threshs(df, orig, f"alnaji2019_{strain}"))

    ### Alnaji2019 Perth ###
    strain = "Perth"
    print(f"### {strain} ###")
    df = load_dataset("Alnaji2019_Perth")
    ns.append(df.shape[0])
    orig = load_alnaji2019_rsc(strain)
    rscs.append(loop_threshs(df, orig, f"alnaji2019_{strain}"))

    ### Alnaji2019 BLEE ###
    strain = "BLEE"
    print(f"### {strain} ###")
    df = load_dataset("Alnaji2019_BLEE")
    ns.append(df.shape[0])
    orig = load_alnaji2019_rsc(strain)
    rscs.append(loop_threshs(df, orig, f"alnaji2019_{strain}"))

    # this shows that no dependences between dataset size and RCS is given
    print(rscs)
    print(np.mean(rscs))
    plt.plot(ns, rscs)
    plt.show()
    plt.close()
