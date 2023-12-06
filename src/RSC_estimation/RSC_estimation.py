'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_single_dataset, join_data
from utils import SEGMENTS, RESULTSPATH, DATAPATH, CUTOFF, CMAP, SEGMENT_DICTS


def load_pelz2021_rsc()-> dict:
    '''
        Loads the data from Pelz et al 2021 publication.
        Is structured the same way as data from Alnaji 2019.
        :param de_novo: if True only de novo candidates are taken
        :param long_dirna: if True loads data set that includes long DI RNA
                           candidates
        :param by_time: if True loads the dataset split up by timepoints

        :return: dictionary with one key, value pair
    '''
    filename = "ShortDeletions_by_timepoints.xlsx"
    file_path = os.path.join(DATAPATH, "RSC_estimation", filename)
    data_dict = pd.read_excel(io=file_path,
                              sheet_name=None,
                              header=0,
                              na_values=["", "None"],
                              keep_default_na=False)

    return data_dict


def load_alnaji2021_rsc()-> dict:
    '''
        Loads the data set of Alnaji et al. 2021. Returns a dictionary with the
        data.

        :return: dictionary with strain name as key and data frame as value
    '''
    path = os.path.join(DATAPATH, "RSC_estimation", "Early_DIs_mbio.xlsx")
    data = pd.read_excel(path, na_values=["", "None"], keep_default_na=False)
    dic = dict({"PR8": data})
    return dic


def load_alnaji2019_rsc()-> dict:
    '''
    '''
    file_path = os.path.join(DATAPATH, "RSC_estimation", "DI_Influenza_FA_JVI.xlsx")
    data_dict = pd.read_excel(io=file_path,
                              sheet_name=None,
                              header=1,
                              na_values=["", "None"],
                              keep_default_na=False,
                              converters={"Start": int,"End": int, "NGS_read_count": int,
                                          "Start.1": int,"End.1": int, "NGS_read_count.1": int})
    # Splitting up the two lines in new data frames and cleaning NaN data
    # For l2 the columns get renamed, they get the same names as in l1
    # Cleaned data is stored in a dict, can be accessed by [datasetname]_[l1/l2]
    # dataset names are "Cal07", "NC", "Perth", "BLEE"
    cleaned_data_dict = dict()
    for key, value in data_dict.items():
        cleaned_data_dict[f"{key}_l1"] = data_dict[key].iloc[:, 0:4]
        cleaned_data_dict[f"{key}_l2"] = data_dict[key].iloc[:, 5:9]

        cleaned_data_dict[f"{key}_l1"].dropna(how="all", inplace=True)
        cleaned_data_dict[f"{key}_l2"].dropna(how="all", inplace=True)

        cleaned_data_dict[f"{key}_l2"].columns = cleaned_data_dict[f"{key}_l1"].columns 

    return cleaned_data_dict


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


def load_lui2019_rsc(name: str)-> dict:
    '''
        :param de_novo: if True only de novo candidates are taken
        :param long_dirna: if True loads data set that includes long DI RNA
                           candidates
        :param by_time: if True loads the dataset split up by timepoints

        :return: dictionary with one key, value pair
    '''
    if name == "SMRT":
        filename = "Virus-1-2_enriched_junctions.tsv"
    elif name == "illumina":
        filename = "Lui2019_Illumina.csv"
    
    file_path = os.path.join(DATAPATH, "RSC_estimation", filename)
    data = pd.read_csv(file_path,
                            header=0,
                            na_values=["", "None"],
                            keep_default_na=False
                            )
    
    data = join_data(data)
    #data = data[(data["End"] - data["Start"]) >= 10]

    return data


def compare_datasets(d1, d2, thresh=1)-> float:
    '''
    
    '''
    d1 = d1[d1["NGS_read_count"] >= thresh]
    d2 = d2[d2["NGS_read_count"] >= thresh]
    
    dfs = list([d1, d2])

    DI_sets = list()
    for d in dfs:
        DI_sets.append(set(d["Segment"] + "_" + d["Start"].astype(str) + "_" + d["End"].astype(str)))
    
    n1 = len(DI_sets[0])
    n2 = len(DI_sets[1])
    n_intersect = len(DI_sets[0] & DI_sets[1])

    return n_intersect, n1, n2


def loop_threshs(d1, d2, name)-> int:
    '''
    
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

        if frac > 0.5 and not above_thresh:
            rsc = t
            above_thresh = True

    plt.plot(threshs, fracs)
    if above_thresh:
        plt.axvline(x=rsc, color='r', linestyle='--')
        plt.text(rsc, 0.5, f'rsc={rsc}', ha='center')
    plt.ylim(0, 1.1)
    plt.ylabel("ratio of common DVGs")
    plt.xlabel("cutoff value")
    save_path = os.path.join(RESULTSPATH, "RSC_estimation")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{name}_common_DVGs.png"))
    plt.close()

    plt.plot(threshs, ns_new, label="selfm")
    plt.plot(threshs, ns_orig, label="orig")
    if above_thresh:
        plt.axvline(x=rsc, color='r', linestyle='--')
        plt.text(rsc, n_orig, f'x={rsc}', ha='center')
    plt.legend()
    plt.ylabel("number of unique DVGs")
    plt.xlabel("cutoff value")

    save_path = os.path.join(RESULTSPATH, "RSC_estimation")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{name}_unique_DVGs.png"))
    plt.close()

    return rsc
 

if __name__ == "__main__":
    rscs = list()
    ns = list()

    ### Pelz seed ###
    name = "pelz_seed"
    seed = load_single_dataset("Pelz2021", "SRR15084925", SEGMENT_DICTS["PR8"])
    ns.append(seed.shape[0])
    orig_pelz = load_pelz2021_rsc()
    orig_seed = orig_pelz["PR8"].iloc[:, :4].copy()
    orig_seed = orig_seed[orig_seed["VB3-Saat"] != 0]
    orig_seed = orig_seed.rename(columns={"VB3-Saat": "NGS_read_count"})
    print(f"### {name} ###")
    rscs.append(loop_threshs(seed, orig_seed, name))
    
    name = "pelz_VB3-15"
    vb3_15 = load_single_dataset("Pelz2021", "SRR15084925", SEGMENT_DICTS["PR8"])
    ns.append(vb3_15.shape[0])
    orig_data = orig_pelz["PR8"].iloc[:, :10].copy()
    orig_data = orig_data[orig_data["VB3-15"] != 0]
    orig_data = orig_data.rename(columns={"VB3-15": "NGS_read_count"})
    print(f"### {name} ###")
    rscs.append(loop_threshs(vb3_15, orig_seed, name))

    ### Alnaji2021 ###
    name = "alnaji2021_repB6"
    repB_6hpi = load_single_dataset("Alnaji2021", "SRR14352110", SEGMENT_DICTS["PR8"])
    ns.append(repB_6hpi.shape[0])
    orig_alnaji2021 = load_alnaji2021_rsc()["PR8"]
    org_B_6 = orig_alnaji2021[(orig_alnaji2021["Replicate"] == "Rep2") & (orig_alnaji2021["Timepoint"] == "6hpi")]
    print(f"### {name} ###")
    rscs.append(loop_threshs(repB_6hpi, org_B_6, name))

    name = "alnaji2021_repC3"
    repC_3hpi = load_single_dataset("Alnaji2021", "SRR14352112", SEGMENT_DICTS["PR8"])
    ns.append(repC_3hpi.shape[0])
    org_C_3 = orig_alnaji2021[(orig_alnaji2021["Replicate"] == "Rep3") & (orig_alnaji2021["Timepoint"] == "3hpi")]
    print(f"### {name} ###")
    rscs.append(loop_threshs(repC_3hpi, org_C_3, name))

    orig_alnaji2019 = load_alnaji2019_rsc()
    ### Alnaji2019 Cal07 ###
    strain = "Cal07"
    for l in [1, 2]:
        name = f"alnaji2019{strain}{l}"
        if l == 1:
            accnum = "SRR8754522"
        elif l == 2:
            accnum = "SRR8754523"
        df = load_single_dataset(f"Alnaji2019_{strain}", accnum, SEGMENT_DICTS[strain])
        ns.append(df.shape[0])
        orig = orig_alnaji2019[f"{strain}_l{l}"]
        print(f"### {name} ###")
        rscs.append(loop_threshs(df, orig, name))

    ### Alnaji2019 NC ###
    strain = "NC"
    for l in [1, 2]:
        name = f"alnaji2019{strain}{l}"
        if l == 1:
            accnum = "SRR8754514"
        elif l == 2:
            accnum = "SRR8754513"
        df = load_single_dataset(f"Alnaji2019_{strain}", accnum, SEGMENT_DICTS[strain])
        ns.append(df.shape[0])
        orig = orig_alnaji2019[f"{strain}_l{l}"]
        print(f"### {name} ###")
        rscs.append(loop_threshs(df, orig, name))

    ### Alnaji2019 Perth ###
    strain = "Perth"
    for l in [1, 2]:
        name = f"alnaji2019{strain}{l}"
        if l == 1:
            accnum = "SRR8754524"
        elif l == 2:
            accnum = "SRR8754525"
        df = load_single_dataset(f"Alnaji2019_{strain}", accnum, SEGMENT_DICTS[strain])
        ns.append(df.shape[0])
        orig = orig_alnaji2019[f"{strain}_l{l}"]
        print(f"### {name} ###")
        rscs.append(loop_threshs(df, orig, name))

    ### Alnaji2019 BLEE ###
    strain = "BLEE"
    for l in [1, 2]:
        name = f"alnaji2019{strain}{l}"
        if l == 1:
            accnum = "SRR8754509"
        elif l == 2:
            accnum = "SRR8754508"
        df = load_single_dataset(f"Alnaji2019_{strain}", accnum, SEGMENT_DICTS[strain])
        ns.append(df.shape[0])
        orig = orig_alnaji2019[f"{strain}_l{l}"]
        print(f"### {name} ###")
        rscs.append(loop_threshs(df, orig, name))

    ### Mendes 2021 ###
    name = "mendes_v12enr"
    v12enriched = load_single_dataset("Mendes2021", "SRR15720521", dict({s: s for s in SEGMENTS}))
    ns.append(v12enriched.shape[0])
    orig_mendes = load_mendes2021_rsc("v12enriched")
    print(f"### {name} ###")
    rscs.append(loop_threshs(v12enriched, orig_mendes, name))

    name = "mendes_v21depl"
    v21depleted = load_single_dataset("Mendes2021", "SRR15720526", dict({s: s for s in SEGMENTS}))
    ns.append(v21depleted.shape[0])
    orig_mendes = load_mendes2021_rsc("v21depleted")
    print(f"### {name} ###")
    rscs.append(loop_threshs(v21depleted, orig_mendes, name))
    
    ### Lui 2019 ###
    name = "lui2019"
    illumina = load_single_dataset("Lui2019", "SRR8949705", SEGMENT_DICTS["Anhui"])
    ns.append(illumina.shape[0])
    orig_lui = load_lui2019_rsc("illumina")
    print(f"### {name} ###")
    rscs.append(loop_threshs(illumina, orig_lui, name))

    # this shows that no dependences between dataset size and RCS is given
    plt.plot(ns, rscs)
    plt.show()
    plt.close()