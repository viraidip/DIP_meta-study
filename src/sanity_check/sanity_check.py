'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_dataset, load_alnaji2019, join_data
from utils import SEGMENTS, RESULTSPATH, DATAPATH, CUTOFF, CMAP, SEGMENT_DICTS


def load_pelz2021_sanity()-> dict:
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
    file_path = os.path.join(DATAPATH, "sanity_check", filename)
    data_dict = pd.read_excel(io=file_path,
                              sheet_name=None,
                              header=0,
                              na_values=["", "None"],
                              keep_default_na=False)

    return data_dict


def load_alnaji2021_sanity()-> dict:
    '''
        Loads the data set of Alnaji et al. 2021. Returns a dictionary with the
        data.

        :return: dictionary with strain name as key and data frame as value
    '''
    path = os.path.join(DATAPATH, "sanity_check", "Early_DIs_mbio.xlsx")
    data = pd.read_excel(path, na_values=["", "None"], keep_default_na=False)
    dic = dict({"PR8": data})
    return dic


def load_alnaji2019_sanity()-> dict:
    '''
    '''
    file_path = os.path.join(DATAPATH, "sanity_check", "DI_Influenza_FA_JVI.xlsx")
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


def load_mendes2021_sanity(name: str)-> dict:
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
    
    file_path = os.path.join(DATAPATH, "sanity_check", filename)
    data = pd.read_csv(file_path,
                            header=0,
                            na_values=["", "None"],
                            keep_default_na=False,
                            sep="\t")
    
    return data


def load_lui2019_sanity(name: str)-> dict:
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
    
    file_path = os.path.join(DATAPATH, "sanity_check", filename)
    data = pd.read_csv(file_path,
                            header=0,
                            na_values=["", "None"],
                            keep_default_na=False
                            )
    
    return join_data(data)


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


def loop_threshs(d1, d2)-> None:
    '''
    
    '''
    threshs = np.arange(50)
    fracs = list()
    ns_new = list()
    ns_orig = list()
    above_thresh = False
    for t in threshs:
        n_inter, n_new, n_orig = compare_datasets(d1, d2, t)
        frac = 2 * n_inter / (n_new + n_orig)
        fracs.append(frac)
        ns_new.append(n_new)
        ns_orig.append(n_orig)

        if frac > 0.85 and not above_thresh:
            x = t
            above_thresh = True

    plt.plot(threshs, fracs)
    if above_thresh:
        plt.axvline(x=x, color='r', linestyle='--')
        plt.text(x, 0.5, f'x={x}', ha='center')
    plt.ylim(0, 1.1)
    plt.ylabel("ratio of common DVGs")
    plt.xlabel("cutoff value")
    plt.show()
    plt.close()

    plt.plot(threshs, ns_new, label="selfm")
    plt.plot(threshs, ns_orig, label="orig")
    if above_thresh:
        plt.axvline(x=x, color='r', linestyle='--')
        plt.text(x, n_orig, f'x={x}', ha='center')
    plt.legend()
    plt.ylabel("number of unique DVGs")
    plt.xlabel("cutoff value")
    plt.show()
    plt.close()
 

if __name__ == "__main__":
    d = dict({
        "AF389115.1": "PB2",
        "AF389116.1": "PB1",
        "AF389117.1": "PA",
        "AF389118.1": "HA",
        "AF389119.1": "NP",
        "AF389120.1": "NA",
        "AF389121.1": "M",
        "AF389122.1": "NS"
    })
    '''
    ### Pelz seed ###
    seed = load_dataset("Pelz2021", "SRR15084925", d)
    orig_pelz = load_pelz2021_sanity()
    orig_seed = orig_pelz["PR8"].iloc[:, :4].copy()
    orig_seed = orig_seed[orig_seed["VB3-Saat"] != 0]
    orig_seed = orig_seed.rename(columns={"VB3-Saat": "NGS_read_count"})
    print("### Pelz seed virus ###")
    loop_threshs(seed, orig_seed)

    vb3_15 = load_dataset("Pelz2021", "SRR15084925", d)
    orig_data = orig_pelz["PR8"].iloc[:, :10].copy()
    orig_data = orig_data[orig_data["VB3-15"] != 0]
    orig_data = orig_data.rename(columns={"VB3-15": "NGS_read_count"})
    print("### Pelz VB3-15 ###")
    loop_threshs(vb3_15, orig_seed)

    ### Alnaji2021 ###
    repB_6hpi = load_dataset("Alnaji2021", "SRR14352110", d)
    orig_alnaji2021 = load_alnaji2021_sanity()["PR8"]
    org_B_6 = orig_alnaji2021[(orig_alnaji2021["Replicate"] == "Rep2") & (orig_alnaji2021["Timepoint"] == "6hpi")]
    print("### Rep B 6 hpi ###")
    loop_threshs(repB_6hpi, org_B_6)

    repC_3hpi = load_dataset("Alnaji2021", "SRR14352112", d)
    org_C_3 = orig_alnaji2021[(orig_alnaji2021["Replicate"] == "Rep3") & (orig_alnaji2021["Timepoint"] == "3hpi")]
    print("### Rep C 3 hpi ###")
    loop_threshs(repC_3hpi, org_C_3)
    
    ### Alnaji2019 ###
    pas_d = dict({"Cal07": "6", "NC": "1", "Perth": "4", "BLEE": "7"})
    orig_alnaji2019 = load_alnaji2019_sanity()
    for st in ["Cal07", "NC", "Perth", "BLEE"]:
        df = load_alnaji2019(st)
        df = df[df["NGS_read_count"] >= 5]
        print(f"\n### {st} ###")
        for pas in df["Passage"].unique():
            pas_df = df[df["Passage"] == pas].copy()
            print(f"# passage {pas} #")
            for l in df["Lineage"].unique():
                l_df = pas_df[pas_df["Lineage"] == l]
                orig = orig_alnaji2019[f"{st}_l{l}"]
#                inter, n1, n2 = compare_datasets(l_df, orig)
#                print(f"lineage {l}")
 #               print(f"orig {n2}")
  #              print(f"self {n1}")
   #             print(f"intersection {inter}")
                if pas_d[st] == pas:    
                    loop_threshs(l_df, orig)

    ### Mendes 2021 ###
    v12enriched = load_dataset("Mendes2021", "SRR15720521", dict({s: s for s in SEGMENTS}))
    orig_mendes = load_mendes2021_sanity("v12enriched")
    print("### Mendes V-1-2 enriched###")
    loop_threshs(v12enriched, orig_mendes)

    v21depleted = load_dataset("Mendes2021", "SRR15720526", dict({s: s for s in SEGMENTS}))
    orig_mendes = load_mendes2021_sanity("v21depleted")
    print("### Mendes V-2-1 enriched###")
    loop_threshs(v21depleted, orig_mendes)
    '''
    ### Lui 2019 ###
    illumina = load_dataset("Lui2019", "SRR8949705", SEGMENT_DICTS["Anhui"])
    orig_lui = load_lui2019_sanity("illumina")
    print("### Lui 2019 Illumina ###")
    loop_threshs(illumina, orig_lui)