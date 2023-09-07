'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_dataset, load_alnaji2019
from utils import SEGMENTS, RESULTSPATH, DATAPATH, CUTOFF


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
    threshs = np.arange(21)
    fracs = list()
    ns_new = list()
    ns_orig = list()
    for t in threshs:
        frac, n_new, n_orig = compare_datasets(d1, d2, t)
        fracs.append(frac)
        ns_new.append(n_new)
        ns_orig.append(n_orig)

    plt.plot(threshs, fracs)
    plt.ylim(0, 1.1)
    plt.show()
    plt.close()

    plt.plot(threshs, ns_new, label="selfm")
    plt.plot(threshs, ns_orig, label="orig")
    plt.legend()
    plt.show()
    plt.close()



def plot_distribution_over_segments(dfs, dfnames, mode)-> None:
    '''

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        col (str, optional): The column name in the DataFrames that contains the sequence segments of interest. 
                             Default is "seq_around_deletion_junction".

    :return: None
    '''
    fig, axs = plt.subplots(figsize=(len(dfs)*1.5, 6), nrows=2, ncols=4)
    cm = plt.get_cmap("tab10")

    i = 0
    j = 0
    li = list()
    for df, dfname in zip(dfs, dfnames):
        fractions = df["Segment"].value_counts() / len(df) * 100
        for s in SEGMENTS:
            if s not in fractions:
                fractions[s] = 0.0
        sorted_fractions = fractions.loc[SEGMENTS]
        li.append(sorted_fractions.values)

        colors = list()
        for k, s in enumerate(SEGMENTS):
            if sorted_fractions[s] != 0.0:
                colors.append(cm(1.*k/8))
            else:
                sorted_fractions.drop(s, inplace=True)

        axs[i,j].set_prop_cycle("color", colors)
        axs[i,j].set_title(dfname)
        
        patches, _ = axs[i,j].pie(sorted_fractions)
        labels = ['{0} {1:1.1f} %'.format(i,j) for i,j in zip(sorted_fractions.index, sorted_fractions)]
        axs[i,j].legend(patches, labels, loc="center", bbox_to_anchor=(-0.2, 0.5), fontsize=10)

        j += 1
        if j == 4:
            i = 1
            j = 0

    table = np.array(li)
    statistic, pvalue, dof, expected_freq = chi2_contingency(table)
    print(statistic)
    print(pvalue)

    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, "figure1", f"fraction_segments_{mode}.png")
    plt.savefig(save_path)
    plt.close()

def calculate_deletion_shifts(dfs, dfnames, mode)-> None:
    '''

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        col (str, optional): The column name in the DataFrames that contains the sequence segments of interest. 
                             Default is "seq_around_deletion_junction".
    :return: None
    '''
    fig, axs = plt.subplots(figsize=(len(dfs) * 1.5, 6), nrows=2, ncols=4)
    cm = plt.get_cmap("tab10")
    colors = [cm(1.*i/3) for i in range(3)]

    i = 0
    j = 0
    overall = np.array([0, 0, 0])
    n = 0
    li = list()

    for df, dfname in zip(dfs, dfnames):
        df["length"] = df["deleted_sequence"].apply(len)
        df["shift"] = df["length"] % 3
        shifts = df["shift"].value_counts()
        sorted_shifts = shifts.loc[[0, 1, 2]]
        overall += sorted_shifts
        n += len(df)
        li.append(shifts)
        shifts = shifts / len(df)

        axs[i,j].set_title(dfname)
        labels = list(["in-frame", "shift +1", "shift -1"])
        axs[i,j].pie(sorted_shifts, labels=labels, autopct="%1.1f%%", colors=colors, textprops={"size": 14})

        j += 1
        if j == 4:
            i = 1
            j = 0


    table = np.array(li)
    statistic, pvalue, dof, expected_freq = chi2_contingency(table)
    print(statistic)
    print(pvalue)

    print(f"mean distribution:\n\t{overall/n}")

    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, "figure1", f"deletion_shifts_{mode}.png")
    plt.savefig(save_path)
    plt.close()
 

if __name__ == "__main__":
    '''
    ### Pelz ###
    seed = load_dataset("Pelz2021", "SRR15084925")
    orig_pelz = load_pelz2021_sanity()
    orig_seed = orig_pelz["PR8"].iloc[:, :4].copy()
    orig_seed = orig_seed[orig_seed["VB3-Saat"] != 0]
    orig_seed = orig_seed.rename(columns={"VB3-Saat": "NGS_read_count"})
    print("### Pelz seed virus ###")
    loop_threshs(seed, orig_seed)

    ### Alnaji2021 ###
    repB_6hpi = load_dataset("Alnaji2021", "SRR14352110")
    orig_alnaji2021 = load_alnaji2021_sanity()["PR8"]
    org_B_6 = orig_alnaji2021[(orig_alnaji2021["Replicate"] == "Rep2") & (orig_alnaji2021["Timepoint"] == "6hpi")]
    print("### Rep B 6 hpi ###")
    loop_threshs(repB_6hpi, org_B_6)

    repC_3hpi = load_dataset("Alnaji2021", "SRR14352112")
    org_C_3 = orig_alnaji2021[(orig_alnaji2021["Replicate"] == "Rep3") & (orig_alnaji2021["Timepoint"] == "3hpi")]
    print("### Rep C 3 hpi ###")
    loop_threshs(repC_3hpi, org_C_3)
    '''
    ### Alnaji2019 ###
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
                inter, n1, n2 = compare_datasets(l_df, orig)

                print(f"lineage {l}")
                print(f"orig {n2}")
                print(f"self {n1}")
                print(f"intersection {inter}")
                