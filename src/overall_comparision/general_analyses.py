'''

'''
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

sys.path.insert(0, "..")
from utils import load_alnaji2019, load_alnaji2021, load_pelz2021, load_mendes2021, load_lui2019, load_penn2022
from utils import preprocess, join_data
from utils import SEGMENTS, RESULTSPATH, CUTOFF


def plot_distribution_over_segments(dfs: list, dfnames: list)-> None:
    '''

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        col (str, optional): The column name in the DataFrames that contains the sequence segments of interest. 
                             Default is "seq_around_deletion_junction".

    :return: None
    '''
    fig, axs = plt.subplots(figsize=(12, 10), nrows=3, ncols=4)
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
            i += 1
            j = 0

    table = np.array(li)
 #   statistic, pvalue, dof, expected_freq = chi2_contingency(table)
  #  print(statistic)
   # print(pvalue)

    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, "segments_shift", "fraction_segments.png")
    plt.savefig(save_path)
    plt.close()


def calculate_deletion_shifts(dfs: list, dfnames: list)-> None:
    '''

    Args:
        dfs (list of pandas.DataFrame): The list of DataFrames containing the data. 
                                        Each dataframe should be preprocessed with sequence_df(df)
        dfnames (list of str): The names associated with each DataFrame in `dfs`.
        col (str, optional): The column name in the DataFrames that contains the sequence segments of interest. 
                             Default is "seq_around_deletion_junction".
    :return: None
    '''
    fig, axs = plt.subplots(figsize=(12, 10), nrows=3, ncols=4)
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
            i += 1
            j = 0


    table = np.array(li)
 #   statistic, pvalue, dof, expected_freq = chi2_contingency(table)
  #  print(statistic)
   # print(pvalue)

    print(f"mean distribution:\n\t{overall/n}")

    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, "segments_shift", "deletion_shifts.png")
    plt.savefig(save_path)
    plt.close()
 

if __name__ == "__main__":
    plt.style.use("seaborn")
    dfs = list()
    dfnames = list()
    expected_dfs = list()


    ### Alnaji2019 ###
    for strain, p in [("Cal07", "6"), ("NC", "1"), ("Perth", "4") , ("BLEE", "7")]:
        df = load_alnaji2019(strain)
        df = df[df["Passage"] == p].copy()
        df = join_data(df)
        dfs.append(preprocess(strain, df, CUTOFF))
        dfnames.append(f"Alnaji2021 {strain}")

    ### Alnaji2021 ###
    strain = "PR8"
    df = join_data(load_alnaji2021())
    dfs.append(preprocess(strain, df, CUTOFF))
    dfnames.append("Alnaji2021")

    ### Pelz2021 ###
    df = join_data(load_pelz2021())
    dfs.append(preprocess(strain, df, CUTOFF))
    dfnames.append("Pelz2021")
    
    ### Mendes2021 ###
    df = load_mendes2021()
    strain = "WSN"
    for virus in ["1", "2"]:
        df_v = df[df["Virus"] == virus].copy()
        df_v = join_data(df_v)
        dfs.append(preprocess(strain, df_v, CUTOFF))
        dfnames.append(f"Mendes2021_V{virus}")

    '''
    ### Lui2019 ###
    strain = "Anhui"
    df = load_lui2019()
    dfs.append(preprocess(strain, df, CUTOFF))
    dfnames.append("Lui2019")
    '''
    ### Penn2022 ###
    strain = "Turkey"
    df = load_penn2022()
    for l in ["1", "2"]:
        df_l = df[df["Lineage"] == l].copy()
        df_l = join_data(df_l)
        dfs.append(preprocess(strain, df_l, CUTOFF))
        dfnames.append(f"Penn2022_l{l}")

    plot_distribution_over_segments(dfs, dfnames)
    calculate_deletion_shifts(dfs, dfnames)