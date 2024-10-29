'''
    Analyse the DelVGs with a sequence length of at least 85 % of the full
    sequence.
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, "..")
from utils import load_all, get_seq_len, get_dataset_names
from utils import RESULTSPATH, SEGMENTS, DATASET_STRAIN_DICT, CMAP


THRESH = 0.85


def get_long_dis(df: pd.DataFrame)-> pd.DataFrame:
    '''
        identifies the DelVGs with a length above the defined threshold.
        :param dfs: DataFrame where long DelVGs should be identified
        
        :return: filtered DataFrame
    '''
    df["len_full"] = df["full_seq"].apply(len)
    df["len_di"] = df["len_full"] - df["deleted_sequence"].apply(len)
    df["len_ratio"] = df["len_di"] / df["len_full"]

    final_df = df[df["len_ratio"] > THRESH].copy()
    final_df.drop(columns=["len_ratio", "len_di", "len_full"], inplace=True)

    return final_df


def fraction_long_dis(dfs: list, dfnames: list)-> None:
    '''
        calculates the fration of long DelVGs in the given datasets.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        
        :return: None
    '''
    counts = list()
    fractions = list()
    for df in dfs:
        n_all_dis = len(df)
        n_long_dis = len(get_long_dis(df))
        f = n_long_dis/n_all_dis * 100
        counts.append(n_long_dis)
        fractions.append(f)
    res_df = pd.DataFrame(dict({"names": dfnames, "long DIs": counts, "fraction DIs": fractions}))
    save_path = os.path.join(RESULTSPATH, "long_delvgs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    res_df = res_df.sort_values(by='fraction DIs', ascending=False)
    res_df.to_csv(os.path.join(save_path, "fractions.csv"), float_format="%.1f", index=False)


def lengths_long_dis(dfs: list, dfnames: list)-> None:
    '''
        plot histrogram of DelVG deletion site lengths.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        
        :return: None
    '''
    save_path = os.path.join(RESULTSPATH, "long_delvgs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for df, dfname in zip(dfs, dfnames):
        for s in SEGMENTS:
            long_df = get_long_dis(df[df["Segment"] == s].copy())
            if long_df.shape[0] <= 20:
                continue
            lengths = long_df["deleted_sequence"].apply(len)
            bins = int(len(lengths)/2)

            plt.hist(lengths, bins=bins, edgecolor="black")
            plt.xlabel("DI length")
            plt.ylabel("Frequency")
            plt.title(f"{dfname} {s}")
            plt.savefig(os.path.join(save_path, f"lengths_{dfname}_{s}.png"))
            plt.close()


def frac_20_100_long_dis(dfs: list)-> None:
    '''
        plot histrogram of DelVG deletion site lengths.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        
        :return: None
    '''
    long_20_100 = 0
    long_all = 0
    for df in dfs:
        for s in SEGMENTS:
            long_df = get_long_dis(df[df["Segment"] == s].copy())
    #        if long_df.shape[0] <= 20:
     #           continue
            lengths = long_df["deleted_sequence"].apply(len)
            long_20_100 += len([x for x in lengths if 20 <= x <= 100])
            long_all += len(lengths)
            
    print(long_all)
    print(long_20_100)
    print(long_20_100/long_all)


def create_start_end_connection_plot(df: pd.DataFrame, dfname: str, strain: str, segment: str)-> None:
    '''
        plot the connection of the start and end of the deletion sites for a
        given dataset. Includes a histogram and marks long DelVGs.
        :param df: DelVG dataset, preprocessed with sequence_df(df)
        :param dfname: The name of the dataset
        :param strain: name of the strain
        :param segment: name of the segment

        :return: None
    '''
    max_val = get_seq_len(strain, segment)
    cm = plt.get_cmap(CMAP)
    colors = [cm(1.*i/8) for i in range(8)]
    positions = list()
    fig, ax = plt.subplots(figsize=(4, 2.5))
    for i, row in df.iterrows():
        positions.append(row["Start"])
        positions.append(row["End"])
        center = row["Start"] + (row["End"] - row["Start"]) / 2
        del_length = (row["End"] - row["Start"])
        radius = del_length / 2
        start_angle = 0
        end_angle = 180
        color = colors[0]
        y = 0
        if (del_length / max_val) <= (1 - THRESH):
            start_angle = 180
            end_angle = 0
            color = colors[5]
            y = -80
        half_cirlce = patches.Arc((center, y), radius*2, radius*2, angle=0, theta1=start_angle, theta2=end_angle, color=color, alpha=0.5)
        ax.add_patch(half_cirlce)

    # add box for RNA sequence
    ax.add_patch(plt.Rectangle((0, -80), max_val, 80, alpha=0.7, color="black"))
    ax.annotate("RNA sequence", (max_val / 2, -70), color="white", ha="center", fontsize=10)
    # add histogram
    if len(positions) > 0:
        ax2 = ax.twinx()
        ax2.hist(positions, density=True, bins=50, alpha=0.4)
    # change some values to improve figure
    ax.set_xlim(0, max_val)
    ax.set_ylim(-max_val / 8, max_val / 2)
    xticks = list(range(0, max_val, 300))
    xticks.append(max_val)
    ax.set_xticks(xticks)
    ax.set_yticks([])
    ax.set_xlabel("Nucleotide position")
    ax.set_title(dfname)

    def align_yaxis(ax1, v1, ax2, v2):
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
        miny, maxy = ax2.get_ylim()
        ax2.set_ylim(miny+dy, maxy+dy)
    if len(positions) > 0:    
        align_yaxis(ax, 0, ax2, 0)
        ax2.yaxis.grid(False)
        ax2.set_ylabel("Probability density")
    # save figure
    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, "start_end", dfname)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{segment}.png"), dpi=300)
    plt.close()


def start_end_positions(dfs: list, dfnames: list)-> None:
    '''
        plot the connection of the start and end of the deletion sites for all
        given dataset.
        :param dfs: The list of DataFrames containing the data, preprocessed
            with sequence_df(df)
        :param dfnames: The names associated with each DataFrame in `dfs`
        
        :return: None
    '''
    for df, dfname in zip(dfs, dfnames):
        strain = DATASET_STRAIN_DICT[dfname]
        for s in SEGMENTS:
            copy_df = df[(df["Segment"] == s)].copy()
            create_start_end_connection_plot(copy_df, dfname, strain, s)


if __name__ == "__main__":
    plt.style.use('seaborn')
    dfnames = get_dataset_names(cutoff=40)
    dfs, _ = load_all(dfnames)

    start_end_positions(dfs, dfnames)
    fraction_long_dis(dfs, dfnames)
    lengths_long_dis(dfs, dfnames)
    frac_20_100_long_dis(dfs)