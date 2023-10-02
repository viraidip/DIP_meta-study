'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats import chi2_contingency

sys.path.insert(0, "..")
from utils import load_all, get_seq_len
from utils import SEGMENTS, RESULTSPATH, DATASET_STRAIN_DICT


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
    fig, axs = plt.subplots(figsize=(len(dfs), 6), nrows=1, ncols=1)
    cm = plt.get_cmap("tab10")
    colors = [cm(1.*i/len(SEGMENTS)) for i in range(len(SEGMENTS))]

    x = np.arange(0, len(dfs))

    y = dict({s: list() for s in SEGMENTS})
    for df in dfs:
        fractions = df["Segment"].value_counts() / len(df)
        for s in SEGMENTS:
            if s not in fractions:
                y[s].append(0.0)
            else:
                y[s].append(fractions[s])

    bar_width = 0.7
    bottom = np.zeros(len(dfs))

    for i, s in enumerate(SEGMENTS):
        axs.bar(x, y[s], bar_width, color=colors[i], label=s, bottom=bottom)
        bottom += y[s]
    
    axs.set_ylabel("relative occurrence of segment")
    axs.set_xlabel("dataset")
    plt.xticks(range(len(dfnames)), dfnames, size='small', rotation=45) 

    box = axs.get_position()
    axs.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    axs.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=8)
    
    save_path = os.path.join(RESULTSPATH, "general_analysis", "fraction_segments.png")
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
    fig, axs = plt.subplots(figsize=(12, 12), nrows=4, ncols=4)
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
    save_path = os.path.join(RESULTSPATH, "general_analysis", "deletion_shifts.png")
    plt.savefig(save_path)
    plt.close()
 

def length_distribution(dfs: list, dfnames: list)-> None:
    '''
    
    '''
    plt.rc("font", size=16)
    overall_count_dict = dict()

    for df, dfname in zip(dfs, dfnames):
        # create a dict for each segment including the NGS read count
        count_dict = dict()
        for s in SEGMENTS:
            count_dict[s] = dict()

        for _, r in df.iterrows():
            DVG_Length = len(r["seq"])-len(r["deleted_sequence"])
            if DVG_Length in count_dict[r["Segment"]]:
                count_dict[r["Segment"]][DVG_Length] += 1
            else:
                count_dict[r["Segment"]][DVG_Length] = 1

        overall_count_dict[dfname] = count_dict



    # create a subplot for each dataset

    for s in SEGMENTS:
        fig, axs = plt.subplots(len(dfnames), 1, figsize=(10, 15), tight_layout=True)
        for i, dfname in enumerate(dfnames):
            count_dict = overall_count_dict[dfname]
            if len(count_dict[s].keys()) > 1:
                counts_list = list()
                for length, count in count_dict[s].items():
                    counts_list.extend([length] * count)

                m = round(np.mean(list(counts_list)), 2)                
                axs[i].hist(count_dict[s].keys(), weights=count_dict[s].values(), bins=100, label=f"{dfname} (µ={m})", alpha=0.3)
                axs[i].set_xlim(left=0)
   #             axs[i].set_yticks([0, 4, 8, 12])
                axs[i].set_xlabel("sequence length")
                axs[i].set_ylabel("occurrences")
                axs[i].legend()
            else:
                axs[i].set_visible(False)
                m = 0


        save_path = os.path.join(RESULTSPATH, "general_analysis", f"{s}_length_del_hist.png")
        plt.savefig(save_path)
        plt.close()



def create_start_end_connection_plot(df: pd.DataFrame,
                                     dfname: str,
                                     strain: str,
                                     segment: str):
    '''
    
    '''
    max_val = get_seq_len(strain, segment)
    cm = plt.get_cmap("tab10")
    colors = [cm(1.*i/10) for i in range(10)]

    fig, ax = plt.subplots(figsize=(5, 3))
    for i, row in df.iterrows():
        center = row["Start"] + (row["End"] - row["Start"]) / 2
        radius = (row["End"] - row["Start"]) / 2
        start_angle = 0
        end_angle = 180
        color = colors[0]
        y = 80
        if radius < 200:
            start_angle = 180
            end_angle = 0
            color = colors[3]
            y = 0
        half_cirlce = patches.Arc((center, y), radius*2, radius*2, angle=0, theta1=start_angle, theta2=end_angle, color=color, alpha=0.5)
        ax.add_patch(half_cirlce)

    # add boxes for start and end of DI RNA sequence
    ax.add_patch(plt.Rectangle((0, 0), max_val, 80, alpha=0.7, color="black"))
    ax.annotate("RNA sequence", (max_val / 2, 10), color="white", ha="center")

    # change some values to improve figure
    ax.set_xlim(0, max_val)
    ax.set_ylim(-200, max_val / 2)
    ax.set_xticks(np.arange(0, max_val, 200))
    ax.set_yticks([])
    ax.set_xlabel("Nucleotide position")

    # save figure
    plt.tight_layout()
    save_path = os.path.join(RESULTSPATH, "start_end", dfname, f"{segment}.png")
    plt.savefig(save_path)
    plt.close()


def start_end_positions(dfs: list, dfnames: list)-> None:
    '''
    
    '''
    for df, dfname in zip(dfs, dfnames):
        strain = DATASET_STRAIN_DICT[dfname]
        for s in SEGMENTS:
            copy_df = df[(df["Segment"] == s)].copy()
            create_start_end_connection_plot(copy_df, dfname, strain, s)

if __name__ == "__main__":
    plt.style.use("seaborn")
    dfs, dfnames, expected_dfs = load_all()

 #   plot_distribution_over_segments(dfs, dfnames)
  #  calculate_deletion_shifts(dfs, dfnames)
   # length_distribution(dfs, dfnames)

    start_end_positions(dfs, dfnames)