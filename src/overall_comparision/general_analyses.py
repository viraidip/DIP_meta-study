'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

sys.path.insert(0, "..")
from utils import load_all
from utils import SEGMENTS, RESULTSPATH


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
 

def length_distribution(dfs, dfnames: list)-> None:
    '''
    
    '''
    plt.rc("font", size=16)
    mean_dict = dict({"Segment": SEGMENTS})

    for df, dfname in zip(dfs, dfnames):
        mean_list = list()
        # create a dict for each segment including the NGS read count
        count_dict = dict()
        for s in SEGMENTS:
            count_dict[s] = dict()

       # df["DVG_Length"] = len(df["seq"])-len(df["deleted_sequence"])
        for _, r in df.iterrows():
            DVG_Length = len(r["seq"])-len(r["deleted_sequence"])
            if DVG_Length in count_dict[r["Segment"]]:
                count_dict[r["Segment"]][DVG_Length] += 1
            else:
                count_dict[r["Segment"]][DVG_Length] = 1

        # create a subplot for each key, value pair in count_dict
        fig, axs = plt.subplots(8, 1, figsize=(10, 15), tight_layout=True)
        for i, s in enumerate(SEGMENTS):
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

            mean_list.append(m)

        mean_dict[dfname] = mean_list

        save_path = os.path.join(RESULTSPATH, "general_analysis", f"{dfname}_length_del_hist.png")
        plt.savefig(save_path)
        plt.close()

    mean_df = pd.DataFrame(mean_dict)
    print(mean_df)


if __name__ == "__main__":
    plt.style.use("seaborn")
    dfs, dfnames, expected_dfs = load_all()

 #   plot_distribution_over_segments(dfs, dfnames)
  #  calculate_deletion_shifts(dfs, dfnames)

    length_distribution(dfs, dfnames)