'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


sys.path.insert(0, "..")
from utils import load_all
from utils import RESULTSPATH, SEGMENTS


def get_long_dis(df):
    '''
    
    '''
    thresh = 0.85
    
    df["len_full"] = df["full_seq"].apply(len)
    df["len_di"] = df["deleted_sequence"].apply(len)
    df["len_ratio"] = df["len_di"] / df["len_full"]

    final_df = df[df["len_ratio"] > thresh].copy()
    final_df.drop(columns=["len_ratio", "len_di", "len_full"], inplace=True)

    return final_df


def fraction_long_dis(dfs: list, dfnames: list):
    '''
    
    '''
    fractions = list()
    for df, dfname in zip(dfs, dfnames):
        n_all_dis = len(df)
        n_long_dis = len(get_long_dis(df))
       
        f = n_long_dis/n_all_dis * 100
        fractions.append(f)
        
    res_df = pd.DataFrame(dict({"names": dfnames, "fraction DIs": fractions}))
    print(res_df)

    filepath = os.path.join(RESULTSPATH, "long_dis", "fractions.csv")
    res_df.to_csv(filepath, index=False)


def lengths_long_dis(dfs, dfnames):
    '''
    
    '''
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

            save_path = os.path.join(RESULTSPATH, "long_dis", f"lengths_{dfname}_{s}.png")
            plt.savefig(save_path)
            plt.close()


if __name__ == "__main__":
    plt.style.use('seaborn')

    dfs, dfnames, expected_dfs = load_all()

    fraction_long_dis(dfs, dfnames)
    lengths_long_dis(dfs, dfnames)