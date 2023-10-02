'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_wang2023, join_data, preprocess
from utils import SEGMENTS, RESULTSPATH, DATAPATH, CUTOFF




if __name__ == "__main__":
    dfs = list()
    dfnames = list()
    key_l = list()

    strain = "WSN"
    df = load_wang2023()
    df = df[(df['IFNAR'] == '0') | (df['IFNLR'] == '0')]

    # filter out DVGs that are in both (enriched and depleted)
    for stat in df["IFNAR"].unique():
        stat_df = df[df["IFNAR"] == stat].copy()
        stat_df = join_data(stat_df)
        stat_df["DI"] = stat_df["Segment"] + "_" + stat_df["Start"].astype(str) + "_" + stat_df["End"].astype(str)
        key_l.append(stat_df["DI"].to_list())
        
    set1 = set(key_l[0])
    set2 = set(key_l[1])
    key_inter = list(set1.intersection(set2))
    
    for stat in df["IFNAR"].unique():
        stat_df = df[df["IFNAR"] == stat].copy()
        stat_df = join_data(stat_df)
        stat_df = stat_df[stat_df["NGS_read_count"] >= CUTOFF].copy()

        stat_df["DI"] = stat_df["Segment"] + "_" + stat_df["Start"].astype(str) + "_" + stat_df["End"].astype(str)
        stat_df = stat_df.loc[~stat_df["DI"].isin(key_inter)]
        stat_df.drop("DI", inplace=True, axis=1)

        stat_df.to_csv(f"Wang2023_{stat}.csv", index=False)



        dfs.append(preprocess(strain, stat_df, CUTOFF))
        dfnames.append(f"Mendes {stat}")



    ### Differences in nucleotide occurrence, but nothing specific over all segments