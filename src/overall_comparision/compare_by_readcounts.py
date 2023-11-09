'''

'''
import os
import sys

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from compare_expected import plot_expected_vs_observed_nucleotide_enrichment_heatmaps, plot_expected_vs_observed_direct_repeat_heatmaps
from general_analyses import plot_distribution_over_segments, diff_start_end_lengths

sys.path.insert(0, "..")
from utils import load_all


def split_by_reads(dfs, split):
    '''
    
    '''
    high_count_dfs = list()
    low_count_dfs = list()

    for df in dfs:
        if split == "median": 
            median = df["NGS_read_count"].median()
            high_df = df[df['NGS_read_count'] > median].copy()
            low_df = df[df['NGS_read_count'] <= median].copy()

        elif split == "33perc":
            perc1 = df["NGS_read_count"].quantile(q=0.33)
            perc2 = df["NGS_read_count"].quantile(q=0.66)

            high_df = df[df['NGS_read_count'] > perc2].copy()
            low_df = df[df['NGS_read_count'] <= perc1].copy()

        high_count_dfs.append(high_df)
        low_count_dfs.append(low_df)

    return high_count_dfs, low_count_dfs


if __name__ == "__main__":
    plt.style.use("seaborn")
    dfs, dfnames, _ = load_all()
    split = "33perc"
    dfs, expected_dfs = split_by_reads(dfs, split)

    name = f"{split}_ngscount"
    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(dfs, dfnames, expected_dfs, name)
    plot_expected_vs_observed_direct_repeat_heatmaps(dfs, dfnames, expected_dfs, "high count-low count", name)

    plot_distribution_over_segments(dfs, dfnames, f"{name}_high")
    diff_start_end_lengths(dfs, dfnames, f"{name}_high")
    
    plot_distribution_over_segments(expected_dfs, dfnames, f"{name}_low")
    diff_start_end_lengths(expected_dfs, dfnames, f"{name}_low")