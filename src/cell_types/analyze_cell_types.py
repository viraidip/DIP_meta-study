'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy import stats
from collections import Counter
from scipy.stats import chi2_contingency

sys.path.insert(0, "..")
from utils import load_dataset, preprocess, join_data, generate_expected_data
from utils import DATASET_STRAIN_DICT, CUTOFF
from overall_comparision.general_analyses import plot_distribution_over_segments, calculate_deletion_shifts, length_distribution_histrogram, length_distribution_violinplot, plot_nucleotide_ratio_around_deletion_junction_heatmaps, plot_direct_repeat_ratio_heatmaps, start_vs_end_lengths, diff_start_end_lengths
from overall_comparision.compare_expected import plot_expected_vs_observed_nucleotide_enrichment_heatmaps, plot_expected_vs_observed_direct_repeat_heatmaps, direct_repeat_composition


if __name__ == "__main__":
    plt.style.use("seaborn")
    
    ### different cell types, only PR8 datasets ### 
    cell_dfs = list()
    cell_dfnames = list()
    expected_cell_dfs = list()
    for dfname in ["Alnaji2021", "Pelz2021", "Wang2020", "EBI2020"]:
        df = load_dataset(dfname)
        strain = DATASET_STRAIN_DICT[dfname]
        if "Cell" in df.columns:
            for cell_type in df["Cell"].unique():
                c_df = df[df["Cell"] == cell_type].copy()

                cell_dfs.append(preprocess(strain, join_data(c_df), CUTOFF))
                cell_dfnames.append(f"{dfname} {cell_type}")
                expected_cell_dfs.append(preprocess(strain, generate_expected_data(strain, df), 1))
        else:
            cell_dfs.append(preprocess(DATASET_STRAIN_DICT[dfname], join_data(df), CUTOFF))
            cell_dfnames.append(f"{dfname}")
            expected_cell_dfs.append(preprocess(strain, generate_expected_data(strain, df), 1))

    folder = "cell_datasets"
    plot_distribution_over_segments(cell_dfs, cell_dfnames, folder=folder)
    calculate_deletion_shifts(cell_dfs, cell_dfnames, folder=folder)
    length_distribution_histrogram(cell_dfs, cell_dfnames, folder=folder)
    length_distribution_violinplot(cell_dfs, cell_dfnames, folder=folder)
    plot_nucleotide_ratio_around_deletion_junction_heatmaps(cell_dfs, cell_dfnames, folder=folder)
    plot_direct_repeat_ratio_heatmaps(cell_dfs, cell_dfnames, folder=folder)
    start_vs_end_lengths(cell_dfs, cell_dfnames, limit=600, folder=folder)
    diff_start_end_lengths(cell_dfs, cell_dfnames, folder=folder)
    
    ### expected analyses ###
    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(cell_dfs, cell_dfnames, expected_cell_dfs, folder)
    plot_expected_vs_observed_direct_repeat_heatmaps(cell_dfs, cell_dfnames, expected_cell_dfs, "observed-expected", folder)
    direct_repeat_composition(cell_dfs, cell_dfnames, expected_cell_dfs, folder)
