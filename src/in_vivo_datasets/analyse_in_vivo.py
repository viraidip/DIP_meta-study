'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_all, load_dataset, preprocess
from utils import DATASET_STRAIN_DICT, CUTOFF
from overall_comparision.general_analyses import plot_distribution_over_segments, calculate_deletion_shifts, length_distribution_histrogram, length_distribution_violinplot, plot_nucleotide_ratio_around_deletion_junction_heatmaps, plot_direct_repeat_ratio_heatmaps, start_vs_end_lengths, diff_start_end_lengths
    

def patient_intersection(dfs, dfnames):
    '''
    
    '''
    for df, dfname in zip(dfs, dfnames):
        print(df.shape)
        count_occurrences = df.groupby("key").size().reset_index(name="count")

        # Display the result
        print(count_occurrences.sort_values(by='count', ascending=False))


if __name__ == "__main__":
    plt.style.use("seaborn")
    
    '''
    in_vivo_dfnames = ["Wang2023", "Penn2022", "Lui2019", "WRA2021_A", "Rattanaburi2022_H3N2", "WRA2021_B", "Sheng2018", "Lauring2019", "Southgate2019"]
    in_vivo_dfs, _ = load_all(in_vivo_dfnames)
    
    folder = "in_vivo_datasets"
    plot_distribution_over_segments(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    calculate_deletion_shifts(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    length_distribution_histrogram(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    length_distribution_violinplot(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    plot_nucleotide_ratio_around_deletion_junction_heatmaps(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    plot_direct_repeat_ratio_heatmaps(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    start_vs_end_lengths(in_vivo_dfs, in_vivo_dfnames, limit=600, folder=folder)
    diff_start_end_lengths(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    '''
    ### intersections of patient data ###
    patient_dfnames = ["WRA2021_B", "WRA2021_B_yamagata", "Lauring2019", "Southgate2019"]
    patient_dfs = list()
    for dfname in patient_dfnames:
        strain = DATASET_STRAIN_DICT[dfname]
        df = load_dataset(dfname)
        patient_dfs.append(preprocess(strain, df, CUTOFF))
    
    patient_intersection(patient_dfs, patient_dfnames)