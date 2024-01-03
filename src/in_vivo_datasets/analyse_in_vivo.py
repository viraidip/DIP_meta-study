'''
    Run same analyses as in 'overall_comparision' folder, but only with the in
    vivo patient and mouse datasets.
'''
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_all, get_dataset_names
from overall_comparision.general_analyses import plot_distribution_over_segments, calculate_deletion_shifts, length_distribution_histrogram, length_distribution_violinplot, plot_nucleotide_ratio_around_deletion_junction_heatmaps, plot_direct_repeat_ratio_heatmaps, start_vs_end_lengths, diff_start_end_lengths
from overall_comparision.compare_expected import plot_expected_vs_observed_nucleotide_enrichment_heatmaps, plot_expected_vs_observed_direct_repeat_heatmaps


if __name__ == "__main__":
    plt.style.use("seaborn")

### in vivo mouse data ###
    folder = "in_vivo_mouse"
    in_vivo_dfnames = get_dataset_names(cutoff=50, selection="in vivo mouse")
    in_vivo_dfs, expected_in_vivo_dfs = load_all(in_vivo_dfnames, expected=True)

    # run basic analyses
    plot_distribution_over_segments(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    calculate_deletion_shifts(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    length_distribution_histrogram(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    length_distribution_violinplot(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    plot_nucleotide_ratio_around_deletion_junction_heatmaps(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    plot_direct_repeat_ratio_heatmaps(in_vivo_dfs, in_vivo_dfnames, folder=folder)
    start_vs_end_lengths(in_vivo_dfs, in_vivo_dfnames, limit=600, folder=folder)
    diff_start_end_lengths(in_vivo_dfs, in_vivo_dfnames, folder=folder)

    # compare against expected
    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(in_vivo_dfs, in_vivo_dfnames, expected_in_vivo_dfs, "observed-expected", folder)
    plot_expected_vs_observed_direct_repeat_heatmaps(in_vivo_dfs, in_vivo_dfnames, expected_in_vivo_dfs, "observed-expected", folder)
 

### in vivo patient data ###
    folder = "in_vivo_patient"
    patient_dfnames = get_dataset_names(cutoff=0, selection="in vivo human")
    patient_dfs, expected_patient_dfs = load_all(patient_dfnames, expected=True)

    # run basic analyses
    plot_distribution_over_segments(patient_dfs, patient_dfnames, folder=folder)
    calculate_deletion_shifts(patient_dfs, patient_dfnames, folder=folder)
    length_distribution_histrogram(patient_dfs, patient_dfnames, folder=folder)
    length_distribution_violinplot(patient_dfs, patient_dfnames, folder=folder)
    plot_nucleotide_ratio_around_deletion_junction_heatmaps(patient_dfs, patient_dfnames, folder=folder)
    plot_direct_repeat_ratio_heatmaps(patient_dfs, patient_dfnames, folder=folder)
    start_vs_end_lengths(patient_dfs, patient_dfnames, limit=600, folder=folder)
    diff_start_end_lengths(patient_dfs, patient_dfnames, folder=folder)

    # compare against expected
    plot_expected_vs_observed_nucleotide_enrichment_heatmaps(patient_dfs, patient_dfnames, expected_patient_dfs, "observed-expected", folder)
    plot_expected_vs_observed_direct_repeat_heatmaps(patient_dfs, patient_dfnames, expected_patient_dfs, "observed-expected", folder)
