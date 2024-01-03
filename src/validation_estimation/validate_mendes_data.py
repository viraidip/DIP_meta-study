### THIS IS OUTDATED ###
### WAS USED TO DETECT THE ANOMALIES IN MENDES DATASET ###
### CHECKING IF THE ANOMALIES ARE DUE TO PROCESSING ###
'''
    Compares the data of Mendes 2021 to the generated data.
    Leads to the conclusion that RNA sequences need to be reversed.    
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_single_dataset, preprocess
from utils import SEGMENTS, CUTOFF
from RSC_estimation import load_mendes2021_rsc
from overall_comparision.general_analyses import diff_start_end_lengths


if __name__ == "__main__":
    plt.style.use("seaborn")
    dfs = list()
    dfnames = list()

    strain = "WSN"
    ### Mendes 2021 ###
    v12enriched = load_single_dataset("Mendes2021", "SRR15720521", dict({s: s for s in SEGMENTS}))
    dfs.append(preprocess(strain, v12enriched, CUTOFF))
    dfnames.append("v12enriched")

    orig_mendes = load_mendes2021_rsc("v12enriched")
    dfs.append(preprocess(strain, orig_mendes, CUTOFF))
    dfnames.append("orig. v12enriched") 

    v21depleted = load_single_dataset("Mendes2021", "SRR15720526", dict({s: s for s in SEGMENTS}))
    dfs.append(preprocess(strain, v21depleted, CUTOFF))
    dfnames.append("v21depl")
  
    orig_mendes = load_mendes2021_rsc("v21depleted")
    dfs.append(preprocess(strain, orig_mendes, CUTOFF))
    dfnames.append("orig. v21depl")

    diff_start_end_lengths(dfs, dfnames, folder="validation_estimation")
