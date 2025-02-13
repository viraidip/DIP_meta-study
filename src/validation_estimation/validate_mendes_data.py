### THIS IS OUTDATED ###
### WAS USED TO DETECT THE ANOMALIES IN MENDES DATASET ###
### CHECKING IF THE ANOMALIES ARE DUE TO PROCESSING ###
'''
    Compares the data of Mendes 2021 to the generated data.
    Leads to the conclusion that RNA sequences need to be reversed.    
'''
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import load_single_dataset, preprocess
from utils import SEGMENTS, CUTOFF, DATAPATH
from overall_comparision.general_analyses import diff_start_end_lengths


def load_mendes2021_rsc(name: str)-> dict:
    '''
        Loads the data set of Mendes et al. 2021.
        :param name: indicates which dataset to load

        :return: dictionary with strain name as key and data frame as value
    '''
    if name == "v12enriched":
        filename = "Virus-1-2_enriched_junctions.tsv"
    elif name == "v21depleted":
        filename = "Virus-2-1_depleted_junctions.tsv"
    
    file_path = os.path.join(DATAPATH, "RSC_estimation", filename)
    data = pd.read_csv(file_path,
                            header=0,
                            na_values=["", "None"],
                            keep_default_na=False,
                            sep="\t")
    
    return data


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
