'''

'''
import os
import sys

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from collections import Counter

sys.path.insert(0, "..")
from utils import load_all
from utils import SEGMENTS
from overall_comparision.general_analyses import calc_DI_lengths


def compare_DI_lengths(vitro_dict, vivo_dict):
    '''
    
    '''
    for s in SEGMENTS:
        x_vitro = [key for key, value in vitro_dict[s].items() for _ in range(value)]
        x_vivo = [key for key, value in vivo_dict[s].items() for _ in range(value)]

        plt.hist(x_vitro, alpha=0.5, label="vitro", bins=20, density=True)
        plt.hist(x_vivo, alpha=0.5, label="vivo", bins=20, density=True)
    
        plt.title(s)

        plt.show()
        

if __name__ == "__main__":
    plt.style.use("seaborn")

    vitro_dfnames = ["Alnaji2021", "Pelz2021"]
    vitro_dfs, _ = load_all(vitro_dfnames)
    vitro_lengths_dict = calc_DI_lengths(vitro_dfs, vitro_dfnames)
    vitro_dict = dict({"PB2": Counter(), "PB1": Counter(), "PA": Counter(), "HA": Counter(), "NP": Counter(), "NA": Counter(), "M": Counter(), "NS": Counter()})
    for d in vitro_lengths_dict.values():
        for s in vitro_dict.keys():
            vitro_dict[s] += Counter(d[s])
    
    vivo_dfnames = ["Southgate2019"]
    vivo_dfs, _ = load_all(vivo_dfnames)
    vivo_lengths_dict = calc_DI_lengths(vivo_dfs, vivo_dfnames)
    vivo_dict = dict({"PB2": Counter(), "PB1": Counter(), "PA": Counter(), "HA": Counter(), "NP": Counter(), "NA": Counter(), "M": Counter(), "NS": Counter()})
    for d in vivo_lengths_dict.values():
        for s in vivo_dict.keys():
            vivo_dict[s] += Counter(d[s])
    
    compare_DI_lengths(vitro_dict, vivo_dict)
