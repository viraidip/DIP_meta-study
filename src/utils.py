'''
    General functions and global parameters, that are used in different scripts
'''
import os
import random
import json

import numpy as np
import pandas as pd
import seaborn as sns

from typing import Tuple
from Bio import SeqIO


# load config and assign values to global variables
DATAPATH = json.load(open("../../.config.json"))["DATAPATH"]
RESULTSPATH = json.load(open("../../.config.json"))["RESULTSPATH"]

# segments, nuclotides, and strains
CUTOFF = 15
SEGMENTS = list(["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"])
NUCLEOTIDES = dict({"A": "Adenine", "C": "Cytosin", "G": "Guanine", "U": "Uracil"})
STRAINS = dict({"Cal07": "A/California/07/2009",
                "NC": "A/New Caledonia/20-JY2/1999",
                "Perth": "A/Perth/16/2009",
                "BLEE": "B/Lee/1940",
                "PR8": "A/Puerto Rico/8/1934",
                "WSN": "A/WSN/1933"
                })

SEGMENT_DICTS = dict({
    "PR8": dict({
        "AF389115.1": "PB2",
        "AF389116.1": "PB1",
        "AF389117.1": "PA",
        "AF389118.1": "HA",
        "AF389119.1": "NP",
        "AF389120.1": "NA",
        "AF389121.1": "M",
        "AF389122.1": "NS"
    }),
    "Cal07": dict({
        "CY121687.1": "PB2",
        "CY121686.1": "PB1",
        "CY121685.1": "PA",
        "CY121680.1": "HA",
        "CY121683.1": "NP",
        "CY121682.1": "NA",
        "CY121681.1": "M",
        "CY121684.1": "NS"
    }),
    "NC": dict({
        "CY147325.1": "PB2",
        "CY147324.1": "PB1",
        "CY147323.1": "PA",
        "CY147318.1": "HA",
        "CY147321.1": "NP",
        "CY147320.1": "NA",
        "CY147319.1": "M",
        "CY147322.1": "NS"
    }),
    "Perth": dict({
        "KJ609203.1": "PB2",
        "KJ609204.1": "PB1",
        "KJ609205.1": "PA",
        "KJ609206.1": "HA",
        "KJ609207.1": "NP",
        "KJ609208.1": "NA",
        "KJ609209.1": "M",
        "KJ609210.1": "NS"
    }),
    "BLEE": dict({
        "CY115118.1": "PB2",
        "CY115117.1": "PB1",
        "CY115116.1": "PA",
        "CY115111.1": "HA",
        "CY115114.1": "NP",
        "CY115113.1": "NA",
        "CY115112.1": "M",
        "CY115115.1": "NS"
    }),
    "WSN": dict({
        "CY034139.1": "PB2",
        "CY034138.1": "PB1",
        "CY034137.1": "PA",
        "CY034132.1": "HA",
        "CY034135.1": "NP",
        "CY034134.1": "NA",
        "CY034133.1": "M",
        "CY034136.1": "NS"
    }),
    "Anhui": dict({
        "439504": "PB2",
        "439508": "PB1",
        "439503": "PA",
        "439507": "HA",
        "439505": "NP",
        "439509": "NA",
        "439506": "M",
        "439510": "NS"
    })
})

# global colors for plotting
COLORS = dict({"A": "deepskyblue", "C": "gold", "G": "springgreen", "U": "salmon"})

# parameters for the sampling
QUANT = 0.1
N_SAMPLES = 2000

def load_dataset(exp: str, acc: str, segment_dict: dict)-> pd.DataFrame:
    '''
    
    '''
    path = os.path.join(DATAPATH, exp, f"{exp}_{acc}.csv")
    df = pd.read_csv(path)
    df["Segment"] = df["Segment"].replace(segment_dict)

    return df

def load_alnaji2019(strain: str):
    '''

    '''
    acc_num_dict = dict({
        "Cal07": dict({"SRR8754522": dict({"Lineage": "1", "Passage": "6"}),
                       "SRR8754523": dict({"Lineage": "2", "Passage": "6"})
                       }),
        "Cal07_time": dict({"SRR8754531": dict({"Lineage": "1", "Passage": "6"}),
                            "SRR8754532": dict({"Lineage": "1", "Passage": "3"}),
                            "SRR8754533": dict({"Lineage": "1", "Passage": "1"})
                            }),
        "NC": dict({"SRR8754513": dict({"Lineage": "2", "Passage": "1"}),
                    "SRR8754514": dict({"Lineage": "1", "Passage": "1"}),
                    "SRR8754527": dict({"Lineage": "1", "Passage": "6"}),
                    "SRR8754538": dict({"Lineage": "2", "Passage": "6"})
                    }),
        "Perth": dict({"SRR8754517": dict({"Lineage": "2", "Passage": "8"}),
                       "SRR8754524": dict({"Lineage": "1", "Passage": "4"}),
                       "SRR8754525": dict({"Lineage": "2", "Passage": "4"}),
                       "SRR8754526": dict({"Lineage": "1", "Passage": "8"})
                       }),
        "BLEE": dict({"SRR8754507": dict({"Lineage": "1", "Passage": "8"}),
                      "SRR8754508": dict({"Lineage": "2", "Passage": "7"}),
                      "SRR8754509": dict({"Lineage": "1", "Passage": "7"}),
                      "SRR8754516": dict({"Lineage": "2", "Passage": "8"})
                      })
    })

    acc_nums = acc_num_dict[strain]

    if strain == "Cal07_time":
        strain = "Cal07"

    dfs = list()
    for acc_num, meta in acc_nums.items():
        df = load_dataset("Alnaji2019", acc_num, SEGMENT_DICTS[strain])
        df["Lineage"] = meta["Lineage"]
        df["Passage"] = meta["Passage"]
        dfs.append(df)
    concat_df = pd.concat(dfs)

    return concat_df

def load_alnaji2021():
    '''

    '''
    acc_nums = dict({
        "SRR14352106": dict({"Replicate": "C", "Time": "24hpi"}),
        "SRR14352107": dict({"Replicate": "B", "Time": "24hpi"}),
        "SRR14352108": dict({"Replicate": "A", "Time": "24hpi"}),
        "SRR14352109": dict({"Replicate": "C", "Time": "6hpi"}),
        "SRR14352110": dict({"Replicate": "B", "Time": "6hpi"}),
        "SRR14352111": dict({"Replicate": "A", "Time": "6hpi"}),
        "SRR14352112": dict({"Replicate": "C", "Time": "3hpi"}),
        "SRR14352113": dict({"Replicate": "X", "Time": "0hpi"}),
        "SRR14352116": dict({"Replicate": "B", "Time": "3hpi"}),
        "SRR14352117": dict({"Replicate": "A", "Time": "3hpi"})
    })

    dfs = list()
    for acc_num, meta in acc_nums.items():
        df = load_dataset("Alnaji2021", acc_num, SEGMENT_DICTS["PR8"])
        df["Replicate"] = meta["Replicate"]
        df["Time"] = meta["Time"]
        dfs.append(df)
    concat_df = pd.concat(dfs)

    # filter out seed virus DIs
    concat_df["DI"] = concat_df["Segment"] + "_" + concat_df["Start"].astype(str) + "_" + concat_df["End"].astype(str)
    seed = concat_df[concat_df["Time"] == "0hpi"]["DI"].to_list()
    concat_df = concat_df.loc[~concat_df["DI"].isin(seed)]
    concat_df.drop("DI", inplace=True, axis=1)

    return concat_df

def load_pelz2021():
    '''
    
    '''
    acc_nums = dict({
        "SRR15084902": dict({"Time": "8dpi"}),
        "SRR15084903": dict({"Time": "5.5dpi"}),
        "SRR15084904": dict({"Time": "5dpi"}),
        "SRR15084905": dict({"Time": "4.5dpi"}),
        "SRR15084906": dict({"Time": "4dpi"}),
        "SRR15084907": dict({"Time": "3.5dpi"}),
        "SRR15084908": dict({"Time": "1.4dpi"}),
        "SRR15084909": dict({"Time": "21dpi"}),
        "SRR15084910": dict({"Time": "20.4dpi"}),
        "SRR15084911": dict({"Time": "20dpi"}),
        "SRR15084912": dict({"Time": "19.5dpi"}),
        "SRR15084913": dict({"Time": "1dpi"}),
        "SRR15084914": dict({"Time": "18dpi"}),
        "SRR15084915": dict({"Time": "17.5dpi"}),
        "SRR15084916": dict({"Time": "17dpi"}),
        "SRR15084917": dict({"Time": "16dpi"}),
        "SRR15084918": dict({"Time": "13.5dpi"}),
        "SRR15084919": dict({"Time": "13dpi"}),
        "SRR15084921": dict({"Time": "12.4dpi"}),
        "SRR15084922": dict({"Time": "9.4dpi"}),
        "SRR15084923": dict({"Time": "9dpi"}),
        "SRR15084924": dict({"Time": "0.5dpi"}),
        "SRR15084925": dict({"Time": "seed"})
    })

    dfs = list()
    for acc_num, meta in acc_nums.items():
        df = load_dataset("Pelz2021", acc_num, SEGMENT_DICTS["PR8"])
        df["Time"] = meta["Time"]
        dfs.append(df)
    concat_df = pd.concat(dfs)

    # filter out seed virus DIs
    concat_df["DI"] = concat_df["Segment"] + "_" + concat_df["Start"].astype(str) + "_" + concat_df["End"].astype(str)
    seed = concat_df[concat_df["Time"] == "seed"]["DI"].to_list()
    concat_df = concat_df.loc[~concat_df["DI"].isin(seed)]
    concat_df.drop("DI", inplace=True, axis=1)

    return concat_df

def load_lui2019():
    '''
    
    '''
    df = load_dataset("Lui2019", "SRR8949705", SEGMENT_DICTS["Anhui"])
    return df

def load_mendes2021():
    '''

    '''
    acc_nums = dict({
        "SRR15720520": dict({"Status": "enriched",
                             "Virus": "1",
                             "Replicate": "1"}),
        "SRR15720521": dict({"Status": "enriched",
                             "Virus": "1",
                             "Replicate": "2"}),
        "SRR15720522": dict({"Status": "enriched",
                             "Virus": "2",
                             "Replicate": "1"}),
        "SRR15720523": dict({"Status": "enriched",
                             "Virus": "2",
                             "Replicate": "2"}),
        "SRR15720524": dict({"Status": "depleted",
                             "Virus": "1",
                             "Replicate": "1"}),
        "SRR15720525": dict({"Status": "depleted",
                             "Virus": "1",
                             "Replicate": "2"}),
        "SRR15720526": dict({"Status": "depleted",
                             "Virus": "2",
                             "Replicate": "1"}),
        "SRR15720527": dict({"Status": "depleted",
                             "Virus": "2",
                             "Replicate": "2"})
    })

    dfs = list()
    for acc_num, meta in acc_nums.items():
        df = load_dataset("Mendes2021", acc_num, SEGMENT_DICTS["PR8"])
        df["Status"] = meta["Status"]
        df["Virus"] = meta["Virus"]
        df["Replicate"] = meta["Replicate"]
        dfs.append(df)
    concat_df = pd.concat(dfs)

    return concat_df


def join_data(df: pd.DataFrame)-> pd.DataFrame:
    '''
    
    '''
    df = df.groupby(["Segment", "Start", "End"]).sum(["NGS_read_count"]).reset_index()
    return df

def get_sequence(strain: str, seg: str, full: bool=False)-> object:
    '''
        Loads a DNA sequence given the strain and segment.
        :param strain: name of the strain
        :param seg: name of the segment
        :param full: if True the whole Biopython Seq Object is returned
                     if False a string object is returned

        :return: Biopython Seq Object or str() of the sequence
    '''
    fasta_file = os.path.join(DATAPATH, "strain_segment_fastas", strain, f"{seg}.fasta")
    seq_obj = SeqIO.read(fasta_file, "fasta")
    if full:
        return seq_obj
    else:
        return str(seq_obj.seq.transcribe())

def get_seq_len(strain: str, seg: str)-> int:
    '''
        Calculates the length of a specific sequence given the strain and
        segment.
        :param strain: name of the strain
        :param seg: name of the segment

        :return: length of the sequence as int
    '''
    return len(get_sequence(strain, seg))

def get_p_value_symbol(p: float)-> str:
    '''
        Indicates the statistical significance by letters. Is used for plots.
        :param p: p-value of the test

        :return: letter indicating the significance level
    '''
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def generate_sampling_data(seq: str, s: Tuple[int, int], e: Tuple[int, int],  n: int) -> object:
    '''
        Generates sampling data by creating random start and end points for
        artificial deletion sites. Generated data is used to calculate the
        expected values.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion site
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion site
        :param n: number of samples to generate

        :return: dataframe with the artifical data set
    '''
    sampling = dict({"Start": [], "End": []})
    for _ in range(n):
        sampling["Start"].append(random.randint(s[0], s[1]))
        sampling["End"].append(random.randint(e[0], e[1]))
    return pd.DataFrame(data=sampling)

def create_sequence_library(data_dict: dict)-> dict:
    '''
        Gets the raw loaded sequence data, which is a dict over all strains.
        In each dict the value is a data frame including DI RNA candidates.
        Creates the DI RNA sequence and adds it to the data frame.
        :param data_dict: dictionary key is strain names, value is df of DI RNA
                          candiates

        :return: dictionary with key for each strain. Value is a pandas df.
    '''
    for k, v in data_dict.items():
        del_seq_list = list()
        for i, row in v.iterrows():
            full_seq = get_sequence(k, row["Segment"])
            del_seq = full_seq[:row["Start"]] + full_seq[row["End"]-1:]
            del_seq_list.append(del_seq)

        data_dict[k]["DIRNASequence"] = del_seq_list

    return data_dict


#####################
### DIRECT REPATS ###
#####################
def calculate_direct_repeat(seq: str,
                            s: int,
                            e: int,
                            w_len: int
                            )-> Tuple[int, str]:
    '''
        Counts the number of overlapping nucleotides directly before start and
        end of junction site --> direct repeats
        :param seq: nucleotide sequence
        :param w_len: length of window to be searched
        :param s: start point
        :param e: end point

        :return: Tuple with two entries:
                    Integer giving the number of overlapping nucleotides
                    String of the overlapping nucleotides
    '''
    counter = 0

    start_window = seq[s-w_len: s]
    end_window = seq[e-1-w_len: e-1]
        
    #if they are the same return directly to avoid off-by-one error
    if start_window == end_window:
        return len(start_window), start_window

    if len(seq) < e:
        return 0, "_"
        
    for i in range(len(end_window) - 1, -1, -1):
        if start_window[i] == end_window[i]:
            counter += 1
        else:
            break
    overlap_seq = str(start_window[i+1:w_len])

    assert counter == len(overlap_seq), f"{counter=}, {len(overlap_seq)}"
    if len(overlap_seq) == 0:
        overlap_seq = "_"

    return counter, overlap_seq

def count_direct_repeats_overall(df: pd.DataFrame,
                                 seq: str
                                 )-> Tuple[dict, dict]:
    '''
        Calculates the number of direct repeats for each data point.
        :param df: dataframe with sequence and junction site data
        :param seq: RNA sequence of the given segement and strain
        :param mode: states which calculation mode is used in 
                     calculate_overlapping_nucleotides() check there for info

        :return: Tuple including a dict with the count of the length of
                 overlapping sequences and a dict with the overlapping
                 sequences and their count.
    '''
    w_len = 5
    nuc_overlap_dict = dict({i: 0 for i in range(0, w_len+1)})
    overlap_seq_dict = dict()
 
    for i, row in df.iterrows():
        s = row["Start"]
        e = row["End"]
        idx, overlap_seq = calculate_direct_repeat(seq, s, e, w_len)
        nuc_overlap_dict[idx] += 1
        if overlap_seq in overlap_seq_dict:
            overlap_seq_dict[overlap_seq] += 1
        else:
            overlap_seq_dict[overlap_seq] = 1

    return nuc_overlap_dict, overlap_seq_dict


def include_correction(nuc_overlap_dict: dict)-> dict:
    '''
        Adds a correction to the counting of the direct repeats. This is due to
        the fact that at these sites the observations get merged towards higher
        lengths of the direct repeat.
        :param nuc_overlap_dict: counting dict of the direct repeats

        :return: corrected counts of the direct repeats (same structure as
                 input
    '''
    new = dict()
    for idx in nuc_overlap_dict.keys():
        orig_value = nuc_overlap_dict[idx]
        if orig_value != 0:
            divided_value = orig_value/(idx+1)
            new[idx] = divided_value
            for idx_2 in range(0, idx):
                new[idx_2] = new[idx_2] + divided_value
        else:
            new[idx] = 0

    return new


#############################
### NUCLEOTIDE ENRICHMENT ###
#############################
def count_nucleotide_occurrence(seq: str,
                                p: int
                                )-> dict:
    '''
        Counts the number of nucleotides next to a given point.
        Goes 5 steps in both directions.
        :param seq: whole RNA sequence
        :param p: point on the sequence where to count

        :return: returns a counter dict with an entry for each nucleotide. In
                 each entry the counter for each position is given.
    '''
    window = seq[p-5:p+5]
    r_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})

    for i, char in enumerate(window):
        r_dict[char][i] = 1
    return r_dict

def count_nucleotide_occurrence_overall(df: pd.DataFrame,
                                        seq: str
                                        )-> Tuple[dict, dict]:
    '''
        Counts the occurrence of each nucleotide at different positions around
        the junction site
        :param df: dataframe with sequence and junction site data
        :param seq: rna sequence where to count the occurrence

        :return: tupel with two entries:
                    dict with nucleotide count for start site
                    dict with nucleotide count for end site
    '''

    count_start_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    count_end_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    normalize = 0

    for i, row in df.iterrows():
        seq_start_dict = count_nucleotide_occurrence(seq, row["Start"]) 
        seq_end_dict = count_nucleotide_occurrence(seq, row["End"]-1)
        normalize += 1
        for nuc in count_start_dict.keys():
            count_start_dict[nuc] += seq_start_dict[nuc]
            count_end_dict[nuc] += seq_end_dict[nuc]

    return count_start_dict, count_end_dict

#####################
### expected data ###
#####################
def generate_expected_data(k, v):
    '''
    
    '''
    for seg in SEGMENTS:
        df = v.loc[v["Segment"] == seg]
        if len(df) == 0:
            continue
        seq = get_sequence(k, seg)
        s = (int(df.Start.quantile(QUANT)), int(df.Start.quantile(1-QUANT)))
        e = (int(df.End.quantile(QUANT)), int(df.End.quantile(1-QUANT)))
        # skip if there is no range given
        # this would lead to oversampling of a single position
        if s[0] == s[1] or e[0] == e[1]:
            continue
        if "samp_df" in locals():
            temp_df = generate_sampling_data(s, e, N_SAMPLES)
            temp_df["Segment"] = seg
            samp_df = pd.concat([samp_df, temp_df], ignore_index=True)
        else:
            samp_df = generate_sampling_data(s, e, N_SAMPLES)
            samp_df["Segment"] = seg
    
    return samp_df

def generate_sampling_data(s: Tuple[int, int], e: Tuple[int, int],  n: int) -> object:
    '''
        Generates sampling data by creating random start and end points for
        artificial deletion sites. Generated data is used to calculate the
        expected values.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion site
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion site
        :param n: number of samples to generate

        :return: dataframe with the artifical data set
    '''
    sampling = dict({"Start": [], "End": []})
    for _ in range(n):
        sampling["Start"].append(random.randint(s[0], s[1]))
        sampling["End"].append(random.randint(e[0], e[1]))
    return pd.DataFrame(data=sampling)


#######################
### Data processing ###
#######################
def create_nucleotide_ratio_matrix(df, col):
    '''
    
    '''
    probability_matrix = pd.DataFrame(columns=NUCLEOTIDES.keys())
    seq_matrix = df.filter([col], axis=1)
    seq_matrix = seq_matrix[col].str.split("", expand=True)
    # drop first and last column
    seq_matrix = seq_matrix.drop([0, len(seq_matrix.columns)-1], axis=1)
    
    for n in NUCLEOTIDES.keys():
        probability_matrix[n] = seq_matrix.apply(lambda x: dict(x.value_counts()).get(n,0)/len(x), axis=0)

    return probability_matrix
def plot_heatmap(y,x,vals,ax, format=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar=False,cbar_ax=None, cbar_kws=None):
    '''
        Plot heatmap for values in vals, with x (columns) and y (rows) as labels.
    '''
    df = pd.DataFrame({"x":x,"y":y,"vals":vals})
    df = pd.pivot_table(df, index="x", columns="y", values="vals", sort=False)
    ax = sns.heatmap(df, fmt=format, annot=True, vmin=vmin, vmax=vmax, ax=ax, cbar=cbar, cmap=cmap, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    return ax
def sequence_df(df, strain, isize=5):
    '''Generate a DataFrame with sequence information.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the DIP candidates in the "key" column. Nomenclature: {seg}_{start}_{end}
        isize (int, optional): The size of the sequence before and after the start and end positions. Default is 5.

    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - "key": The original key from the input DataFrame.
            - "seg": The segment extracted from the key.
            - "start": The start position extracted from the key.
            - "end": The end position extracted from the key.
            - "seq": The dip sequence obtained from the "key".
            - "deleted_sequence": The deleted sequence obtained from the "key".
            - "isize": The specified size for the before and after sequences.
            - "seq_before_start": The sequence before the start position of length "isize".
            - "seq_after_start": The sequence after the start position of length "isize".
            - "seq_before_end": The sequence before the end position of length "isize".
            - "seq_after_end": The sequence after the end position of length "isize".
    '''
    res_df = pd.DataFrame(columns=["key","Segment", "Start","End","seq", "deleted_sequence", "isize", "full_seq", "Strain", "seq_around_deletion_junction"])
    for k in df.key:
        seq, seq_head, seq_foot = get_dip_sequence(k, strain)
        start = int(k.split("_")[1].split("_")[0])
        end = int(k.split("_")[2])
        seg = k.split("_")[0]
        full_seq = get_sequence(strain, seg)
        deleted_seq = get_deleted_sequence(k, strain)
        seq_before_start = seq_head[-isize:]
        seq_after_start = deleted_seq[:isize]
        seq_before_end = deleted_seq[-isize:]
        seq_after_end = seq_foot[:isize]

        seq_around_deletion_junction = seq_before_start + seq_after_start + seq_before_end + seq_after_end
        res_df = pd.concat([res_df, pd.DataFrame({"key":k, "Segment":seg, "Start":start, "End":end, "seq":seq, "isize":isize, "full_seq": full_seq, "Strain": strain,
                                "deleted_sequence":deleted_seq, "seq_around_deletion_junction": seq_around_deletion_junction}, index=[0])], ignore_index=True)
    return res_df
def preprocess(strain, df, cutoff):
    '''
    
    '''
    df = df[df["NGS_read_count"] >= cutoff].copy()
    df["key"] = df["Segment"] + "_" + df["Start"].map(str) + "_" + df["End"].map(str)
    return sequence_df(df, strain)
def get_deleted_sequence(dip_id, strain):
    '''
    return the sequence of a dip_id

    Args:
        dip_id (str): the id of the dip

    Returns:
        str: the sequence of the dip
    '''
    seg, start, end = dip_id.split("_")
    seq = get_sequence(strain, seg)
    return seq[int(start):int(end)-1]
def get_dip_sequence(dip_id, strain):
    '''
    
    '''
    seg, start, end = dip_id.split("_")
    fl_seq = get_sequence(strain, seg)
    seq_head = fl_seq[:int(start)]
    seq_foot = fl_seq[int(end)-1:]
    del_length = int(end)-int(start)
    return seq_head + "*"*del_length + seq_foot, seq_head, seq_foot