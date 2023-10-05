'''
    General functions and global parameters, that are used in different scripts
    of the ML part. Includes functions to load data and generate features.
'''
import os
import sys
import RNA
import json

import numpy as np
import pandas as pd

from Bio.Seq import Seq
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "..")
from utils import load_all, get_sequence, get_seq_len, calculate_direct_repeat
from utils import DATASET_STRAIN_DICT, DATAPATH


CHARS = 'ACGU'
CHARS_COUNT = len(CHARS)
MAX_LEN = 2361 # B Lee

### feature generation ###
def generate_features(df: pd.DataFrame,
                      features: list
                      )-> Tuple[pd.DataFrame, list]:
    '''
        Main function to generate/load the features.
        :param df: data frame including the deletion site data
        :param features: list indicating which features to include
        :param load_precalc: if True features are loaded from precalculated
                             file

        :return: data frame including the features,
                 list with the names of the columns for the features
    '''
    if "Segment" in features:
        df, segment_cols = segment_ohe(df)
    if "DI_Length" in features:
        df["DI_Length"] = df.apply(get_dirna_length, axis=1)
    if "Direct_repeat" in features:
        df["Direct_repeat"] = df.apply(get_direct_repeat_length, axis=1)
    if "Junction" in features:
        df, junction_start_cols = junction_site_ohe(df, "Start")
        df, junction_end_cols = junction_site_ohe(df, "End")
    if "3_5_ratio" in features:
        df["3_5_ratio"] = df.apply(get_3_to_5_ratio, axis=1)
    if "length_proportion" in features:
        df["length_proportion"] = df.apply(get_length_proportion, axis=1)
    if "full_sequence" in features:
        df, sequence_cols = full_sequence_ohe(df)
    if "delta_G" in features:
        df["delta_G"] = df.apply(get_delta_G, axis=1)
    if "Peptide_Length" in features:
        df["Peptide_Length"] = df.apply(get_peptide_length, axis=1)
    if "Inframe_Deletion" in features:
        df["Inframe_Deletion"] = df.apply(get_inframe_deletion, axis=1)


    df["NGS_read_count"] = df["NGS_read_count"].astype(float)
    df = df[df["NGS_read_count"] > 0].copy()
    df["NGS_log"] = np.log(df["NGS_read_count"]).astype(float)
    df["NGS_norm"] = df["NGS_read_count"]/max(df["NGS_read_count"])
    df["NGS_log_norm"] = df["NGS_log"]/max(df["NGS_log"])

    df.write_csv(DATAPATH, "ML", "features.csv", index=False)

def segment_ohe(df: pd.DataFrame)-> Tuple[pd.DataFrame, list]:
    '''
        Converts the column with segment names into an one hot encoding.
        :param df: data frame including a row called 'Segment'

        :return: Tuple with two entries:
                    data frame including original data and OHE data
                    list with the column names of the OHE
    '''
    ohe = OneHotEncoder()
    segment_df = pd.DataFrame(ohe.fit_transform(df[["Segment"]]).toarray())
    ohe_cols = ohe.get_feature_names_out().tolist()
    segment_df.columns = ohe_cols
    df = df.join(segment_df)
    return df, ohe_cols

def get_dirna_length(row: pd.Series)-> int:
    '''
        Calculates the length of the DI RNA sequence given a row of a data
        frame with the necessary data.
        :param row: data frame row including Strain, Segment, Start, and End
        
        :return: length of DI RNA sequence
    '''
    seq_len = get_seq_len(row["Strain"], row["Segment"])
    return row["Start"] + (seq_len - row["End"] + 1)

def get_direct_repeat_length(row: pd.Series)-> int:
    '''
        Calculates the length of the direct repeat given a row of a data frame
        with the necessary data.
        :param row: data frame row including Strain, Segment, Start, and End
        
        :return: length of direct repeat
    '''
    seq = get_sequence(row["Strain"], row["Segment"])
    s = row["Start"]
    e = row["End"]
    n, _ = calculate_direct_repeat(seq, s, e, 15)
    return n

def junction_site_ohe(df: pd.DataFrame,
                      position: str
                      )-> Tuple[pd.DataFrame, list]:
    '''
        Gets the sequence around the start or end of a given deletion site and
        converts the sequence into an one hot encoding.
        :param df: data frame including Start, End, Strain, and Segment
        :param position: is either 'Start' or 'End' to indicate which site
        
        :return: Tuple with two entries:
                    data frame including original data and OHE data
                    list with the column names of the OHE
    '''
    # initializing matrix
    n = df.shape[0]
    res = np.zeros((n, CHARS_COUNT * 10), dtype=np.uint8)

    # getting sequence window for each row and do OHE
    for i, r in df.iterrows():
        s = r[position]
        seq = get_sequence(r["Strain"], r["Segment"])
        seq = seq[s-5:s+5]
        # Write down OHE in numpy array
        for j, char in enumerate(seq):
            pos = CHARS.rfind(char)
            res[i][j*CHARS_COUNT+pos] = 1

    # format as data frame and create columns names of OHE
    encoded_df = pd.DataFrame(res)
    col_names = [f"{position}_{i}_{ch}" for i in range(1, 11) for ch in CHARS]
    encoded_df.columns = col_names
    df = df.join(encoded_df)

    return df, col_names

def get_3_to_5_ratio(row: pd.Series)-> float:
    '''
        Calculates the proportion of the 3' sequence to the 5' sequence given
        a row of a data frame.
        :param row: data frame row including Strain, Segment, Start, and End
        
        :return: ratio of 3' to 5' sequence length
    '''
    seq_len = get_seq_len(row["Strain"], row["Segment"])
    len3 = row["Start"]
    len5 = seq_len - row["End"] + 1
    # this is not a ratio anymore but the difference
    return len3 - len5

def get_length_proportion(row: pd.Series)-> float:
    '''
        Calculates the proportion of the length of the DI RNA sequence to the
        full length sequence given a row of a data frame.
        :param row: data frame row including Strain, Segment, Start, and End
        
        :return: ratio of DI RNA lenght to full length sequence
    '''
    seq_len = get_seq_len(row["Strain"], row["Segment"])
    dirna_len = row["Start"] + (seq_len - row["End"] + 1)
    return dirna_len/seq_len

def full_sequence_ohe(df: pd.DataFrame)-> Tuple[pd.DataFrame, list]:
    '''
        Gets the whole sequence as an one hot encoding. Sequences get
        normalized to the longest sequence length by adding * at the end
        :param df: data frame including Start, End, Strain, and Segment
        
        :return: Tuple with two entries:
                    data frame including original data and OHE data
                    list with the column names of the OHE
    '''
    # defining initializing matrix
    n = df.shape[0]
    res = np.zeros((n, CHARS_COUNT * MAX_LEN), dtype=np.uint8)

    # getting sequence window for each row and do OHE
    for i, r in df.iterrows():
        seq = get_sequence(r["Strain"], r["Segment"])
        seq = seq + "*" * (MAX_LEN - len(seq))
        # Write down OHE in numpy array
        for j, char in enumerate(seq):
            pos = CHARS.rfind(char)
            res[i][j*CHARS_COUNT+pos] = 1

    # format as data frame and create columns names of OHE
    encoded_df = pd.DataFrame(res)
    col_names = [f"{i}_{ch}" for i in range(1, MAX_LEN+1) for ch in CHARS]
    encoded_df.columns = col_names
    df = df.join(encoded_df)

    return df, col_names

def get_delta_G(row: pd.Series)-> float:
    '''
        Calculate Gibbs freee energy for a DI RNA candidate and return it
        normalized by its length.
        :param row: data frame row including Strain, Segment, Start, and End
        
        :return: ratio of DI RNA lenght to full length sequence
    '''
    seq = get_sequence(row["Strain"], row["Segment"])
    del_seq = seq[:row["Start"]] + seq[row["End"]-1:]
    mfe = RNA.fold_compound(del_seq).mfe()[1]
    return mfe/len(del_seq)

def get_peptide_length(row: pd.Series)-> float:
    '''
        Translates a given RNA sequence into amino acids and calculates the
        length of the resulting peptide.
        :param row: data frame row including Strain, Segment, Start, and End
        
        :return: length of resulting peptid
    '''
    strain = row["Strain"]
    segment = row["Segment"]
    seq = get_sequence(strain, segment)
    f = open(os.path.join(DATAPATH, "strain_segment_fastas", "translation_indices.json"))
    indices = json.load(f)
    f.close()
    s = indices[strain][segment]["start"]-1
    e = indices[strain][segment]["end"]
    seq_obj = Seq(seq[s:row["Start"]] + seq[row["End"]-1:e])
    pep_seq = seq_obj.translate(to_stop=True)
    return len(pep_seq)

def get_inframe_deletion(row: pd.Series)-> int:
    '''
        Checks for a DI candidate if the deletion is inframe. This is achieved
        by applying modulo 3.
        :param row: data frame row including Strain, Segment, Start, and End    

        :return: 0 if inframe deletion
                 -1 or 1 if no inframe deletion
    '''
    m = (row["End"] - 1 - row["Start"]) % 3
    if m == 2:
        m = -1
    return m


if __name__ == "__main__":
    dfs, dfnames = load_all()
    for df, dfname in zip(dfs, dfnames):
        df["dataset_name"] = dfname
        df["strain"] = DATASET_STRAIN_DICT[dfname]
    concat_df = pd.concat(dfs)

    features = list(["Segment",
                     "DI_Length",
                     "Direct_repeat",
                     "Junction",
                     "3_5_ratio",
                     "length_proportion",
                     "full_sequence",
                     "delta_G",
                     "Peptide_Length",
                     "Inframe_Deletion"
                     ])
    generate_features(concat_df, features)
