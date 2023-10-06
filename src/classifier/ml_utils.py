'''

'''
import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, "..")
from utils import DATAPATH

def prepare_y(df):
    y = list()
    median = df["NGS_log_norm"].median()

    for row in df.iterrows():
        r = row[1]
        y.append("low" if r["NGS_log_norm"] < median else "high")

    print(f"label 'low':\t{y.count('low')}")
    print(f"label 'high':\t{y.count('high')}")

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)   

    return y

def preprocessing(df):
    '''
    
    '''
    print(df.shape)
    dupl = identify_duplicates(df)
    df = df[~df["DI"].isin(dupl)]
    df.drop(["DI"], axis=1, inplace=True)
    print(df.shape)

    # Separate features (X) and target (y)
    X = df.drop(["Segment", "NGS_read_count", "dataset_name","Strain","NGS_log","NGS_norm","NGS_log_norm"], axis=1).copy()
    y = prepare_y(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (optional but recommended)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test



def identify_duplicates(df):
    '''

    '''
    df["label"] = prepare_y(df)
    df["DI"] = df["Segment"] + "_" + df["Start"].astype(str) + "_" + df["End"].astype(str) + "_" + df["Strain"]

    entries = df["DI"].tolist()
    dupl = [e for e in entries if entries.count(e) > 1]

    '''
    all = len(dupl)
    equal = 0
    unequal = 0

    for d in dupl:
        if len(df[df["DI"] == d]["label"].unique()) == 1:
            equal += 1
        else:
            unequal += 1

    print(all)
    print(equal)
    print(unequal)
    '''
    return dupl


if __name__ == "__main__":
    # load precalculated data from file
    path = os.path.join(DATAPATH, "ML", "features.csv")
    df = pd.read_csv(path, na_values=["", "None"], keep_default_na=False)

    identify_duplicates(df)
