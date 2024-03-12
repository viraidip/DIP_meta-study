'''
    Does a linear and exponential regression for data from Schwartz 2016 and 
    Alnaji 2019. Data is normalized by sum of y values for all data sets.
    Expected value is calculated by dividing length of each segment with sum of
    the length of all segements.

    Also creates a model for all three IAV strains together.
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "..")
from utils import RESULTSPATH, SEGMENTS, DATASET_STRAIN_DICT
from utils import get_seq_len, load_dataset


def format_dataset_for_plotting(df: pd.DataFrame,
                                dataset_name: str,
                                )-> Tuple[list, list, list]:
    '''
        Formats the dataset to have it ready for plotting and doing the linear
        regression.
        :param df: data frame including the data to prepare
        :param dataset_name: indicates which data set is loaded and will be
                             formatted

        :return: tupel with three entries:
                    x values to plot
                    y values to plot
                    error values (if available)
    '''
    x = list()
    y = list()
    err = list()
    all_sum = df["NGS_read_count"].sum()
    for i, s in enumerate(SEGMENTS):
        df_s = df.loc[df["Segment"] == s]
        y.append(df_s["NGS_read_count"].sum())
        x.append(get_seq_len(DATASET_STRAIN_DICT[dataset_name], s))
        if df_s.size == 0:
            err.append(0)
        else:
            err.append(np.std(df_s["NGS_read_count"]) / all_sum)

    return np.array(x), np.array(y) / np.array(y).sum(), err


def fit_models_and_plot_data(x: list,
                             y: list,
                             err: list,
                             k: str
                             )-> None:
    '''
        Creates the linear and exponential model for the given data and plots
        the results.
        :param x: data for x axis (segment length)
        :param y: data for y axis (DI occurrence) as relative values 
        :param err: data for the error bar (only for schwartz dataset)
        :param k: name of the strain/data set

        :return: None
    '''
    def label_scatter(x, y, k):
        # This is for adjusting the labels by hand
        if k == "Alnaji2021":
            y[2] = y[2] - 0.01 # PA
            y[3] = y[3] - 0.01 # HA
            x[4] = x[4] - 130 # NP
            y[6] = y[6] - 0.01 # M
        elif k == "Berry2021_B":
            x[7] = x[7] - 150 # NS
            y[7] = y[7] - 0.01 # NS
        for i, s in enumerate(SEGMENTS):
            ax.annotate(s, (x[i], y[i]))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    
    ax.scatter(x, y, label="observed", marker="o", color="green", s=40)
    label_scatter(x.copy(), y.copy(), k)
    y_expected = x / x.sum()
    ax.scatter(x, y_expected, label="expected", marker="o", color="grey", s=40)

    model = LinearRegression().fit(x.reshape((-1, 1)), y)
    y_pred = model.predict(x.reshape((-1, 1)))

    inter = model.intercept_
    coef = model.coef_[0]
    inter_p = -inter/coef

    # plotting the results
    score = model.score(x[:-2].reshape((-1, 1)), y[:-2])
    ax.plot(np.insert(x, 0, inter_p), np.insert(y_pred, 0, 0), color="green")
    ax.plot(np.insert(x, 0, 0), np.insert(y_expected, 0, 0), color="grey")

    # set labels and title
    ax.legend(loc="upper left", fontsize=14)
    ax.set_title(k, fontsize=20)
    ax.set_xlim(left=0, right=2600)
    ax.set_ylim(bottom=0, top=0.4)
    ax.set_xlabel("sequence length (nts.)", fontsize=14)
    ax.set_ylabel("relative DelVG occurrence", fontsize=14)

    # save final figure
    fname = f"regression_analysis_{k}.png"
    save_path = os.path.join(RESULTSPATH, "additional_analyses", fname)
    plt.savefig(save_path)
    plt.close()


def perform_regression_analysis(data: dict)-> None:
    '''
        Loops over all datasets. Creates a linear and an exponential model for
        all and plots these. Also creates one for all three Influenza strains
        together.
        :param data: dictionary with name as key and data frame as value

        :return: None
    '''
    for k, v in data.items():
        x, y, err = format_dataset_for_plotting(v, k)
        fit_models_and_plot_data(x, y, err, k)


if __name__ == "__main__":
    RESULTSPATH = os.path.dirname(RESULTSPATH)
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    d = dict()
    d["Alnaji2021"] = load_dataset("Alnaji2021")
    d["Berry2021_B"] = load_dataset("Berry2021_B")

    perform_regression_analysis(d)

