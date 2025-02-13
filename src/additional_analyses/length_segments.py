'''
    Gives two examples for how the expected values for the segment distribution
    was calcualted. The length of each segment was divided by the length of the
    full genome.
'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from typing import Tuple
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "..")
from utils import RESULTSPATH, SEGMENTS, DATASET_STRAIN_DICT
from utils import get_seq_len, load_all


def format_dataset_for_plotting(df: pd.DataFrame,
                                dataset_name: str,
                                )-> Tuple[list, list]:
    '''
        Formats the dataset to have it ready for plotting and doing the linear
        regression.
        :param df: data frame including the data to prepare
        :param dataset_name: indicates which data set is loaded and will be
                             formatted

        :return: tupel with three entries:
                    x values to plot
                    y values to plot
    '''
    x = list()
    y = list()
    for s in SEGMENTS:
        df_s = df.loc[df["Segment"] == s]
        y.append(df_s["NGS_read_count"].sum())
        x.append(get_seq_len(DATASET_STRAIN_DICT[dataset_name], s))
    return np.array(x), np.array(y) / np.array(y).sum() * 100


def perform_regression_analysis(dfs, dfnames):
    for df, dfname in zip(dfs, dfnames):
        f_obs = df["Segment"].value_counts()
        for s in SEGMENTS:
            if s not in f_obs:
                f_obs[s] = 0
        
        full_seqs = np.array([get_seq_len(DATASET_STRAIN_DICT[dfname], seg) for seg in SEGMENTS])
        f_exp = full_seqs / sum(full_seqs) * f_obs.sum()

        r, pvalue = stats.chisquare(f_obs, f_exp)
        if pvalue < 0.05:
            v_data = np.stack((f_obs.to_numpy(), np.rint(f_exp)), axis=1).astype(int)
            cramers_v = stats.contingency.association(v_data)
            cramers_v = round(cramers_v, 2)
        else:
            cramers_v = "n.a."

        print(dfname)
        print(r, pvalue, cramers_v)
        
        x, y = format_dataset_for_plotting(df, dfname)

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
        label_scatter(x.copy(), y.copy(), dfname)
        y_expected = x / x.sum() * 100
        ax.scatter(x, y_expected, label="expected", marker="o", color="grey", s=40)

        # linear regression to show expected data
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
        ax.set_title(dfname, fontsize=20)
        ax.set_xlim(left=0, right=2600)
        ax.set_ylim(bottom=0, top=40)
        ax.set_xlabel("Sequence length (nt.)", fontsize=14)
        ax.set_ylabel("Fraction of DelVGs [%]", fontsize=14)

        # save final figure
        fname = f"regression_analysis_{dfname}.png"
        save_path = os.path.join(RESULTSPATH, "additional_analyses", fname)
        plt.savefig(save_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    RESULTSPATH = os.path.dirname(RESULTSPATH)
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    dfnames = ["Alnaji2021", "Berry2021_B"]
    dfs, _ = load_all(dfnames)

    perform_regression_analysis(dfs, dfnames)

