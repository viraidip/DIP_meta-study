'''

'''
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "..")
from utils import RESULTSPATH, SEGMENTS
from utils import load_all, get_sequence, count_direct_repeats_overall


def create_direct_repeats_plot(df, dfname, expected_df):
    fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True)
    final_d = dict()
    expected_final_d = dict()
    for s in SEGMENTS:
        df_s = df[df["Segment"] == s]
        expected_df_s = expected_df[expected_df["Segment"] == s]
        n_samples = len(df_s)
        if n_samples == 0:
            continue
        seq = get_sequence(df_s["Strain"].unique()[0], s)            
        counts, _ = count_direct_repeats_overall(df_s, seq)
        for k, v in counts.items():
            if k in final_d:
                final_d[k] += v
            else:
                final_d[k] = v

        expected_counts, _ = count_direct_repeats_overall(expected_df_s, seq)
        for k, v in expected_counts.items():
            if k in expected_final_d:
                expected_final_d[k] += v
            else:
                expected_final_d[k] = v

    final = np.array(list(final_d.values()))
    expected_final = np.array(list(expected_final_d.values()))
    f_obs = final/final.sum() * 100 # normalize to percentage
    f_exp = expected_final/expected_final.sum() * 100 # normalize to percentage
    x = list(final_d.keys())

    ax.bar(x=x, height=f_obs, width=-0.4, align="edge", label="observed", edgecolor="black", color="firebrick")
    ax.bar(x=x, height=f_exp, width=0.4, align="edge", label="expected", edgecolor="black", color="royalblue")

    ax.set_title(dfname)
    ax.legend(loc="upper right")
    ax.set_xlabel("Direct repeat length")
    ax.set_ylabel("Occurrence")
    ax.set_xticks(x, ["0", "1", "2", "3", "4", ">4"])

    # save final figure
    fname = f"direct_repeats_{dfname}.png"
    save_path = os.path.join(RESULTSPATH, "additional_analyses", fname)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    RESULTSPATH = os.path.dirname(RESULTSPATH)
    plt.style.use("seaborn")
    plt.rc("font", size=12)

    dfname = "Alnaji2021"
    dfs, expected_dfs = load_all([dfname], expected=True)
    create_direct_repeats_plot(dfs[0], dfname, expected_dfs[0])


