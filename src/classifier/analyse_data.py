'''

'''
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from ml_utils import preprocessing

sys.path.insert(0, "..")
from utils import DATAPATH, RESULTSPATH



def run_dim_red_method(X: pd.DataFrame,
         y: pd.Series,
         method: str
         )-> None:
    '''
        Runs a dim. red. method for two dimensions and plots the results
        :param X: input features as data frame
        :param y: output vector as series
        :param name: string of all used datasets
        :param method: name of the method which should be performed

        :return: None
    '''
    if method == "pca":
        dr_obj = PCA(n_components=2)
    elif method == "tsne":
        dr_obj = TSNE(n_components=2)
    elif method == "umap":
        dr_obj = UMAP(n_components=2)

    X_embedded = dr_obj.fit_transform(X)
    plot_df = pd.DataFrame(data=X_embedded, columns=["x", "y"])

    for l in sorted([0, 1]):
        indices = y == l
        f1 = plot_df.loc[indices, "x"]
        f2 = plot_df.loc[indices, "y"]
        plt.scatter(f1, f2, alpha=0.2, label=l)

  
    plt.legend()
    plt.title(method)
    save_path = os.path.join(RESULTSPATH, "ML", f"{method}.png")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # load precalculated data from file
    path = os.path.join(DATAPATH, "ML", "features.csv")
    df = pd.read_csv(path, na_values=["", "None"], keep_default_na=False)
    df = df[df["NGS_read_count"] > 10].copy()

    X_1, X_2, y_1, y_2 = preprocessing(df)
    X = np.vstack((X_1, X_2))
    y = np.concatenate((y_1, y_2), axis=0)

    methods = ["pca", "tsne"]#, "umap"]
    for method in methods:
        print(method)
        run_dim_red_method(X, y, method)  
