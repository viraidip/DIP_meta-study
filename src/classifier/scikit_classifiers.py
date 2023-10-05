'''
    Tests different classifers on different data sets. Also tests which
    combination of features is the best.
'''
import os
import sys
import shap
import joblib
import logging
import argparse
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, RocCurveDisplay, confusion_matrix, make_scorer

from ml_utils import preprocessing

sys.path.insert(0, "..")
from utils import DATAPATH, RESULTSPATH


def select_classifier(clf_name: str,
                      grid_search: bool=False
                      )-> Tuple[object, dict]:
    '''
        Selects a scikit-learn classifier by a given name. Is implemented in an
        extra function to use the same parameters in each usage of the
        classifiers.
        :param clf_name: name of the classifier
        :param grid_search: Bool indicating if a grid search will be performed

        :return: 1. Selected classifier as class implemented in scikit-learn
                 2. parameter grid, if grid search is True, else empty dict
    '''
    if clf_name == "logistic_regression":
        if grid_search:
            clf = LogisticRegression(max_iter=10000, solver="saga", l1_ratio=0.5)
            param_grid = {
                "penalty": ["l1", "l2", "elasticnet"], 
                "C" : [0.01, 0.1, 1.0],
            }
        else:
            clf = LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=10000)
            param_grid = dict()
    elif clf_name == "svc":
        if grid_search:
            clf = SVC(gamma=2, C=1)
            param_grid = {
                "C" : [0.01, 0.1, 1.0],
                "kernel" : ["linear", "rbf", "sigmoid"],
                "gamma" : ["scale", "auto", 2],
            }
        else:
            clf = SVC(gamma="scale", C=1.0, kernel="rbf")
            param_grid = dict()
    elif clf_name == "random_forest":
        if grid_search:
            clf = RandomForestClassifier()
            param_grid = {
                "min_samples_split": [3, 5, 10], 
                "n_estimators" : [100, 300],
                "max_depth": [3, 5, 15, 25],
                "max_features": [3, 5, 10, 20]
            }
        else:
            clf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=10, max_features=20)
            param_grid = dict()
    elif clf_name == "mlp":
        if grid_search:
            clf = MLPClassifier(max_iter=10000)
            param_grid = {
                "hidden_layer_sizes": [(50,), (100,), (250,)], 
                "alpha" : [0.001, 0.0001, 0.00001]
            }
        else:
            clf = MLPClassifier(alpha=0.0001, hidden_layer_sizes=(100,), max_iter=10000)
            param_grid = dict()
    elif clf_name == "ada_boost":
        if grid_search:
            clf = AdaBoostClassifier(n_estimators=25, learning_rate=0.1)
            param_grid = {
                "n_estimators": [25, 50, 100,], 
                "learning_rate" : [0.1, 0.5, 1.0],
            }
        else:
            clf = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
            param_grid = dict()
    elif clf_name == "naive_bayes":
        clf = GaussianNB(var_smoothing=0.0000000001)
        if grid_search:
            param_grid = {
                "var_smoothing": [0.000000001, 0.0000000001, 0.0000000001]
            }
        else:
            param_grid = dict()
    else:
        print(f"classifier {clf_name} unknown!")
        exit()
    return clf, param_grid




    # Selecting train/test and validation data sets



if __name__ == "__main__":
    perform_grid_search = True

    # load precalculated data from file
    path = os.path.join(DATAPATH, "ML", "features.csv")
    df = pd.read_csv(path, na_values=["", "None"], keep_default_na=False)


    
    #    df["DI"] = df["Segment"] + "_" + df["Start"].map(str) + "_" + df["End"].map(str)
     #   df.drop_duplicates("DI", keep=False, inplace=True, ignore_index=True)




    

    X, y, X_val, y_val = preprocessing(df)

    logging.info("Distribution of labels:")
    logging.info(y.value_counts())
    logging.info(y_val.value_counts())
    logging.info("#####\n")

    # Testing different classifiers
    clf_names = ["logistic_regression", "svc", "random_forest", "mlp", "ada_boost", "naive_bayes"]
    data_dict = dict()
    data_dict["param"] = ["accuracy"]
    for clf_name in clf_names:
        print(clf_name)
        logging.info(f"\n### {clf_name} ###")

        data_dict[clf_name] = list()
        # setting up classifier and k-fold validation
        clf, param_grid = select_classifier(clf_name, grid_search=perform_grid_search)
        skf = StratifiedKFold(n_splits=5)
        scorers = {"accuracy_score": make_scorer(accuracy_score)}

        # perform grid search for best parameters
        grid_search = GridSearchCV(clf, param_grid, scoring=scorers, cv=skf, return_train_score=True, refit="accuracy_score")
        grid_search.fit(X, y)

        print(f"training acc.:\t{grid_search.best_score_}")

        if perform_grid_search:
            print(grid_search.best_params_)
            logging.info(grid_search.best_params_)

        # fit on overall model and create confusion matrix for validation set
        predicted_val = grid_search.predict(X_val)
        acc_score = accuracy_score(predicted_val, y_val)
        confusion_m = confusion_matrix(predicted_val, y_val)

        print(f"validation acc.:{acc_score}")
        print(confusion_m)
        data_dict[clf_name].append(acc_score)

        plt.rc("font", size=14)
        fig, axs = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)

        y_shuffled = y_val.sample(frac=1, random_state=42, ignore_index=True).to_numpy()

        RocCurveDisplay.from_estimator(grid_search, X_val, y_val, name=clf_name, ax=axs)
        RocCurveDisplay.from_estimator(grid_search, X_val, y_shuffled, name="shuffled", ax=axs)
        plt.plot([0,1], [0,1])
        path = os.path.join(RESULTSPATH, "ML", f"{clf_name}_roc_curve.png")
        plt.savefig(path)
        plt.close()

    o_df = pd.DataFrame(data_dict)
    o_df["mean"] = o_df.mean(axis=1)
    path = os.path.join(RESULTSPATH, "ML", f"means.tex")
    o_df.to_latex(path, index=False, float_format="%.2f", longtable=True)
