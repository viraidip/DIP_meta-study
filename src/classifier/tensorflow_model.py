'''

'''
import os
import sys
import shap

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, RocCurveDisplay

from ml_utils import preprocessing

sys.path.insert(0, "..")

from utils import DATAPATH



if __name__ == "__main__":
    # load precalculated data from file
    path = os.path.join(DATAPATH, "ML", "features.csv")
    df = pd.read_csv(path, na_values=["", "None"], keep_default_na=False)

    df = df[df["NGS_read_count"] > 10].copy()

    train_df = df[df["dataset_name"].isin(["Wang2023", "Wang2020", "Alnaji2021", "Pelz2021", "Kupke2020", "Mendes2021", "Alnaji2019 NC", "Alnaji2019 Cal07", "Alnaji2019 Perth", "Alnaji2019 BLEE", "Lui2019", "Penn2022"])]
    validate_df = df[df["dataset_name"].isin(["Alnaji2021", "Pelz2021", "Kupke2020", "Mendes2021", "Alnaji2019 NC", "Alnaji2019 Cal07", "Alnaji2019 Perth", "Alnaji2019 BLEE", "Lui2019", "Penn2022"])]

    X_train, X_test, y_train, y_test = preprocessing(train_df)
    
    # Define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    epochs = 10 # Adjust the number of epochs as needed
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['acc.', 'loss'], loc='upper left')
    plt.show()
    plt.close()

    '''
    X = np.vstack((X_train, X_test))
    X100 = shap.utils.sample(X, 100)
    explainer = shap.Explainer(model, X100)
    shap_values = explainer(X)

    fig, axs = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
    fig = shap.plots.beeswarm(shap_values, show=False)

#    path = os.path.join(folder, f"{clf_name}_shap_beeswarm.png")
 #   plt.savefig(path)
    plt.show()
    plt.close()
    exit()
    '''


    # Evaluate the model on the testing data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    
    for n in validate_df["dataset_name"].unique():
        print(n)
        n_df = validate_df[validate_df["dataset_name"] == n].copy()
        X_1, X_2, y_1, y_2 = preprocessing(n_df)

        X = np.vstack((X_1, X_2))
        y = np.concatenate((y_1, y_2), axis=0)

        y_pred = model.predict(X)
        
        RocCurveDisplay.from_predictions(y, y_pred)
        plt.plot([0,1], [0,1])
        #path = os.path.join(folder, f"{clf_name}_roc_curve.png")
        #plt.savefig(path)
        plt.show()
        plt.close()



        hist_df = pd.DataFrame(dict({"true": y, "pred": y_pred.T[0]}))

        grouped = hist_df.groupby("true")
        fig, ax = plt.subplots()

        # Plot each group separately with a different color
        for label, group in grouped:
            ax.hist(group['pred'], bins=10, label=label, alpha=0.5)

        plt.legend()
        plt.show()
        plt.close()

        sns.set(style="whitegrid")

        # Create a boxplot using seaborn
        plt.figure(figsize=(8, 6))  # Set the figure size
        sns.boxplot(x='true', y='pred', data=hist_df, palette='Set2')
        plt.legend()
        plt.show()
        plt.close()


        y_pred = np.where(y_pred >= 0.5, 1, 0)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", cm)
