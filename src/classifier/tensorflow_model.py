'''

'''
import os
import sys

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

from ml_utils import preprocessing, prepare_y

sys.path.insert(0, "..")

from utils import DATAPATH



if __name__ == "__main__":
    # load precalculated data from file
    path = os.path.join(DATAPATH, "ML", "features.csv")
    df = pd.read_csv(path, na_values=["", "None"], keep_default_na=False)

    df = df[df["NGS_read_count"] > 10].copy()

    train_df = df[df["dataset_name"].isin(["Wang2023", "Wang2020"])]
    validate_df = df[df["dataset_name"].isin(["Alnaji2021", "Pelz2021", "Kupke2020"])]

    X_train, X_test, y_train, y_test = preprocessing(df)
    
    # Define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    epochs = 10  # Adjust the number of epochs as needed
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)

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
        print(y_pred)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        print(y)
        print(y_pred)

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



