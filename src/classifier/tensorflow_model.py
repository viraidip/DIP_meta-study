'''

'''
import os
import sys

import classifier.tensorflow_model as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

from ml_utils import preprocessing, prepare_y

sys.path.insert(0, "..")

from utils import DATAPATH



if __name__ == "__main__":
    # load precalculated data from file
    path = os.path.join(DATAPATH, "ML", "features.csv")
    df = pd.read_csv(path, na_values=["", "None"], keep_default_na=False)

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

    for n in df["dataset_name"].unique():
        print(n)
        n_df = df[df["dataset_name"] == n].copy()
        X = n_df.drop(["Unnamed: 0", "Segment", "NGS_read_count", "class", "dataset_name","Strain","NGS_log","NGS_norm","NGS_log_norm","int_dup","Duplicate","comb_dup"], axis=1)
        y = prepare_y(n_df)

        loss, accuracy = model.evaluate(X, y)
        print(f'Test Loss: {loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')


