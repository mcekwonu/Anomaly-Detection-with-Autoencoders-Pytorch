import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew


def load_train_val_test(path, remove_features="SPD", transform=True):
    """Load training, validation and test sequences

    Parameters:
        path: Source directory containing train and validation dataset
        remove_features: Feature to be dropped
        transform: Transform dataset using MinMaxScaler

    Returns:
        train, validation and test sequences, length of sequence, number of features and scaler
    """
    if isinstance(remove_features, str):
        remove_features = [remove_features]
    elif isinstance(remove_features, tuple):
        remove_features = [*remove_features]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    path = [os.path.join(path, f) for f in os.listdir(path)]
    train_path = [f for f in path if "train" in f][0]
    val_path = [f for f in path if "val" in f][0]
    test_path = [f for f in path if "test" in f][0]

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    if len(remove_features) == 1:
        train_df = train_df.drop(remove_features, axis="columns").astype(np.float32)
        val_df = val_df.drop(remove_features, axis="columns").astype(np.float32)
        test_df = test_df.drop(remove_features, axis="columns").astype(np.float32)
    else:
        train_df = train_df.drop(train_df.loc[:, remove_features].columns, axis=1).astype(np.float32)
        val_df = val_df.drop(val_df.loc[:, remove_features].columns, axis=1).astype(np.float32)
        test_df = test_df.drop(test_df.loc[:, remove_features].columns, axis=1).astype(np.float32)

    if transform:
        train_df = scaler.fit_transform(train_df.values)
        val_df = scaler.fit_transform(val_df.values)
        test_df = scaler.fit_transform(test_df.values)
    else:
        train_df = train_df.values
        val_df = val_df.values
        test_df = test_df.values

    train_seq, seq_len, n_features = create_dataset(train_df)
    val_seq, _, _ = create_dataset(val_df)
    test_seq, _, _ = create_dataset(test_df)

    return train_seq, val_seq, test_seq, seq_len, n_features, scaler


def create_dataset(sequences):
    """Create dataset to be passed to LSTM autoencoder"""
    dataset = [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape

    return dataset, seq_len, n_features


if __name__ == "__main__":
    train_sequence, val_sequence, test_sequence, target_len, num_features, scaler = \
        load_train_val_test(path="../Data/20220823", remove_features=("SPD", "ANG"))

    print(f"Number of features: {num_features}")
    print(f"train input size: {train_sequence[0].shape}")
