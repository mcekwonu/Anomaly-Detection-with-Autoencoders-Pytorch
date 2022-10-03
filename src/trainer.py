import os
import random
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm.auto import trange
from collections import defaultdict
from sklearn import metrics

from generate_dataset import load_train_val_test
from lstm_ae import LSTMAE

RANDOM_SEED = 11
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def mean_absolute_error(true, pred):
    if isinstance(true, tensor) and isinstance(pred, tensor):
        true = true.numpy()
        pred = pred.numpy()

    return np.mean(np.abs(true - pred), axis=1)


class Trainer(nn.Module):
    """Training class for LSTM autoencoder and make predictions"""

    def __init__(self, log_dir):
        super().__init__()
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.L1Loss(reduction="mean").to(self.device)
        self.history = defaultdict(list)
        self.model_name = None

    def fit(self, model, train_dataset, val_dataset, num_epochs, learning_rate=1e-03):
        """Train model

        Parameters:
            model (nn.Module): LSTM autoencoder
            train_dataset (Tensor): train dataset
            val_dataset (Tensor): validation dataset
            num_epochs (int):      number of epochs
            learning_rate (float): learning rate

        Returns:
            losses:  array of loss function for each epoch
        """
        self.model_name = f"{model.name}_{model.num_layers}"
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.to(device)
        criterion = self.criterion

        print(f"Training {model.name} AutoEncoder for {num_epochs} epochs...\n")
        for epoch in range(num_epochs):
            model.train()
            train_losses = []

            for i, seq_input in enumerate(train_dataset):
                optimizer.zero_grad()
                seq_input = seq_input.to(device)
                seq_pred = model(seq_input)
                loss = criterion(seq_pred, seq_input)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            model.eval()
            with torch.no_grad():
                for seq_input in val_dataset:
                    seq_input = seq_input.to(device)
                    seq_pred = model(seq_input)
                    loss = criterion(seq_pred, seq_input)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            self.history["train loss"].append(train_loss)
            self.history["val loss"].append(val_loss)

            print(f"Epoch: {epoch + 1}  train loss: {train_loss:.5f}  Val loss: {val_loss:.5f}")

        torch.save(model.state_dict(), f"{self.log_dir}/{self.model_name}.pth")

        np.savez(
            f"{self.log_dir}/{self.model_name}_losses", train_loss=self.history["train loss"],
            val_loss=self.history["val loss"]
        )

        return self.history

    def plot_loss_history(self, history, filename):
        save_dir = "../Figures"
        os.makedirs(save_dir, exist_ok=True)

        ax = plt.figure().gca()
        ax.plot(history["train loss"], color="green", label="train")
        ax.plot(history["val loss"], color="orange", label="val")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{self.model_name}_{filename}.png", dpi=500)
        plt.close()

    def predict(self, model, model_path, dataset):
        n_features = model.n_features
        criterion = self.criterion
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        model.to(self.device)

        predictions, losses = [], []
        with torch.no_grad():
            model.eval()
            for seq_input in dataset:
                seq_input = seq_input.to(self.device)
                seq_pred = model(seq_input)
                loss = criterion(seq_pred, seq_input)
                seq_pred = seq_pred.cpu().numpy().flatten()
                predictions.append(seq_pred)
                losses.append(loss.item())

        losses = np.array(losses).reshape(-1, 1)

        if n_features == 1:
            predictions = np.array(predictions).reshape(-1, 1)
        elif n_features == 2:
            predictions = np.array(predictions).reshape(-1, 2)

        return predictions, losses

    def plot_prediction(self, dataset, model, model_path, scaler, show_plot=True):
        n_features = model.n_features
        predictions, losses = self.predict(model=model, model_path=model_path, dataset=dataset)

        # Load training loss and fix threshold
        loss_data = np.load(f"../training_logs/{self.model_name}_losses.npz", allow_pickle=True)
        reconstruction_error = np.asarray(loss_data["train_loss"])
        threshold = np.mean(reconstruction_error) + np.std(reconstruction_error)

        # convert dataset to ndarray and unscale the values
        if n_features == 1:
            true_values = np.asarray([s.numpy() for s in dataset]).reshape((-1, 1))
        elif n_features == 2:
            true_values = np.asarray([s.numpy() for s in dataset]).reshape((-1, 2))

        true_values = scaler.inverse_transform(true_values)
        predictions = scaler.inverse_transform(predictions)

        # compute metrics
        mae = metrics.mean_absolute_error(true_values, predictions)
        R2 = metrics.r2_score(true_values, predictions)
        correct = sum(l <= threshold for l in losses)
        anomaly = [l >= threshold for l in losses]

        # plot reconstruction error, and  test error
        sns.histplot(reconstruction_error, bins=50, kde=True, stat="probability", legend=False)
        plt.xlabel("Reconstruction Error")
        plt.savefig(f"../Figures/{self.model_name}_recon_error.png", bbox_inches="tight", dpi=500)
        plt.close()

        sns.histplot(losses, bins=50, kde=True, stat="probability", legend=False)
        plt.xlabel("Reconstruction Error")
        plt.savefig(f"../Figures/{self.model_name}_test_error.png", bbox_inches="tight", dpi=500)
        plt.close()

        if show_plot:
            plt.show()

        with open(f"../training_logs/{self.model_name}_result.txt", "w") as f:
            f.write(f"Threshold: {threshold:.3f}\n")
            f.write(f"Accuracy: {correct[0] / len(dataset) * 100: .2f}\n")
            f.write(f"Mean Absolute Error: {mae:.4f}\n")
            f.write(f"R2 score: {R2:.4f}")

        if n_features == 1:
            data = {"TRQ": list(true_values[:, 0]), "TRQ pred": list(predictions[:, 0]),
                    "MAE": list(losses.flatten()), "Threshold": [threshold] * predictions.shape[0]
                    }
        elif n_features == 2:
            data = {"ANG": list(true_values[:, 0]), "TRQ": list(true_values[:, 1]),
                    "ANG pred": list(predictions[:, 0]), "TRQ pred": list(predictions[:, 1]),
                    "MAE": list(losses.flatten()), "Threshold": [threshold] * predictions.shape[0]
                    }

        df = pd.DataFrame.from_dict(data, orient="columns")
        df["Anomaly"] = df["MAE"] >= threshold
        df.to_csv(f"../training_logs/{self.model_name}.csv", index=False)