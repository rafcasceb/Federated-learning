import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from client_app import _calculate_average_test_metrics, test, train
from model import NeuralNetwork
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from task import (ClientContext, create_logger, load_client_context,
                  plot_accuracy_and_loss, plot_loaded_data, preprocess_data)
from torch.utils.data import DataLoader, TensorDataset



CONFIGURATION_FILE = "config.yaml"
CLIENT_ID = 0
IS_TEST_MODE = False



def load_data_for_centralized(excel_file: str, temp_csv: str, context: ClientContext):
    folder = "data"
    os.makedirs(folder, exist_ok=True)
    path_excel = os.path.join(folder, excel_file)
    path_csv = os.path.join(folder, temp_csv)

    context.logger.info("Reading Excel from %s", path_excel)
    df = pd.read_excel(path_excel)
    df.to_csv(path_csv, sep=";", index=False)
    df = pd.read_csv(path_csv, sep=";")
    context.logger.info("Loaded dataset. Shape: %s", df.shape)

    df = preprocess_data(df)
    context.logger.info("Preprocessed data. Shape: %s", df.shape)
    plot_loaded_data(df, CLIENT_ID)

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    x = StandardScaler().fit_transform(x)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return x_tensor, y_tensor


def centralized_train_eval():
    logger = create_logger("centralized.log")
    logger.info("Starting centralized training...")

    context = load_client_context(CLIENT_ID, logger, CONFIGURATION_FILE, IS_TEST_MODE)

    if context.random_state.is_test_mode:
        np.random.seed(context.random_state.random_seed)
        torch.manual_seed(context.random_state.random_seed)
        logger.info("Test mode. Seeds set.")

    hp = context.hyperparams
    x, y = load_data_for_centralized("PI-CAI_3__part0.xlsx", "centralized_temp.csv", context)

    model = NeuralNetwork(
        input_size=hp.input_size,
        hidden_sizes=hp.hidden_sizes,
        output_size=hp.output_size,
        dropout=hp.dropout
    )

    kfold = KFold(n_splits=hp.num_cross_val_folds, shuffle=True, random_state=context.random_state.random_seed)
    val_metrics_all = []
    val_losses_all = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(x)):
        logger.info(f"--- Fold {fold+1}/{hp.num_cross_val_folds} ---")
        model_fold = NeuralNetwork(
            input_size=hp.input_size,
            hidden_sizes=hp.hidden_sizes,
            output_size=hp.output_size,
            dropout=hp.dropout
        )

        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False)

        train(model_fold, train_loader, context)
        loss, metrics = test(model_fold, val_loader, context)

        val_losses_all.append(loss)
        val_metrics_all.append(metrics.get("Accuracy"))

    avg_loss = np.mean(val_losses_all)
    avg_acc = np.mean(val_metrics_all)
    logger.info("=== Cross-validation done ===")
    logger.info(f"Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}")

    plot_accuracy_and_loss(
        context.metrics_tracker.train_accuracies,
        context.metrics_tracker.train_losses,
        context.metrics_tracker.test_accuracies,
        context.metrics_tracker.test_losses,
        CLIENT_ID,
        hp
    )

    logger.info("Centralized training finished.")


if __name__ == "__main__":
    centralized_train_eval()
