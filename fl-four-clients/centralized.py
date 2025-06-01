from typing import Dict, Tuple

import numpy as np
import torch
from client_app import configure_environment, load_data, test, train
from model import NeuralNetwork
from sklearn.model_selection import KFold, train_test_split
from task import (ClientContext, create_logger, load_client_context,
                  plot_accuracy_and_loss)
from torch.utils.data import DataLoader, TensorDataset



CONFIGURATION_FILE = "config.yaml"
EXCEL_NAME = "PI-CAI_3.xlsx" 
TEMP_CSV_NAME = "temp_database.csv"
CLIENT_ID = 0
IS_TEST_MODE = False




# -------------------------
# 1. Training and Evaluation
# -------------------------

def cross_validation_test(x: torch.Tensor, y: torch.Tensor, context: ClientContext) -> None:
    context.logger.info("Starting cross-validation training and testing...")
    hp = context.hyperparams
    rs = context.random_state

    kfold = KFold(n_splits=hp.num_cross_val_folds, shuffle=True, random_state=rs.random_seed)
    loss_all_folds = []
    acc_all_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(x)):
        context.logger.info(f"Training cross-validation fold {fold_idx+1}/{hp.num_cross_val_folds}...")
        
        model_fold = NeuralNetwork(
            input_size=hp.input_size,
            hidden_sizes=hp.hidden_sizes,
            output_size=hp.output_size,
            dropout=hp.dropout
        )

        x_train_fold, y_train_fold = x[train_idx], y[train_idx]
        x_test_fold, y_test_fold = x[test_idx], y[test_idx]
        train_dataset = TensorDataset(x_train_fold, y_train_fold)
        test_dataset = TensorDataset(x_test_fold, y_test_fold)
        train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=rs.shuffle_loaders)
        test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=rs.shuffle_loaders)

        train(model_fold, train_loader, context)
        test_loss, test_metrics = test(model_fold, test_loader, context)
        loss_all_folds.append(test_loss)
        acc_all_folds.append(test_metrics.get("Accuracy"))
        #! TODO: podría coger todas las métricas

    avg_loss_all_folds = np.mean(loss_all_folds)
    avg_acc_all_folds = np.mean(acc_all_folds)
    context.logger.info("Cross-validation finished.")
    context.logger.info(f"Cross-validation -- Avg Loss: {avg_loss_all_folds:.4f}, Avg Accuracy: {avg_acc_all_folds:.4f}")


def centralized_train_eval(context: ClientContext) -> Tuple[float, Dict[str,float]]:
    hp = context.hyperparams
    rs = context.random_state
    logger = context.logger
    x, y = load_data(EXCEL_NAME, TEMP_CSV_NAME, context)

    model = NeuralNetwork(
        input_size=hp.input_size,
        hidden_sizes=hp.hidden_sizes,
        output_size=hp.output_size,
        dropout=hp.dropout
    )

    context.logger.info("Starting local model training...")
    do_cross_validation = (hp.do_cross_val) and (hp.num_cross_val_folds > 1)
    
    if do_cross_validation:
        whole_dataset = TensorDataset(x, y)
        whole_dataloader = DataLoader(whole_dataset, batch_size=hp.batch_size, shuffle=rs.shuffle_loaders)
        train(model, whole_dataloader, context)
        logger.info("Local training complete.")
        loss, metrics = cross_validation_test(x, y, context)
        
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size = hp.test_size,
            random_state = rs.random_seed
        )
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False)

        train(model, train_loader, context)
        logger.info("Local training complete.")
        loss, metrics = test(model, test_loader, context)
    
    return loss, metrics



# -------------------------
# 2. Main Execution
# -------------------------

def start_centralized_training(client_id: int, is_test_mode: bool=False):    
    logger_name = "centralized.log"
    logger = create_logger(logger_name)
    logger.info("Starting centralized training...")

    try:
        context = load_client_context(client_id, logger, CONFIGURATION_FILE, is_test_mode)
    except Exception as e:
        logger.error(f"Failed to load context: {str(e)}")
        logger.info("Centralized training finished")
        return

    configure_environment(context)
    centralized_train_eval(context)

    mt = context.metrics_tracker
    plot_accuracy_and_loss(
        mt.train_accuracies, mt.train_losses, mt.test_accuracies, mt.test_losses,
        context.client_id, context.hyperparams
    )
    logger.info("Centralized training finished")


if __name__ == "__main__":
    start_centralized_training(CLIENT_ID, IS_TEST_MODE)
