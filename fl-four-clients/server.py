import json
import os
from glob import glob
from typing import List, Tuple

import torch
from flwr.common import Metrics
from flwr.server import ServerConfig, start_server
from model import NeuralNetwork
from strategy import FedProxSaveModel
from task import HyperParameters, ServerContext, create_logger, load_hyperparameters, load_server_context



METRICS_NAMES = ["Accuracy", "Precision", "Recall", "F1 score", "Balanced accuracy", "MCC"]
CONFIGURATION_FILE = "config.yaml"
METRICS_FOLDER = "logs"
METRICS_FILE = "final_aggr_metrics.json"




# -------------------------
# 1. Obtain metrics
# -------------------------

def x(context: ServerContext):
    logger = context.logger
    
    def _weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        aggregated_metrics = dict()
        total_num_cases = sum(num_cases for num_cases, _ in metrics)
        
        for metric in METRICS_NAMES:
            weighted_sum = sum(num_cases * m[metric] for num_cases, m in metrics)
            aggregated_metrics[metric] = weighted_sum / total_num_cases

        logger.info(
            f"Round metrics -- " + ", ".join(
                [f"{metric}: {aggregated_metrics[metric]:.2f}" for metric in METRICS_NAMES]
            )
        )
        
        context.round_metrics = aggregated_metrics.copy()
        
        return aggregated_metrics
        
    return _weighted_average


def _save_round_metrics_json(context: ServerContext):    
    os.makedirs(METRICS_FOLDER, exist_ok=True)
    metrics_path = os.path.join(METRICS_FOLDER, METRICS_FILE)

    with open(metrics_path, "w") as file:
        json.dump(context.round_metrics, file, indent=2)
        
    context.logger.info(f"Saved final metrics to {metrics_path}")





# -------------------------
# 2. Configure server)
# -------------------------

def y(context: ServerContext):
    logger = context.logger
    
    def _on_fit_config_fn(server_round: int):
        '''Log the round number'''
        logger.info(f"[ROUND {server_round}]")
        return {}

    return _on_fit_config_fn


def _initialize_model(context: ServerContext):
    hp = context.hyperparams
    logger = context.logger
    
    # Initialize the same model architecture
    model = NeuralNetwork(
        input_size = hp.input_size,
        hidden_sizes = hp.hidden_sizes,
        output_size = hp.output_size,
        dropout = hp.dropout)
    
    # Get a list of checkpoint files that match pattern
    folder_name = "aggregated_models"
    file_pattern = "server_model_weights_r*.pt"
    file_path = os.path.join(folder_name, file_pattern)
    checkpoint_files = glob(file_path)
    checkpoint_files.sort(
        key=lambda x: int(x.split("_")[-1].replace("r", "").replace(".pt", "")),
        reverse=True
    )
    
    if checkpoint_files:
        # Load the weights from the latest previous training round
        latest_checkpoint_path = checkpoint_files[0]
        model.load_state_dict(torch.load(latest_checkpoint_path))
        logger.info(f"Loaded global model from the latest checkpoint: {latest_checkpoint_path}")
        
        # Delete previous checkpoints
        for checkpoint in checkpoint_files:
            os.remove(checkpoint)
            logger.info(f"Deleted old checkpoint: {checkpoint}")
    else:
        logger.warning(f"No checkpoint found. Starting fresh.")
        
    return model


def _configure_server(context: ServerContext) -> Tuple[ServerConfig, FedProxSaveModel]:
    hp = context.hyperparams
    
    config = ServerConfig(
        num_rounds=hp.num_rounds,
        round_timeout=600
    )
    
    model = _initialize_model(context)

    strategy = FedProxSaveModel(
        model = model,
        logger = context.logger,
        fraction_fit = hp.fraction_fit,                     # fraction of clients that will be sampled per round
        fraction_evaluate = hp.fraction_evaluate,           # fraction of clients sampled for evaluation
        min_fit_clients = hp.min_fit_clients,               # minimum of clients in a training round
        min_evaluate_clients = hp.min_evaluate_clients,     # minimum of clients for evaluation
        min_available_clients = hp.min_available_clients,   # minimum of clients to stablish connection (modify for testing)
        proximal_mu = hp.proximal_mu,                       # regularization strength
        evaluate_metrics_aggregation_fn = x(context),
        on_fit_config_fn = y(context),
    )

    return config, strategy




# -------------------------
# 3. Main Execution (legacy mode)
# -------------------------

def main():
    logger = create_logger("server.log")
    context = load_server_context(logger, CONFIGURATION_FILE)
    
    logger.info("Starting FL server...")
    
    server_ip = "192.168.18.12"
    server_port = "8081"
    server_address = f"{server_ip}:{server_port}"

    config, strategy = _configure_server(context)
    logger.info("Server configuration complete. Listening on %s", server_address)

    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )

    _save_round_metrics_json(context)

    logger.info("Closing FL server...")


if __name__ == "__main__":
    main()
