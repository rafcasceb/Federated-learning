import os
from typing import List, Tuple

import torch
from flwr.common import Metrics
from flwr.server import ServerConfig, start_server
from model import NeuralNetwork
from strategy import FedProxSaveModel
from task import HyperParameters, create_logger, load_hyperparameters



METRICS_NAMES = ["Accuracy", "Precision", "Recall", "F1 score", "Balanced accuracy", "MCC"]
CONFIGURATION_FILE = "config.yaml"



# -------------------------
# 1. Obtain metrics
# -------------------------

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
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
    
    return aggregated_metrics




# -------------------------
# 2. Configure server
# -------------------------

def __on_fit_config_fn(server_round: int):
    '''Log the round number'''
    logger.info(f"[ROUND {server_round}]")
    return {}


def __initialize_model(hyperparams: HyperParameters):
    # Initialize the same model architecture
    model = NeuralNetwork(
        input_size = hyperparams.input_size,
        hidden_sizes = hyperparams.hidden_sizes,
        output_size = hyperparams.output_size,
        dropout = hyperparams.dropout)

    # Load the weights from the last training round
    checkpoint_path = "model_round_20.pt"  #! TODO: don't call 20 manually
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Loaded global model from checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
        
    return model


def configure_server(hyperparams: HyperParameters) -> Tuple[ServerConfig, FedProxSaveModel]:
    config = ServerConfig(
        num_rounds=hyperparams.num_rounds,
        round_timeout=600
    )
    
    model = __initialize_model(hyperparams)

    strategy = FedProxSaveModel(
        model=model,
        fraction_fit=1.0,  # fraction of clients that will be sampled per round
        fraction_evaluate=1.0,  # fraction of clients sampled for evaluation
        min_fit_clients=2,  # minimum of clients in a training round
        min_evaluate_clients=2,  # minimum of clients for evaluation
        min_available_clients=2,  # minimum of clients to stablish connection (modify for testing)
        proximal_mu=0.01,  # regularization strength
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=__on_fit_config_fn,
    )

    return config, strategy




# -------------------------
# 3. Main Execution (legacy mode)
# -------------------------

if __name__ == "__main__":
    # Function start_server is deprecated but it is the only current way to use a custom server_ip
    
    logger = create_logger("server.log")
    logger.info("Starting FL server...")
    
    server_ip = "192.168.18.12"
    server_port = "8081"
    server_address = f"{server_ip}:{server_port}"
    
    hyperparams = load_hyperparameters(CONFIGURATION_FILE)
    
    config, strategy = configure_server(hyperparams)
    logger.info("Server configuration complete. Listening on %s", server_address)

    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )
    logger.info("Closing FL server...")
