from typing import List, Tuple

import yaml
from flwr.common import Metrics
from flwr.server import ServerConfig
from flwr.server.strategy import FedProx
from task import create_logger

METRICS_NAMES = ["Accuracy", "Precision", "Recall", "F1 score", "Balanced accuracy", "MCC"]



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

def read_from_yaml(file_name: str):
    with open(file_name, "r") as file:
        run_config = yaml.safe_load(file)
        return run_config


def configure_server() -> Tuple[ServerConfig, FedProx]:
    config = ServerConfig(
        num_rounds=20,
        round_timeout=600
    )

    strategy = FedProx(
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
    
    config, strategy = configure_server()
    logger.info("Server configuration complete. Listening on %s", server_address)

    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )
    logger.info("Closing FL server...")
