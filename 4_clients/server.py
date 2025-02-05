from typing import List, Tuple

from flwr.common import Metrics
from flwr.server import ServerApp, ServerConfig, start_server
from flwr.server.strategy import FedAvg
from task import create_logger



METRICS_NAMES = ["accuracy", "precision", "recall", "f1_score"]



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
            [f"{metric.capitalize()}: {aggregated_metrics[metric]:.2f}" for metric in METRICS_NAMES]
        )
    )
    
    return aggregated_metrics




# -------------------------
# 3. Configure server
# -------------------------

def on_fit_config_fn(server_round: int):
    '''Log the round number'''
    logger.info(f"[ROUND {server_round}]")
    return {}


def configure_server() -> Tuple[FedAvg, ServerConfig, ServerApp]:
    config = ServerConfig(
        num_rounds = 20,
        round_timeout=600
    )

    strategy = FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config_fn,
        min_fit_clients=2,       # minimum of clients in a round
        min_evaluate_clients=2,  # minimum of clients for evaluation
        min_available_clients=2  # minimum of clients to stablish connection (modify for testing)
    )   
    
    return config, strategy




# -------------------------
# 4. Main Execution (legacy mode)
# -------------------------

if __name__ == "__main__":
    # Function start_server is deprecated but it is the only current way to use a custom server_ip
    
    logger = create_logger("server.log")
    
    logger.info("Starting FL server...")
    
    #server_ip = input("SERVER IP: ") 
    #server_port = input("SERVER PORT: ") 
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