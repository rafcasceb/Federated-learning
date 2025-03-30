from typing import List, Tuple

from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, start_server
from flwr.server.strategy import FedAvg, FedProx, FedAdagrad
from client_app import NeuralNetwork
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

def on_fit_config_fn(server_round: int):
    '''Log the round number'''
    logger.info(f"[ROUND {server_round}]")
    return {}


def get_initial_parameters():
    """Initialize the model and convert parameters to Flower format."""
    input_size = 3  # Change this based on feature count
    hidden_sizes = [128, 128]
    output_size = 1

    model = NeuralNetwork(input_size, hidden_sizes, output_size)
    numpy_params = [val.cpu().numpy() for _, val in model.state_dict().items()]

    return ndarrays_to_parameters(numpy_params)


def configure_server() -> Tuple[ServerConfig, FedAdagrad]:
    config = ServerConfig(
        num_rounds=20,
        round_timeout=600
    )

    strategy = FedAdagrad(
        initial_parameters=get_initial_parameters(),
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config_fn,
        min_fit_clients=2,       
        min_evaluate_clients=2,  
        min_available_clients=2  
    )   

    return config, strategy




# -------------------------
# 3. Main Execution (legacy mode)
# -------------------------

if __name__ == "__main__":
    # Function start_server is deprecated but it is the only current way to use a custom server_ip
    
    logger = create_logger("server.log")
    
    #server_ip = input("SERVER IP: ") 
    #server_port = input("SERVER PORT: ") 
    server_ip = "192.168.18.12"
    server_port = "8081"
    server_address = f"{server_ip}:{server_port}"
    
    logger.info("Starting FL server...")
    
    config, strategy = configure_server()
    logger.info("Server configuration complete. Listening on %s", server_address)

    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )
    logger.info("Closing FL server...")