from typing import List, Tuple

from flwr.common import Metrics
from flwr.server import ServerApp, ServerConfig, start_server
from flwr.server.strategy import FedAvg


# -------------------------
# 1. Obtain metrics
# -------------------------

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    
    num_examples_list = [num_examples for num_examples, _ in metrics]
    total_num_examples = sum(num_examples_list)

    metrics = {
        "accuracy": sum(accuracies) / total_num_examples,
        "precision": sum(precisions) / total_num_examples,
        "recall": sum(recalls) / total_num_examples,
        "f1_score": sum(f1_scores) / total_num_examples,
    }
    
    print(sum(accuracies) / total_num_examples,
          sum(precisions) / total_num_examples,
          sum(recalls) / total_num_examples,
          sum(f1_scores) / total_num_examples)
    
    return metrics




# -------------------------
# 2. Configure server
# -------------------------
def configure_server() -> Tuple[FedAvg, ServerConfig, ServerApp]:
    config = ServerConfig(
        num_rounds = 20,
        round_timeout=600
    )

    strategy = FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        min_fit_clients=1,       # minimum of clients in a round
        min_evaluate_clients=1,  # minimum of clients for evaluation
        min_available_clients=1  # minimum of clients to stablish connection (modify for testing)
    )
    
    return config, strategy




# -------------------------
# 3. Main Execution (legacy mode)
# -------------------------

if __name__ == "__main__":
    # Function start_server is deprecated but it is the only current way to use a custom server_ip
    
    #server_ip = input("SERVER IP: ") 
    #server_port = input("SERVER PORT: ") 
    server_ip = "192.168.18.12"
    server_port = "8081"
    server_address = f"{server_ip}:{server_port}"
    
    config, strategy = configure_server()

    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )
