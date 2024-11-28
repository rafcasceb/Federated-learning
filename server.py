from typing import List, Tuple

from flwr.common import Metrics
from flwr.server import ServerApp, ServerConfig, start_server
from flwr.server.strategy import FedAvg



def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    custom_accuracies = [num_examples * m["custom_accuracy"] for num_examples, m in metrics]
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    
    num_examples_list = [num_examples for num_examples, _ in metrics]
    total_num_examples = sum(num_examples_list)

    metrics = {
        "custom_accuracy": sum(custom_accuracies) / total_num_examples,
        "accuracy": sum(accuracies) / total_num_examples,
        "precision": sum(precisions) / total_num_examples,
        "recall": sum(recalls) / total_num_examples,
        "f1_score": sum(f1_scores) / total_num_examples,
    }
    
    return metrics



# Define strategy with updated parameters
strategy = FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=1, # 1 client minimum     
    min_available_clients=2  # cambiar cuando solo quiera probar con un único cliente, cambiar 2
)

# Define config
config = ServerConfig(
    num_rounds=30,
    round_timeout=600
)

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)



# Legacy mode
if __name__ == "__main__":
    
    server_ip = input("SERVER IP: ") 
    server_port = input("SERVER PORT: ") 
    server_address = f"{server_ip}:{server_port}"

    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )
