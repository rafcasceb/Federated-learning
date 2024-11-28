from typing import List, Tuple

from flwr.common import Metrics
from flwr.server import ServerApp, ServerConfig, start_server
from flwr.server.strategy import FedAvg



#netstat -na | grep 8080

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    average_accuracy = sum(accuracies) / sum(examples)
    return {"average accuracy": average_accuracy}



# Define strategy with updated parameters
strategy = FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=1, # 1 client minimum     
    min_available_clients=4  # cambiar cuando solo quiera probar con un único cliente, cambiar 2
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
