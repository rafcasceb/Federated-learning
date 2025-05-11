import os
import torch
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedProx


class FedProxSaveModel(FedProx):
    
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = model


    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_weights is not None:
            # Save the weights in the model
            ndarrays = parameters_to_ndarrays(aggregated_weights)
            state_dict = {k: torch.tensor(v) for k, v in zip(self.global_model.state_dict().keys(), ndarrays)}
            self.global_model.load_state_dict(state_dict)
            
            # Save to file
            folder_name = "aggregated_models"
            os.makedirs(folder_name, exist_ok=True)
            file_path = os.path.join(folder_name, f"model_round_{rnd}.pt")
            torch.save(self.global_model.state_dict(), file_path)
            # logger.info(f"Saved aggregated global model after round {rnd}.")
            
        return aggregated_weights, aggregated_metrics
