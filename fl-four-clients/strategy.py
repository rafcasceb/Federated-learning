import os

import torch
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedProx



MODELS_FOLDER = "aggregated_models"
MODELS_FILE_TEMPLATE = "server_model_weights_r{rnd}.pt"



class FedProxSaveModel(FedProx):

    def __init__(self, model, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = model
        self.logger = logger


    def aggregate_fit(self, rnd, results, failures):
        if failures:
            self.logger.warning(f"Failures in round {rnd}: {failures}")

        aggregated_weights, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        if aggregated_weights is not None:
            # Save the weights in the model
            ndarrays = parameters_to_ndarrays(aggregated_weights)
            state_dict = {k: torch.tensor(v) for k, v in zip(self.global_model.state_dict().keys(), ndarrays)}
            self.global_model.load_state_dict(state_dict)

            # Save to file
            os.makedirs(MODELS_FOLDER, exist_ok=True)
            model_file_name = MODELS_FILE_TEMPLATE.format(rnd=rnd)
            file_path = os.path.join(MODELS_FOLDER, model_file_name)
            torch.save(self.global_model.state_dict(), file_path)
            self.logger.info(f"Saved aggregated global model after round {rnd} at {file_path}.")

        return aggregated_weights, aggregated_metrics
