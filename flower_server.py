import random
import time
from typing import Dict, List, Tuple

from flwr.server import ServerApp, ServerConfig, ClientManager
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from general_utils import read_json
from model_synthetic import MLP
from flwr.server.client_proxy import ClientProxy
import torch
import logging
import numpy as np
import pandas as pd
from flwr.server.client_manager import SimpleClientManager
import os


class WaitingClientManager(SimpleClientManager):
    def __init__(self, min_clients: int) -> None:
        super().__init__()
        self.min_clients = min_clients
        self.selected_clients = []

    def wait_for_min_clients(self):
        while len(self.clients) < self.min_clients:
            logging.info(
                f"Waiting for at least {self.min_clients} clients to connect. Currently connected: {len(self.clients)}"
            )
            time.sleep(10)

    def sample(
        self, num_clients: int, min_num_clients: int = None
    ) -> List[ClientProxy]:
        selected_clients = super().sample(num_clients, min_num_clients)
        self.selected_clients = selected_clients
        return selected_clients


class CustomStrategy(FedAvg):
    def __init__(self, initial_parameters: Parameters):
        super().__init__(
            min_fit_clients=run_config["num_clients"],
            min_evaluate_clients=run_config["num_clients"],
        )
        self.initial_parameters = initial_parameters
        self.pattern = run_config["pattern"]
        self.client_logs = pd.DataFrame(
            [],
            columns=[
                "ROUND",
                "CLIENT",
                "TRAIN_LOSS",
                "VAL_LOSS",
                "VAL_PERFORMANCE",
                "GLOBAL_VAL_LOSS",
                "GLOBAL_VAL_PERFORMANCE",
                "USED_TRAIN_SAMPLES",
                "PERFORMANCE_SLO",
                "TIME_SLO",
            ],
        )
        self.selected_clients = None
        self.is_continuous = False

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_manager.wait_for_min_clients()
        logging.info(f"Configuring fit for round {server_round}")
        config = {
            "batch_size": run_config["batch_size"],
            "lr": run_config["lr"],
            "local_epochs": 3,
            "is_last": server_round == run_config["num_epochs"],
            "num_round": server_round,
            "is_continuous": self.is_continuous,
        }
        fit_ins = FitIns(parameters, config)
        if server_round != run_config["num_epochs"]:
            clients = client_manager.sample(
                num_clients=run_config["num_clients"], min_num_clients=1
            )
        else:
            logging.info("Finishing training...")
            clients = [client for key, client in client_manager.clients.items()]
            self.client_logs.to_csv(
                f"./logs/{self.pattern}/centralized_logs.csv", sep="|", index=False
            )
        if not clients:
            logging.warning("No clients selected, cancel")
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Exception],
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        logging.info(f"Aggregating fit results for round {server_round}")
        if len(failures) > 0:
            for ex in failures:
                logging.error(ex)
        weights_results = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        num_clients = len(weights_results)
        if num_clients == 0:
            return self.initial_parameters, {}

        aggregated_weights = [
            torch.zeros_like(torch.tensor(weight)) for weight in weights_results[0]
        ]

        for weights in weights_results:
            for i, weight in enumerate(weights):
                aggregated_weights[i] += torch.tensor(weight)
        for i, weight in enumerate(aggregated_weights):
            aggregated_weights[i] = (weight / num_clients).numpy()

        new_parameters = ndarrays_to_parameters(aggregated_weights)

        if server_round == run_config["num_epochs"]:
            logging.info("Last aggregation step.")
            return (new_parameters, {})

        metrics_aggregated = {
            "num_clients": num_clients,
            "example_metric": sum(
                result.metrics["used_samples"] for _, result in results
            ),
            "mean_performance": np.mean(
                [result.metrics["performance"] for _, result in results]
            ),
        }

        if not self.is_continuous and (
            metrics_aggregated["mean_performance"]
            >= slo_config["global_slo"]["PERFORMANCE"]
        ):
            self.is_continuous = True

        for _, result in results:
            client_num = result.metrics["client_num"]
            logging.info(f"Saving result from training {client_num}")
            self.client_logs.loc[len(self.client_logs)] = [
                server_round,
                client_num,
                result.metrics["train_loss"],
                result.metrics["val_loss"],
                result.metrics["performance"],
                -1,
                -1,  # -1 is for global val loss/accuracy that is set during eval
                result.metrics["used_samples"],
                result.metrics["performance_slo_fulfillment"],
                result.metrics["time_slo_fulfillment"],
            ]

        return (new_parameters, metrics_aggregated)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        config = {
            "batch_size": run_config["batch_size"],
            "is_last": server_round == run_config["num_epochs"],
        }
        evaluate_ins = EvaluateIns(parameters, config)

        return [(client, evaluate_ins) for client in client_manager.selected_clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Exception],
    ) -> float:
        if len(failures) > 0:
            for ex in failures:
                logging.error(ex)

        if server_round == run_config["num_epochs"]:
            return (-1, {})

        losses = []
        performance = []
        for _, result in results:
            client_num = result.metrics["client_id"]
            logging.info(f"Saving result from eval {client_num}")
            losses.append(result.loss)
            performance.append(result.metrics["performance"])
            self.client_logs.loc[
                (self.client_logs["ROUND"] == server_round)
                & (self.client_logs["CLIENT"] == result.metrics["client_id"]),
                ["GLOBAL_VAL_LOSS", "GLOBAL_VAL_PERFORMANCE"],
            ] = [result.loss, result.metrics["performance"]]

        return (float(np.mean(losses)), {})


def get_initial_parameters(data_config):
    model = MLP(data_config["num_features"], data_config["num_classes"])
    initial_weights = [val.cpu().numpy() for val in model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(initial_weights)
    return initial_parameters


data_config = read_json("./config/data_configuration.json")
run_config = read_json("./config/run_configuration.json")
slo_config = read_json("./config/slo_configuration.json")

# for experiments on devices uncomment log files
# os.makedirs(f"./logs/{run_config['pattern']}", exist_ok=True)
# log_file = open(f"./logs/{run_config['pattern']}/server_output.log", "a")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"  # ,
    # handlers=[
    #    logging.StreamHandler(log_file)
    # ]
)

# sys.stdout = log_file
# sys.stderr = log_file

seed = int(run_config["seed"])
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

initial_parameters = get_initial_parameters(data_config)
strategy = CustomStrategy(initial_parameters=initial_parameters)

config = ServerConfig(num_rounds=run_config["num_epochs"])

app = ServerApp(
    config=config,
    strategy=strategy,
)
wait_client_manager = WaitingClientManager(min_clients=run_config["num_clients"])

try:
    if __name__ == "__main__":
        from flwr.server import start_server

        start_server(
            server_address="0.0.0.0:8082",
            config=config,
            strategy=strategy,
            client_manager=wait_client_manager,
        )
finally:
    print("end")
    # log_file.close()
