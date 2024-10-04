from collections import OrderedDict
import random
import sys
from typing import Dict
import os

import torch
import argparse
from general_utils import read_json
import logging
from federated_client import FederatedClient
from river.datasets.synth import RandomRBFDrift

import flwr as fl
import numpy as np

server_address = (
    "127.0.0.1:8082"  # os.getenv('SERVER_ADDRESS') -- for experiments on devices
)

parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int, required=True, help="ID of the client")
args = parser.parse_args()
client_id = args.client_id

data_config = read_json("./config/data_configuration.json")
slo_config = read_json("./config/slo_configuration.json")
run_config = read_json("./config/run_configuration.json")
os.makedirs(f"./logs/{run_config['pattern']}", exist_ok=True)
os.makedirs(f"./models/", exist_ok=True)

log_file = open(f"./logs/{run_config['pattern']}/{client_id}_output.log", "a")

sys.stdout = log_file
sys.stderr = log_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(log_file)],
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device {DEVICE}")


seed = int(run_config["seed"])
os.environ["PYTHONHASHSEED"] = str(seed)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def initialize_client():
    datastream_generator = RandomRBFDrift(
        seed_model=seed,
        seed_sample=seed + client_id,
        n_classes=data_config["num_classes"],
        n_features=data_config["num_features"],
        n_centroids=data_config["num_centroids"],
        change_speed=data_config["start_drift_speed"],
        n_drift_centroids=data_config["num_drifting_centroids"],
    )

    client = FederatedClient(
        datastream_generator, data_config, client_id, slo_config, run_config
    )
    return client


class FlowerClient(fl.client.NumPyClient):
    def __init__(self) -> None:
        super().__init__()
        self.client = initialize_client()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.client.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.client.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.client.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        logging.info(f"Starting fit: {config}")
        self.set_parameters(parameters)
        params = self.client.train(
            config["local_epochs"],
            config["batch_size"],
            config["lr"],
            config["is_last"],
            config["num_round"],
            config["is_continuous"],
        )
        logging.info(f"Finished fit: {params}")
        return self.get_parameters(config={}), -1, params

    def evaluate(self, parameters, config):
        logging.info(f"Starting eval: {config}")
        self.set_parameters(parameters)
        loss, performance, client_id = self.client.evaluate()
        logging.info(f"Finished eval: {performance}")
        return (
            float(loss),
            -1,
            {"performance": float(performance), "client_id": client_id},
        )


try:
    fl.client.start_client(
        server_address=server_address, client=FlowerClient().to_client()
    )
finally:
    log_file.close()
