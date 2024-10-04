import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import time
import psutil
from AIFAgent import AIFAgentDevice
import os
import copy
from model_synthetic import MLP
from slo_utils import encode_sensory_input
from river.datasets.synth import RandomRBFDrift


class FederatedClient:
    def __init__(
        self,
        data_generator: RandomRBFDrift,
        data_settings: dict,
        client_num: int,
        slo_config: dict,
        run_config: dict,
    ):
        self.data_generator = data_generator
        self.data_settings = data_settings
        self.train_dataset = self.sample_dataset(
            self.data_settings["start_num_samples"]
        )
        self.val_dataset = self.sample_dataset(self.data_settings["start_num_samples"])
        self.last_val_loader = None

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.client_num = client_num
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # torch.device("cpu")
        self.targets = slo_config["slo_description"]
        self.mode = run_config["mode"]
        self.pattern = run_config["pattern"]
        self.random_state = run_config["seed"]
        self.AIFagent = AIFAgentDevice(
            client_num, slo_config, self.mode, run_config["seed"]
        )
        self.model = MLP(data_settings["num_features"], data_settings["num_classes"])
        self.generator = torch.Generator()
        self.generator.manual_seed(self.random_state)
        self.history_observations = pd.DataFrame(
            [],
            columns=[
                "PERFORMANCE",
                "TIME",
                "BATCH_SIZE",
                "LR",
                "CPU_USAGE",
                "MEMORY_USAGE",
            ],
        )

    def train(self, local_epochs, batch_size, lr, is_last, num_round, is_continuous):

        if is_last:
            self.end_work()
            return {}

        samples_to_take = self.data_settings["start_num_samples"]

        if num_round > 100:
            samples_to_take = int(samples_to_take * 3)
        elif num_round > 50:
            samples_to_take = int(samples_to_take * 2)

        print(
            f"({num_round}) Training started for client {self.client_num}, taking {samples_to_take} samples:"
        )
        if self.mode != "base":
            batch_size, lr = self.AIFagent.find_next_configuration(())
        self.train_dataset = copy.deepcopy(self.val_dataset)
        self.val_dataset = self.sample_dataset(samples_to_take)

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=self.generator,
        )
        self.last_val_loader = val_loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=self.generator,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-2)
        cpu_usage = []
        mem_usage = []
        start_time = time.time()
        eval_performance = []

        self.model.to(self.device)

        for epoch in range(local_epochs):
            self.model.train()
            train_loss = 0
            i = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                loss.backward()
                mem_usage.append(psutil.virtual_memory().percent)
                cpu_usage.append(psutil.cpu_percent() / os.cpu_count())
                optimizer.step()
                train_loss += loss.cpu().item()
                i += 1

            train_loss = train_loss / i
            test_loss, performance, _ = self.evaluate(val_loader)
            eval_performance.append(performance)

            print(
                f"Epoch [{epoch+1}/{local_epochs}], Train loss: {train_loss:.4f}, Validation loss: {test_loss:.4f}"
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed training time: {elapsed_time}")

        observation = encode_sensory_input(
            eval_performance[-1],
            cpu_usage,
            mem_usage,
            batch_size,
            lr,
            elapsed_time,
            self.targets,
        )
        performance_slo, time_slo = (observation[0], observation[1])

        if is_continuous:

            self.history_observations.loc[len(self.history_observations)] = observation

            if self.mode != "base":
                self.AIFagent.process_surprise(self.history_observations)

        print(observation)

        return {
            "client_num": self.client_num,
            "used_samples": local_epochs * samples_to_take,
            "train_loss": train_loss,
            "val_loss": test_loss,
            "performance": performance,
            "performance_slo_fulfillment": int(performance_slo),
            "time_slo_fulfillment": int(time_slo),
        }

    def evaluate(self, val_loader=None):

        record_performance = True if val_loader == None else False
        if val_loader == None:
            val_loader = self.last_val_loader

        actual_y = np.array([], dtype=float)
        predicted_y = np.array([], dtype=float)

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            test_loss = 0
            i = 0
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                loss = self.criterion(outputs.to("cpu"), y_batch)
                test_loss += loss.item()
                actual_y = np.concatenate(
                    (actual_y, y_batch.detach().cpu().numpy().flatten()), axis=0
                )
                predicted_y = np.concatenate(
                    (predicted_y, predictions.detach().cpu().numpy().flatten()), axis=0
                )
                i += 1

        test_loss = test_loss / i
        accuracy = accuracy_score(actual_y, predicted_y)
        print(f"For client {self.client_num }: Accuracy {accuracy:.4f}")
        print(f"Validation loss: {test_loss:.4f}")
        if record_performance:
            self.previous_performance = accuracy

        return test_loss, accuracy, self.client_num

    def sample_dataset(self, num_samples):
        samples = self.data_generator.take(num_samples)
        x_list = []
        y_list = []
        for sample in samples:
            x, y = sample
            x_list.append(list(x.values()))
            y_list.append(y)
        return TensorDataset(torch.tensor(x_list), torch.tensor(y_list))

    def end_work(self):
        self.history_observations.to_csv(
            f"./logs/{self.pattern}/observations_client_{self.client_num}.csv",
            sep="|",
            index=False,
        )
        if self.mode != "base":
            self.AIFagent.dump_logs(self.pattern, ())
