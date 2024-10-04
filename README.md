# Adaptive Active Inference Agents for Heterogenous and Lifelong Federated Learning

Repository for implementation and experiments. This repository contains AIF agent implementation that optimizes training time and validation accuracy SLOs for lifelong federated learning tasks based on synthetic data streams. The streams can be modified to introduce concept or quantity shifts.

This repository also contains docker files that were used to run the proposed solution on edge devices.

## How to:

1. Set the preferred SLO configuration by modifying slo_configuration.json in config/ folder. SLOs targets are controlled by slo_description field.
2. Set the data configuration in data_configuration.json; start_drift_speed field controls the presence and the speed of a concept drift; num_drifting_centroids specify how many centroids are moving to produce the drift.
3. Specify the default experiment configuration and FL parameters in run_configuration.json possible values for modes are: base (defined batch size and learning rate will be used for all FL rounds), 'aif' (active inference) and 'random'.
4. Run run_clients.py if you want to run all 3 modes with 10 different random seeds. 
5. If you want to test possible configurations and find the optimal one -- use run_clients.py script with test_approaches variable set to FALSE. This will run the pipeline with mode 'base' under all possible configurations once with the random seed 0.

To specify quantity drift, you need to modify the number of samples drawn from the generator during FL training. It can be done in federated_client.py

## Logs:

Each mode saves:
- centralized_logs.csv: logs for each client and each round accuracy, loss and slos fulfillment that are reported to the orchestrator by clients
- {id}_output.log: all recorded logs of the client {id} training
- observations_client_{id}.csv: list of observations generated at each FL round

Active inference agent additionally saves:
- all_configurations_{id}.csv: contains pragmatic value, ig, EFE and probability of each SLO being fulfilled for each configuration for each FL round
- chosen_configurations_client_{id}.csv: contains all chosen configurations
- configurations_client_{id}.csv: configurations table from AIF agent at the end of the training

## Docker:

There are two docker files:

1. server.dockerfile - Raspberry Pi 4 compatible docker container that uses python 3.8. 
2. client.docker - build upon Nvidia-provided base image with PyTorch and CUDA

Warning: do not modify Python version to be lower than 3.8, it will cause compatibility problems with the Flower framework version

For smoother configuration of the build images, consider mapping configs/ folder from outside the running container.

To use GPU, add --runtime nvidia to the docker run command

Specify CLIENT_ID and SERVER_ADDRESS (uncomment env var in flower_client.py) when running training on separate devices