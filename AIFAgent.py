import numpy as np
from bayesian_network import (
    get_mbs_as_bn,
    do_structural_learning,
    initialize_causal_model,
    simulate_update,
)
import pandas as pd
import copy
from slo_utils import get_batch_size_category, get_lr_category
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFWriter
import itertools
from pymdp.maths import spm_MDP_G


class AIFAgentDevice:
    def __init__(self, client_num, slo_config: dict, mode: str, random_state: int):
        self.client_num = client_num
        self.causal_model = initialize_causal_model(slo_config["variables"])
        self.initial_model = copy.deepcopy(self.causal_model)
        self.slo_list = slo_config["slo_list"]
        self.targets_slo = slo_config["targets"]
        self.random_state = random_state
        self.target_slos_thresholds = slo_config["slo_description"]
        self.history_configurations = pd.DataFrame(
            [],
            columns=[
                "BATCH_SIZE",
                "LR",
                "PERFORMANCE",
                "TIME",
                "PROB",
                "IG",
                "EFE",
                "OBS_IG",
            ],
        )
        self.history_available_configs = pd.DataFrame(
            [],
            columns=[
                "BATCH_SIZE",
                "LR",
                "PERFORMANCE",
                "TIME",
                "PROB",
                "IG",
                "EFE",
                "ROUND",
            ],
        )
        self.round_counter = 0
        self.mode = mode
        self.expected_A = None
        self.config_vars = slo_config["configuration_variables"]
        self.utility = [0, 0.001, 0.001, 0.998]
        self.preference = [np.exp(self.utility[i]) for i in range(4)]
        self.preference = self.preference / sum(self.preference)
        self.preference = np.log(self.preference)

    def process_surprise(self, history_observations):
        predicted_ig = self.history_configurations.iloc[[-1]]["IG"].iloc[0]
        num_samples = len(history_observations)
        kl_div = self.get_surprise_for_data(
            self.causal_model, history_observations.iloc[[-1]], num_samples
        )
        self.history_configurations.loc[
            self.history_configurations.index[-1], "OBS_IG"
        ] = kl_div

        if kl_div > predicted_ig:
            print(
                f"Perform structure learning as planned IG = {predicted_ig} but observed IG = {kl_div}"
            )
            self.causal_model = do_structural_learning(
                self.initial_model,
                self.slo_list,
                history_observations,
                self.config_vars,
            )  # give initial_model to have more room to add new edges
        else:
            self.causal_model.fit_update(
                history_observations.iloc[[-1]], num_samples + 1
            )

    def select_configuration(self, evidence):
        if self.mode == "aif":
            configurations = self.get_configs_for_model(evidence)
            next_config = configurations[
                configurations["EFE"] == configurations["EFE"].min()
            ].sample(
                random_state=self.random_state + self.round_counter + self.client_num
            )
            configurations["ROUND"] = self.round_counter
            print(f"Best inferred configuration:\n {next_config}")
        else:
            batch_size_set = set(
                self.causal_model.get_cpds("BATCH_SIZE").__getattribute__(
                    "state_names"
                )["BATCH_SIZE"]
            )
            lr_set = set(
                self.causal_model.get_cpds("LR").__getattribute__("state_names")["LR"]
            )
            permutations = list(itertools.product(batch_size_set, lr_set))
            batch_sizes, lr_set = zip(*permutations)
            ig = [0] * len(batch_sizes)
            prob = [0] * len(batch_sizes)
            efe = [0] * len(batch_sizes)
            performance = [0] * len(batch_sizes)
            time = [0] * len(batch_sizes)
            round = [self.round_counter] * len(batch_sizes)
            configurations = pd.DataFrame(
                list(zip(batch_sizes, lr_set, performance, time, prob, ig, efe, round)),
                columns=self.history_available_configs.columns,
            )
            next_config = configurations.sample(
                random_state=self.random_state + self.round_counter + self.client_num
            ).drop(columns=["ROUND"])
            print(f"Randomly chosen configuration:\n {next_config}")

        next_config["OBS_IG"] = -1
        self.history_available_configs = pd.concat(
            [self.history_available_configs, configurations]
        )
        self.history_configurations = pd.concat(
            [self.history_configurations, next_config], ignore_index=True
        )
        return next_config

    def find_next_configuration(self, evidence):
        config = self.select_configuration(evidence)
        batch_size = get_batch_size_category(config["BATCH_SIZE"].iloc[0], False)
        lr = get_lr_category(config["LR"].iloc[0], False)
        print(f"Current training configuration: batch size - {batch_size}, lr - {lr}")

        self.round_counter += 1
        return batch_size, lr

    def get_value_from_array(self, arr, indices):
        current = arr
        for index in indices:
            if index is not None:
                current = current[index]
        return current

    def predict_missing_values(self, inference, observation):
        predicted_observations = observation.copy()
        for index, row in observation.iterrows():
            evidence = {var: val for var, val in row.items() if not pd.isnull(val)}
            missing_vars = [var for var, val in row.items() if pd.isnull(val)]
            if missing_vars:
                prediction = inference.map_query(
                    variables=missing_vars, evidence=evidence, show_progress=False
                )
                for var, val in prediction.items():
                    predicted_observations.at[index, var] = val
        return predicted_observations

    def get_configs_for_model(self, evidence_data):
        bn_mb = get_mbs_as_bn(self.causal_model, self.slo_list)
        var_el = VariableElimination(bn_mb)

        batch_size_list = self.causal_model.get_cpds("BATCH_SIZE").__getattribute__(
            "state_names"
        )["BATCH_SIZE"]
        LR_list = self.causal_model.get_cpds("LR").__getattribute__("state_names")["LR"]
        config_line = []
        for bs in batch_size_list:
            for lr in LR_list:
                evidence = {}

                if "BATCH_SIZE" in bn_mb:
                    evidence.update({"BATCH_SIZE": bs})

                if "LR" in bn_mb:
                    evidence.update({"LR": lr})

                # individual SLO probability of fulfillment
                performance = np.round(
                    var_el.query(variables=["PERFORMANCE"], evidence=evidence).values[
                        self.targets_slo["PERFORMANCE"]
                    ],
                    decimals=4,
                )
                time = np.round(
                    var_el.query(variables=["TIME"], evidence=evidence).values[
                        self.targets_slo["TIME"]
                    ],
                    decimals=4,
                )

                # pragmatic value
                pv = sum(
                    var_el.query(
                        variables=["PERFORMANCE", "TIME"], evidence=evidence
                    ).values.flatten()
                    * self.preference
                )

                # envisioning the future observation under config
                observation = pd.DataFrame(
                    [[np.nan, np.nan, bs, lr, np.nan, np.nan]],
                    columns=[
                        "CPU_USAGE",
                        "MEMORY_USAGE",
                        "BATCH_SIZE",
                        "LR",
                        "PERFORMANCE",
                        "TIME",
                    ],
                )
                inference = VariableElimination(self.causal_model)
                observation = self.predict_missing_values(inference, observation)
                fake_model = simulate_update(
                    self.causal_model, observation, len(self.history_configurations)
                )
                bn_mb_posterior = get_mbs_as_bn(
                    fake_model, self.slo_list + self.config_vars
                )
                var_el_posterior = VariableElimination(bn_mb_posterior)

                # calculating information gain
                evidence_ig = {}
                for var in bn_mb_posterior.nodes():
                    if var not in self.config_vars:
                        evidence_ig.update({var: observation[var].iloc[0]})

                q_t = var_el_posterior.query(
                    variables=self.config_vars, evidence=evidence_ig, joint=True
                ).values.flatten()
                mat_A = self.compute_A(var_el_posterior, bn_mb_posterior)
                ig = spm_MDP_G(mat_A, np.asarray(q_t).flatten())
                if ig == 0:
                    ig += 1e-7

                # merge results into EFE
                efe = -ig - pv
                config_line.append([bs, lr, performance, time, pv, ig, efe])

        config_df = pd.DataFrame(
            config_line,
            columns=["BATCH_SIZE", "LR", "PERFORMANCE", "TIME", "PROB", "IG", "EFE"],
        )
        print(config_df)
        return config_df

    def compute_A(self, var_el, bn_mb):
        batch_size_list = self.causal_model.get_cpds("BATCH_SIZE").__getattribute__(
            "state_names"
        )["BATCH_SIZE"]
        LR_list = self.causal_model.get_cpds("LR").__getattribute__("state_names")["LR"]
        mat_A = [[] for i in range(4)]
        for bs in batch_size_list:
            for lr in LR_list:
                evidence = {}
                if "BATCH_SIZE" in bn_mb:
                    evidence.update({"BATCH_SIZE": bs})

                if "LR" in bn_mb:
                    evidence.update({"LR": lr})
                prob_outcomes = var_el.query(
                    variables=self.slo_list, evidence=evidence, joint=True
                ).values
                prob_outcomes = prob_outcomes.flatten()
                for i in range(len(prob_outcomes)):
                    mat_A[i].append(prob_outcomes[i])
        for i in range(len(prob_outcomes)):
            mat_A[i] = np.asarray(mat_A[i])
        mat_A = np.asarray(mat_A)
        return mat_A

    def get_surprise_for_data(self, model, data, sample):
        fake_model = simulate_update(model, data, sample - 1)
        suprise = 0
        try:
            bn_mb_posterior = get_mbs_as_bn(
                fake_model, self.slo_list + self.config_vars
            )
            var_el_posterior = VariableElimination(bn_mb_posterior)
            evidence = {}
            x_list = []
            for var in bn_mb_posterior.nodes():
                if var in self.config_vars:
                    x_list.append(var)
                else:
                    evidence.update({var: data[var].iloc[0]})

            q_t = var_el_posterior.query(
                variables=x_list, evidence=evidence, joint=True
            ).values.flatten()

            expected_A = self.compute_A(var_el_posterior, bn_mb_posterior)
            suprise = spm_MDP_G(expected_A, q_t)
        except Exception as e:
            print(str(e))

        return suprise

    def dump_logs(self, timestamp, evidence):
        self.history_configurations.to_csv(
            f"./logs/{timestamp}/chosen_configurations_client_{self.client_num}.csv",
            sep="|",
            index=False,
        )
        print(self.causal_model.edges())
        XMLBIFWriter(self.causal_model).write_xmlbif(
            f"./models/{timestamp}_BN_client_{self.client_num}.xml"
        )
        configurations_ranked = self.get_configs_for_model(evidence)
        configurations_ranked.to_csv(
            f"./logs/{timestamp}/configurations_client_{self.client_num}.csv",
            sep="|",
            index=False,
        )
        self.history_available_configs.to_csv(
            f"./logs/{timestamp}/all_configurations_{self.client_num}.csv",
            sep="|",
            index=False,
        )
