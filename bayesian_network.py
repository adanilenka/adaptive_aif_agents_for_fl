from typing import List
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.estimators import BayesianEstimator
import copy
import numpy as np
from collections import OrderedDict


def initialize_causal_model(var_dict: dict):

    nodes = list(var_dict.keys())
    print(nodes)
    causal_model = BayesianNetwork()
    causal_model.add_nodes_from(nodes)

    for key, value in var_dict.items():
        values = [[1 / value] for i in range(value)]
        cpd = TabularCPD(variable=key, variable_card=value, values=values)
        causal_model.add_cpds(cpd)
    assert causal_model.check_model(), "The model has inconsistencies."
    return causal_model


def simulate_update(model, obs, sample):
    fake_model = copy.deepcopy(model)
    fake_model.fit_update(obs, sample + 1)
    return fake_model


def get_mbs_as_bn(model: BayesianNetwork, center: List[str]):
    mb_list = []
    for node in center:
        mb_list.extend(model.get_markov_blanket(node))
    mb = copy.deepcopy(model)

    mb_list.extend(center)
    for n in model.nodes:
        if n not in mb_list:
            mb.remove_node(n)

    return mb


def do_structural_learning(model, slo_list, data, config_vars):
    var_states = {}
    for node in list(model.nodes()):
        var_states.update({node: list(range(model.get_cardinality(node)))})
    print(model.edges())
    hc = HillClimbSearch(data, state_names=var_states)
    best_model = hc.estimate()
    print(best_model.edges())
    edges = best_model.edges()
    new_model = inspect_edges(edges, model, slo_list, data, config_vars)
    model_score = BicScore(data).score(model)
    new_model_score = BicScore(data).score(new_model)
    print(f"Old model BIC score: {model_score}, New model BIC score: {new_model_score}")
    return new_model


def calculate_ess_exponential_decay(step):
    initial_ess = 20
    decay_rate = 0.05
    ess = initial_ess * np.exp(-decay_rate * step)
    if ess == 0:
        return 1
    else:
        return ess


def sort_edge(x, slo_list, config_vars):
    if x[0] in slo_list or x[1] in slo_list:
        if x[0] in config_vars or x[1] in config_vars:
            return (0, x)
        else:
            return (1, x)
    if x[0] in config_vars or x[1] in config_vars:
        return (3, x)
    return (999, x)


def rearrange_edges(edges, slo_list):
    return list(
        OrderedDict.fromkeys(
            [(edge[1], edge[0]) if edge[0] in slo_list else edge for edge in edges]
        )
    )


def inspect_edges(edges, model, slo_list, hist_data, config_vars):
    new_model = BayesianNetwork()
    new_model.add_nodes_from(model.nodes())
    var_states = {}
    for node in list(model.nodes()):
        var_states.update({node: list(range(model.get_cardinality(node)))})

    edges = sorted(edges, key=lambda x: sort_edge(x, slo_list, config_vars))
    edges = rearrange_edges(edges, slo_list)

    print(f"Considering edges: {edges}")
    for start_edge, end_edge in edges:
        try:
            new_model.add_edge(start_edge, end_edge)
            print(f"Added edge: {(start_edge, end_edge)}")
        except Exception as e:
            print(f"Couldn't add edge: {(start_edge, end_edge)}")
            print(e)

    ess = calculate_ess_exponential_decay(len(hist_data))
    print(f"Current ESS: {ess}, dataset len {len(hist_data)}")
    new_model.fit(
        hist_data,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=ess,
        state_names=var_states,
    )
    return new_model
