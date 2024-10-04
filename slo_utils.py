import numpy as np
from variable_categories import batch_size_mapping, lr_mapping


def get_time_category(sec, target):
    if sec <= target:
        return 1
    else:
        return 0


def get_performance_category(performance, target_performance):
    if np.round(performance, decimals=2) >= target_performance:
        return 1
    else:
        return 0


def map_percents(x):
    return x // 10


def get_batch_size_category(bs, is_key=True):

    if is_key:
        return batch_size_mapping[bs]
    else:
        keys = [key for key, value in batch_size_mapping.items() if value == bs]
        return int(keys[0])


def get_lr_category(bs, is_key=True):

    if is_key:
        return lr_mapping[bs]
    else:
        keys = [key for key, value in lr_mapping.items() if value == bs]
        return float(keys[0])


def encode_sensory_input(
    val_metric, cpu_usage, mem_usage, batch_size, lr, elapsed_time, slo_dict
):
    performance = get_performance_category(val_metric, slo_dict["PERFORMANCE"])
    cpu_usage = int(map_percents(np.round(np.mean(cpu_usage), decimals=0)))
    memory_usage = int(map_percents(np.round(np.mean(mem_usage), decimals=0)))
    batch_size_category = get_batch_size_category(str(batch_size))
    lr_category = get_lr_category(str(lr))
    time_category = get_time_category(elapsed_time, slo_dict["TIME"])

    result = [
        performance,
        time_category,
        batch_size_category,
        lr_category,
        cpu_usage,
        memory_usage,
    ]
    return result
