import datetime
import subprocess
import os
import concurrent.futures
from general_utils import read_json, write_json
import numpy as np
import random
from variable_categories import batch_size_mapping, lr_mapping


def run_training():
    run_config = read_json(run_config_path)
    seed = int(run_config["seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    pattern = run_config["pattern"]
    client_ids = range(run_config["num_clients"])
    scripts = [
        {
            "script": f"python flower_server.py",
            "log_file": f"./run_logs/{pattern}/output.log",
        }
    ]

    for id in client_ids:
        scripts.append(
            {
                "script": f"python flower_client.py --client_id {id}",
                "log_file": f"./run_logs/{pattern}/{id}_output.log",
            }
        )

    def run_script(script_info):
        log_file_path = script_info["log_file"]
        script_command = script_info["script"]

        with open(log_file_path, "w") as log_file:
            process = subprocess.Popen(
                script_command, shell=True, stdout=log_file, stderr=log_file
            )
            process.wait()

    m_workers = None
    if run_config["num_clients"] > 10:
        m_workers = run_config["num_clients"] + 3

    with concurrent.futures.ThreadPoolExecutor(max_workers=m_workers) as executor:
        futures = [executor.submit(run_script, script_info) for script_info in scripts]
        concurrent.futures.wait(futures)

    print("All scripts have been executed.")


seeds = [0, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
modes = ["aif", "random", "base"]

test_approaches = True
experiment = "_test_2_quantity"
run_config_path = "./config/run_configuration.json"
timestamp = datetime.datetime.now()
timestamp = timestamp.strftime("%Y_%m_%d_%H_%M")

if test_approaches:
    for seed in seeds:
        for mode in modes:
            print(
                f"{datetime.datetime.now()}: Running training for mode {mode} seed {seed}..."
            )
            pattern = f"{timestamp}_{mode}_{seed}{experiment}"
            run_config = read_json(run_config_path)
            run_config["seed"] = seed
            run_config["mode"] = mode
            run_config["pattern"] = pattern
            os.makedirs(f"./run_logs/{pattern}", exist_ok=True)
            write_json(run_config, run_config_path)
            run_training()
else:
    for bs in batch_size_mapping.keys():
        for lr in lr_mapping.keys():
            print(f"{datetime.datetime.now()}: Running training for bs {bs} lr {lr}...")
            pattern = f"{timestamp}_{bs}_{lr}{experiment}"
            run_config = read_json(run_config_path)
            run_config["seed"] = 0
            run_config["mode"] = "base"
            run_config["pattern"] = pattern
            run_config["batch_size"] = int(bs)
            run_config["lr"] = float(lr)
            os.makedirs(f"./run_logs/{pattern}", exist_ok=True)
            write_json(run_config, run_config_path)
            run_training()
    print("Remember to set proper config in the run_configuration.json")
