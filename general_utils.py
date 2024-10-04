import json


def read_json(path):
    with open(path, "r") as file:
        dict_from_json = json.load(file)
    return dict_from_json


def write_json(dict_to_write, path):
    with open(path, "w") as file:
        json.dump(dict_to_write, file)
