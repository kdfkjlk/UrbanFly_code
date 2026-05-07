import json
import os


def read_data(data_path):
    with open(data_path, 'r') as infile:
        data = json.load(infile)
    return data


import os
import json


def load_one_map_data(dataset_dir, map_name, mode="test"):
    """
    Load one UrbanFly map split.

    Expected structure:
        UrbanFly/DATA/{mode}/{map_name}/{mode}.json

    Example:
        UrbanFly/DATA/test/ModernCityEnvironment/test.json
    """
    file_path = os.path.join(dataset_dir, mode, map_name, f"{mode}.json")

    if not os.path.exists(file_path):
        print(f"[WARNING] Missing file: {file_path}")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data
    


def load_all_maps(dataset_dir, map_names, mode):
    all_data = []

    for map_name in map_names:
        path = os.path.join(dataset_dir, mode, map_name, mode + '.json')
        if not os.path.exists(path):
            print(f"[警告] 跳过缺失文件：{path}")
            continue
        data = read_data(path)
        
        all_data.extend(data)
    return all_data