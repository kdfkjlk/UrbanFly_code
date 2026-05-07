import json
import os
from datetime import datetime



CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))




def read_data(data_path):
    with open(data_path, 'r') as infile:
        data = json.load(infile)
    return data


def load_one_map_data(path):
    # path = os.path.join(dataset_dir, mode, map_name, mode + '.json')

    if not os.path.exists(path):
        print(f"[警告] 跳过缺失文件：{path}")
    else:
        data = read_data(path)
        return data



def load_all_maps(dataset_dir, map_names, mode):
    all_data = {}

    for map_name in map_names:
        path = os.path.join(dataset_dir, map_name, mode + '.json')
        if not os.path.exists(path):
            print(f"[警告] 跳过缺失文件：{path}")
            continue
        data = read_data(path)
        
        all_data[map_name] = data
    return all_data





class DataLoad_Manager:
    def __init__(
            self,
            dataset_dir = None,
            mode = 'train'
        ):
        if dataset_dir is None:
            self.dataset_dir = os.path.join(WORKSPACE_ROOT, "DATA", mode)
        else:
            self.dataset_dir = dataset_dir
        self.mode = mode

        self.episode_id = 0

    
    def init_map_names(self, map_names=None):

        if map_names is None:
            self.map_names = os.listdir(self.dataset_dir)
            self.map_names = [name for name in self.map_names if os.path.isdir(os.path.join(self.dataset_dir, name))]

        elif isinstance(map_names, list):
            self.map_names = map_names
        
        elif isinstance(map_names, str):
            self.map_names = [map_names]

        return self.map_names



    def load_data_all(self):
        self.init_map_names()
        self.data = load_all_maps(self.dataset_dir, self.map_names, self.mode)
        return self.data
    


    def load_data_one_map(self, map_name=None):
        map_name = map_name if map_name is not None else self.map_names
        path = os.path.join(self.dataset_dir, map_name, self.mode + '.json')
        self.data = load_one_map_data(path)
        return self.data
    

    def load_data_part_maps(self, map_names=None):
        map_names = map_names if map_names is not None else self.map_names

        self.data = {}
        for map_name in map_names:
            path = os.path.join(self.dataset_dir, map_name, self.mode + '.json')
            map_data = load_one_map_data(path)
            if map_data is not None:
                self.data[map_name] = map_data
        return self.data
    
    
    def get_episode_length(self, map_name):
        return len(self.data[map_name])
    







if __name__ == "__main__":
    dataload_manager = DataLoad_Manager(mode='test')
    print(dataload_manager.map_names)

    episode_data = dataload_manager.load_data_all()
    print(len(episode_data))
    # episode_0 = dataload_manager.get_episode_data()
    # print(episode_0)

    # rid_out = dataload_manager.check_episode_usage()
    # print(rid_out)