import json
import time
import os


class Sim_state():
    def __init__(self, init_data):
        self.map_name = init_data['map_name']
        self.marker_pose = init_data['marker_pose']
        self.drone_pose = init_data['drone_pose']
        self.env_time_param = init_data['time']
        self.env_weather_param = init_data['weather']

        self.seen_flag = init_data['seen_flag']
        self.hard_level = init_data['hard_level']

        self.is_collide = False
        self.flight_state ='search'
        self.is_end = False
        self.time_is_up = False

        self.start_fly = False


    def start_flying_logs(self):
        self.start_time = time.time()
        self.trajectory = [{
            'time_step': 0,
            'drone_position': self.drone_pose[0:-1],
            'drone_orientation': self.drone_pose[-1],
            'flight_stage': self.flight_state,
            'time_s': 0,
            'key_press': 0,
            'is_end': False,
            'is_collide': False
        }]


    def update_trajectory(self, key, drone_pose, drone_ori):
        info = {
            'time_step': len(self.trajectory),
            'drone_position': drone_pose,
            'drone_orientation': drone_ori,
            'flight_stage': self.flight_state,
            'time_s': round(time.time() - self.start_time, 2),
            'key_press': key,
            'is_end': self.is_end,
            'is_collide': self.is_collide
        }
        self.trajectory.append(info)


    def save_flight_data(self, save_path):
        flight_data = {
            "map_name": self.map_name,
            "marker_pose": self.marker_pose,
            "drone_pose": self.drone_pose,
            "env_time_param": self.env_time_param,
            "env_weather_param": self.env_weather_param,
            "seen_flag": self.seen_flag,
            "hard_level": self.hard_level,
            "collision": self.is_collide,
            "trajectory": self.trajectory,
            "is_end": self.is_end
        }

        # 保存数据到JSON文件
        with open(save_path, 'w') as file:
            json.dump(flight_data, file, indent=4)

        print(f"Flight data saved to {save_path}")






