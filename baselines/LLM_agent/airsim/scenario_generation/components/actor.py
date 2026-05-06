
from .base_component import BaseComponent 
from ...types import *
from .utils import pose_to_vec, pose_to_dict, pose_from_dict, pose_from_vec, mutate_pose
import random
from .config import ACTOR_TYPE, MODE_TYPE


class Actor(BaseComponent):
    def __init__(self, type:str, start_pose:Pose, end_pose:Pose, speed, name='actor_comp'):
        super().__init__(name)
        self.type = ACTOR_TYPE[type]
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.speed = speed
        # self.mode = MODE_TYPE[mode]

    def get_type_name(self):
        # get the type name by checking the ACTOR_TYPE dictionary, from the type value to get the key
        return list(ACTOR_TYPE.keys())[list(ACTOR_TYPE.values()).index(self.type)]
    
    def get_mode_name(self):
        return list(MODE_TYPE.keys())[list(MODE_TYPE.values()).index(self.mode)]

    def to_vec(self):
        return [self.type] + pose_to_vec(self.start_pose) + pose_to_vec(self.end_pose) + [self.speed]

    def to_dict(self):
        actor_dict = {'type': self.get_type_name(self.type), 'mode': self.get_mode_name(self.mode)}
        actor_dict['start_pose'] = pose_to_dict(self.start_pose)
        actor_dict['end_pose'] = pose_to_dict(self.end_pose.to_dict())
        actor_dict['speed'] = self.speed
        return actor_dict


    @classmethod
    def from_dict(cls, data):
        start_pose = pose_from_dict(data['start_pose'])
        end_pose = pose_from_dict(data['end_pose'])
        return cls(data['type'], start_pose, end_pose, data['speed'], data['mode'])

    @classmethod
    def from_vec(cls, data):
        start_pose = pose_from_vec(data[1:8])
        end_pose = pose_from_vec(data[8:15])
        return cls(data[0], start_pose, end_pose, data[-1])    


    @classmethod
    def crossover(cls, actor_1, actor_2):
        actor_1_vec = actor_1.to_vec()
        actor_2_vec = actor_2.to_vec()
        half_len = len(actor_1_vec) // 2
        new_actor_vec_1 = actor_1_vec[:half_len] + actor_2_vec[half_len:]
        new_actor_vec_2 = actor_2_vec[:half_len] + actor_1_vec[half_len:]

        return cls.from_vec(new_actor_vec_1), cls.from_vec(new_actor_vec_2)
    
    def mutate(self, mutation_rate=0.3):
        self.start_pose = mutate_pose(self.start_pose, mutation_rate)  # Change x by [-1, 1] units
        self.end_pose = mutate_pose(self.end_pose, mutation_rate)  # Change x by [-1, 1] units

        flag = random.random()
        if flag < mutation_rate:
            self.type = random.choice(list(ACTOR_TYPE.keys()))
            if self.type == 'bird':
                self.start_pose.position.z_val = random.uniform(3, 8)
                self.end_pose.position.z_val = self.start_pose.z

        flag = random.random()
        if flag < mutation_rate:        
            self.speed = random.random()
            
        flag = random.random()
        if flag < mutation_rate:
            self.mode = random.choice(list(MODE_TYPE.keys()))