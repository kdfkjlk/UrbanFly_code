from .base_component import BaseComponent 
from ...types import *
from .utils import pose_to_vec, pose_to_dict, pose_from_dict, pose_from_vec
from .config import ACTOR_TYPE
import random

class Marker(BaseComponent):
    def __init__(self, id, material, pose:Pose, name='marker_comp'):
        super().__init__(name)
        self.id = id
        self.material = material
        self.pose = pose
        self.type = ACTOR_TYPE['marker']

    def to_vec(self):
        return [self.id, self.material] + pose_to_vec(self.pose)

    def to_dict(self):
        marker_dict = {'id': self.id, 'material': self.material}
        marker_dict.update(pose_to_dict(self.pose))
        return marker_dict


    @classmethod
    def from_dict(cls, data):
        pose = pose_from_dict(data)
        return cls(data['id'], data['material'], pose)
    
    @classmethod
    def from_vec(cls, data):
        pose = pose_from_vec(data[2:])
        return cls(data[0], data[1], pose)
    
    @classmethod
    def crossover(cls, marker_1, marker_2):
        marker_1_vec = marker_1.to_vec()
        marker_2_vec = marker_2.to_vec()
        half_len = len(marker_1_vec) // 2
        new_marker_vec_1 = marker_1_vec[:half_len] + marker_2_vec[half_len:]
        new_marker_vec_2 = marker_2_vec[:half_len] + marker_1_vec[half_len:]

        return cls.from_vec(new_marker_vec_1), cls.from_vec(new_marker_vec_2)
    
    def mutate(self, mutation_rate=0.3, mutate_id=False):
        # Introduce small random changes to the position and angle of the marker
        self.pose.mutate(mutation_rate)
        flag = random.random()
        if flag <= mutation_rate:
            if mutate_id:
                self.id = random.choice([2, 3, 4, 5, 6, 7])