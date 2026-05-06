from ...utils import *
from ...types import *
import random

def pose_to_vec(pose):
    return [pose.position.x_val, pose.position.y_val, pose.position.z_val,
            pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val]

def pose_to_dict(pose):
    return {
        'position': {
            'x_val': pose.position.x_val,
            'y_val': pose.position.y_val,
            'z_val': pose.position.z_val
        },
        'orientation': {
            'x_val': pose.orientation.x_val,
            'y_val': pose.orientation.y_val,
            'z_val': pose.orientation.z_val,
            'w_val': pose.orientation.w_val
        }
    }

def pose_from_dict(data):
    position = Vector3r(data['position']['x_val'], data['position']['y_val'], data['position']['z_val'])
    orientation = Quaternionr(data['orientation']['x_val'], data['orientation']['y_val'], data['orientation']['z_val'], data['orientation']['w_val'])
    return Pose(position, orientation)

def pose_from_vec(data):
    position = Vector3r(data[0], data[1], data[2])
    orientation = Quaternionr(data[3], data[4], data[5], data[6])
    return Pose(position, orientation)

def mutate_pose(pose, mutation_rate=0.3):
    flag = random.random()
    if flag <= mutation_rate:
        pose.position.x_val += random.uniform(-1, 1)
        pose.position.y_val += random.uniform(-1, 1)
        pose.position.z_val += random.uniform(-1, 1)
        # pose.orientation.x_val += random.uniform(-1, 1)
        # pose.orientation.y_val += random.uniform(-1, 1)
        # pose.orientation.z_val += random.uniform(-1, 1)
        # pose.orientation.w_val += random.uniform(-1, 1)

        pose.position.x_val = max(min(pose.position.x_val, 100), -100)
        pose.position.y_val = max(min(pose.position.y_val, 100), -100)
        pose.position.z_val = max(min(pose.position.z_val, 100), -100)
        # pose.orientation.x_val = max(min(pose.orientation.x_val, 1), -1)
        # pose.orientation.y_val = max(min(pose.orientation.y_val, 1), -1)
        # pose.orientation.z_val = max(min(pose.orientation.z_val, 1), -1)
        # pose.orientation.w_val = max(min(pose.orientation.w_val, 1), -1)
    
    return pose