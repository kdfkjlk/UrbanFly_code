import airsim
import numpy as np
import math
import time
import json
from typing import List, Dict
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union



def read_data(data_path):
    with open(data_path, 'r') as infile:
        data = json.load(infile)
    return data


def interpret_action(action, current_position, current_rotation, step_size=1, step_size_z=1, turn_angle=90):
    '''
    param:
        action index: action index (int, 1 forward, 2 turn left, 3 turn right, 4 ascend, 5 descend, others: stop or hover)
        current_position: [x, y, z], z is negative in airsim
        current_rotation: [x, y, z, w]
        step_size: step size
        step_size_z: step size in z direction
        turn_angle: turn angle  

    return:
        new_pose: airsim.Pose

    '''
    current_position = np.array(current_position) 
    current_rotation = np.array(current_rotation)

    pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
        x_val=current_rotation[0],
        y_val=current_rotation[1],
        z_val=current_rotation[2],
        w_val=current_rotation[3]
    ))
    pitch = 0
    roll = 0

    # forward
    if action == 1:
        unit_x = 1 * math.cos(pitch) * math.cos(yaw)
        unit_y = 1 * math.cos(pitch) * math.sin(yaw)
        unit_z = 1 * math.sin(pitch) * (-1)
        unit_vector = np.array([unit_x, unit_y, unit_z])
        assert unit_z == 0

        new_position = np.array(current_position) + unit_vector * step_size
        new_rotation = current_rotation.copy()

    # turn left
    elif action == 2:
        new_position = current_position.copy()

        new_pitch = pitch
        new_roll = roll
        new_yaw = yaw - math.radians(turn_angle)
        if float(new_yaw * 180 / math.pi) < -180:
            new_yaw = math.radians(360) + new_yaw
        new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
        new_rotation = [
                new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
            ]

    # turn right
    elif action == 3:
        new_position = current_position.copy()

        new_pitch = pitch
        new_roll = roll
        new_yaw = yaw + math.radians(turn_angle)
        if float(new_yaw * 180 / math.pi) > 180:    
            new_yaw = math.radians(-360) + new_yaw
        new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
        new_rotation = [
                new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
            ]
    
    # ascend
    elif action == 4:
        unit_vector = np.array([0, 0, -1])
        new_position = np.array(current_position) + unit_vector * step_size_z
        new_rotation = current_rotation.copy()

    # descend
    elif action == 5:
        unit_vector = np.array([0, 0, 1])
        new_position = np.array(current_position) + unit_vector * step_size_z
        new_rotation = current_rotation.copy()

    # stop or hover
    else:
        new_position = current_position.copy()
        new_rotation = current_rotation.copy()
    

    new_pose = airsim.Pose(
            position_val=airsim.Vector3r(
                x_val=new_position[0],
                y_val=new_position[1],
                z_val=new_position[2]
            ),
            orientation_val=airsim.Quaternionr(
                x_val=new_rotation[0],
                y_val=new_rotation[1],
                z_val=new_rotation[2],
                w_val=new_rotation[3]
            )
        )

    return new_pose  



def update_path_length(action, current_path_length, step_size=1, step_size_z=1):
    if action == 1: # forward
        current_path_length += step_size
    
    elif action == 4 or action == 5: # ascend or descend
        current_path_length += step_size_z

    return current_path_length



def update_distance_drone2marker(current_position, marker_position):
    '''
    param:
        current_position: list, [x, y, z]
        marker_position: list, [x, y, z]

    return:
        distance: float
    '''
    
    distance = np.linalg.norm(np.array(current_position) - np.array(marker_position))
    return distance



def update_success(distance_drone2marker, success_distance=2):
    success = False
    if distance_drone2marker < success_distance:
        success = True
    return success



def get_current_pose(drone_tool):
    agent_state = drone_tool.get_drone_pose()
    x = agent_state.position.x_val   ## x left positive
    y = agent_state.position.y_val 
    z = agent_state.position.z_val
    return [x,y,z]

def get_current_rotation(drone_tool):
    agent_state = drone_tool.get_drone_pose()
    rotation = [agent_state.orientation.x_val, 
                agent_state.orientation.y_val, 
                agent_state.orientation.z_val, 
                agent_state.orientation.w_val]
    return rotation


def get_agent_pose(drone_tool):
    agent_state = drone_tool.get_drone_pose()
    x = agent_state.position.x_val   ## x left positive
    y = agent_state.position.y_val 
    return [x,y]
    
def get_agent_height(drone_tool):
    agent_state = drone_tool.get_drone_pose()
    z = -agent_state.position.z_val
    return z
    
def get_forward_img_rgb(drone_tool):
    rgb_image_forward = drone_tool.get_current_image(image_type='scene', external=False, angle='forward')
    img = np.clip(rgb_image_forward, 0, 255)
    return img
    
def get_forward_img_depth(drone_tool):
    depth_image_forward = drone_tool.get_current_image(image_type='depth', external=False, angle='forward')
    img = np.clip(depth_image_forward, 0, 255)
    return img
    
def get_downward_img_rgb(drone_tool):
    rgb_image_downward = drone_tool.get_current_image(image_type='scene', external=False, angle='downward')
    img = np.clip(rgb_image_downward, 0, 255)
    return img

def get_downward_img_depth(drone_tool):
    depth_image_downward = drone_tool.get_current_image(image_type='depth', external=False, angle='downward')
    img = np.clip(depth_image_downward, 0, 255)
    return img



def setup_episode(episode_info, drone_tool):
    marker_pose = episode_info['marker_pose']
    drone_pose = episode_info['drone_pose']
    episode_time = episode_info['time']
    weather = episode_info['weather']

    drone_tool.set_marker_pose(marker_pose)
    drone_tool.reset_drone_pose(init_pose=drone_pose[0:-1], init_yaw=drone_pose[-1])
    drone_tool.set_time_of_day(episode_time)
    drone_tool.set_weather(weather)

    time.sleep(1)



def setup_episode_no_time_and_weather(episode_info, drone_tool):
    marker_pose = episode_info['marker_pose']
    drone_pose = episode_info['drone_pose']
    # episode_time = episode_info['time']
    # weather = episode_info['weather']

    drone_tool.set_marker_pose(marker_pose)
    drone_tool.reset_drone_pose(init_pose=drone_pose[0:-1], init_yaw=drone_pose[-1])
    # drone_tool.set_time_of_day(episode_time)
    # drone_tool.set_weather(weather)

    time.sleep(1)


def convert_eularianAngle2orientation(x_val, y_val, z_val, w_val):
    pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            x_val=x_val,
            y_val=y_val,
            z_val=z_val,
            w_val=w_val
        ))
    return pitch, roll, yaw




def check_collision(depth_image):
    '''
    depth_image: np.array, value range from 0 to 255, with size (h, w)
    '''
    image = np.clip(depth_image, 0, 255)
    img_collision_result = (image / 255 < 0.004).sum() / image.flatten().shape[0]
    collision = True if img_collision_result > 0.1 else False
    return collision


def shortest_geodesic_dist(data):
    """
    读取 ground-truth test.json 中第一个样本，
    用 marker_pose 与 drone_pose 的 (x, y) 计算直线距离。
    如果 test.json 里 marker_pose 与 drone_pose 在不同样本里不一致，可自行修改取法。
    """

    marker_x, marker_y = data["marker_pose"][:2]
    drone_x, drone_y = data["drone_pose"][:2]

    return math.hypot(marker_x - drone_x, marker_y - drone_y)
    

def compute_spl(path_len, shortest_dist, success):
    """
    根据公式 SPL = success * (shortest) / max(path_len, shortest)
    若 episode 未找到 marker，success = 0 → SPL = 0
    """

    if path_len == 0:
        # 没有行走（极端情况），视为失败
        return 0.0

    return (1 if success else 0) * shortest_dist / max(path_len, shortest_dist)


    
def compute_path_length(trajectory):
    '''
    trajectory: List (pose_x, pose_y, pose_z, ori_x, ori_y, ori_z, ori_w)
    movement pattern: xy-plane, first turn and then move forward, on z-axis, independently move vertically
    '''
    path_length = 0.0
    for i in range(1, len(trajectory)):
        dx = trajectory[i]['position_x'] - trajectory[i-1]['position_x']
        dy = trajectory[i]['position_y'] - trajectory[i-1]['position_y']
        dz = trajectory[i]['position_z'] - trajectory[i-1]['position_z']

        horizontal = math.hypot(dx, dy)  # √(dx² + dy²)
        vertical   = abs(dz)             # 单独上下移动
        path_length += horizontal + vertical
    return path_length

    

def coverage_ratio(
    traj: List[Dict[str, float]],
    radius: float = 30.0,
    fov_side: float = 9.0,
) -> float:
    """
    计算 coverage:
      - 以第 0 个点为圆心，半径 = radius (m) 的圆为目标区域；
      - 每个轨迹点覆盖一个边长 fov_side (m) 的正方形 (轴对齐)；
      - 计算正方形的并集与圆形的交集面积，占圆形总面积的百分比。

    Returns
    -------
    coverage ∈ [0, 1]
    """
    if not traj:
        return 0.0

    # 1️⃣ 圆形目标区域
    center_x = traj[0]["position_x"]
    center_y = traj[0]["position_y"]
    circle = Point(center_x, center_y).buffer(radius)  # buffer 用默认 16 - segment 逼近圆

    # 2️⃣ 所有正方形 polygon
    half = fov_side / 2.0
    squares = []
    for p in traj:
        cx, cy = p["position_x"], p["position_y"]
        squares.append(
            Polygon([
                (cx - half, cy - half),
                (cx + half, cy - half),
                (cx + half, cy + half),
                (cx - half, cy + half)
            ])
        )

    # 3️⃣ 并集 → 与圆形相交 → 面积
    union_squares = unary_union(squares)
    covered_area = union_squares.intersection(circle).area
    total_area   = circle.area

    return covered_area / total_area




