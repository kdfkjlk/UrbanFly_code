import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import math
from collections import defaultdict
from typing import List, Dict, Any
import numbers
import cv2
import time
from natsort import natsorted
import argparse

import airsim

from evaluation.client import Drone_tool, drone_config
from Obj_detect.yolo11_detector import detector
from evaluation.env_utils import (
    read_data,
    update_distance_drone2marker,
    update_success,
    get_downward_img_rgb,
    get_downward_img_depth,
    get_forward_img_rgb,
    get_forward_img_depth,
    setup_episode,
    setup_episode_no_time_and_weather,
    convert_eularianAngle2orientation,
    get_current_pose,
    check_collision,
    shortest_geodesic_dist,
    compute_spl,
    compute_path_length,
    coverage_ratio,
)

from llm_localization.convert_detection_coord import convert_coord_pix2realworld




def detect_marker(rgb_image, depth_image, drone_pose, conf_thres=0.8):
    _, det_marker_pose_g, detect_result_conf = detector.detect_marker(rgb_image, depth_image, drone_pose, conf_thres)
    
    if detect_result_conf >= conf_thres:
        return det_marker_pose_g   ## (x,y,z)
    else:
        return None
    



def img_show(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)




def get_collision_3D(collision_file_path):
    with open(collision_file_path, "r") as f:
        data = json.load(f)

    if data == 'no_collision':
        collision = False
    elif data == 'has_collision':
        collision = True
    
    return collision


def get_move_result_explore_history(movement_result_dir, episode_idx):
    '''
    input: movement dir and name of the episode (in str(idx) format)
    function: read explore.json file, including "next_region", "found_marker", "reason", "confidence" at each step
    return: [{"next_region", "found_marker", "reasoning", "confidence"}, ...]
    '''
    result_folder = os.path.join(movement_result_dir, episode_idx)
    explore_file_path = os.path.join(result_folder, 'explore.json')
    explore_result = read_data(explore_file_path)
    return explore_result


def get_move_result_trajectory(movement_result_dir, episode_idx, move_type='2D'):
    '''
    function: read trajectory.json
    output: [{ "position_x", "position_y", "position_z", 
    "orientation_x", "orientation_y", "orientation_z", "orientation_w"}, ...]
    '''
    result_folder = os.path.join(movement_result_dir, episode_idx)

    if move_type == '3D':
        trajectory_file_path = os.path.join(result_folder, 'trajectory.json')
        trajectory = read_data(trajectory_file_path)

    elif move_type == '2D':
        try:
            trajectory_file_path = os.path.join(result_folder, 'trajectory_correct.json')
            trajectory = read_data(trajectory_file_path)
        except:
            trajectory_file_path = os.path.join(result_folder, 'trajectory.json')
            trajectory = read_data(trajectory_file_path)

    return trajectory


def get_move_result_collision(movement_result_dir=None, episode_idx=None, trajectory=None, drone_tool=None, move_type='2D'):
    '''
    function: get collision info in the movement of one episode
            if 3D movement format: read collision.json directly;
            if 2D movement format: move along the trajectory and get collision info
    output: return collisin (True / False)

    Note: for moreExp, all 2D version contain collision.json file,
          for previous experiments, only 3D version contain collision json file
    '''

    ## for previous experiments, only 3D version contain collision json file -------------------------------
    # if move_type == '3D':
    #     assert movement_result_dir is not None
    #     assert episode_idx is not None

    #     result_folder = os.path.join(movement_result_dir, episode_idx)
    #     collision_file_path = os.path.join(result_folder, 'collision.json')
    #     collision = get_collision_3D(collision_file_path)

    # elif move_type == '2D':
    #     assert drone_tool is not None
    #     assert trajectory is not None
    #     collision = test_collision_one_episode(drone_tool, trajectory) 


    ## for moreExp, all 2D version contain collision.json file -------------------------------------------
    assert movement_result_dir is not None
    assert episode_idx is not None

    result_folder = os.path.join(movement_result_dir, episode_idx)
    collision_file_path = os.path.join(result_folder, 'collision.json')
    collision = get_collision_3D(collision_file_path)


    return collision



def fine_grained_detection(drone_tool, conf_thres):
    '''
    function: fine-grained detection of the marker
    input: drone_tool, llm_det_file_path, episode_idx, det_type, conf_thres
    output: det_marker_pose_g
    '''
    drone_pose = drone_tool.get_drone_pose()
            
    # set drone position at marker nearby location
    target_drone_height = 10
    drone_x = drone_pose.position.x_val
    drone_y = drone_pose.position.y_val
    
    pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
        x_val=drone_pose.orientation.x_val,
        y_val=drone_pose.orientation.y_val,
        z_val=drone_pose.orientation.z_val,
        w_val=drone_pose.orientation.w_val
    ))
    target_yaw = math.degrees(yaw)


    target_pose_candidates = [
        [drone_x - 5, drone_y - 5, target_drone_height],
        [drone_x - 5, drone_y + 5, target_drone_height],
        [drone_x + 5, drone_y - 5, target_drone_height],
        [drone_x + 5, drone_y + 5, target_drone_height]
    ]


    for target_pose in target_pose_candidates:
            
        drone_tool.reset_drone_pose(target_pose, target_yaw)
        time.sleep(0.5)

        img_rgb_downward = get_downward_img_rgb(drone_tool)
        img_depth_downward = get_downward_img_depth(drone_tool)
        drone_pose = drone_tool.get_drone_pose()
        det_marker_pose_g = detect_marker(img_rgb_downward, img_depth_downward, drone_pose, conf_thres)

        if det_marker_pose_g is not None:
            return det_marker_pose_g

    return None


def get_detection_result_world_coord(drone_tool, llm_det_file_path=None, episode_idx=None, det_type='detector', conf_thres=0.8, use_fine_grind_det=False):
    '''
    input: llm marker detection dir
            info for drone position setting
    function: get detection result in pixel coordinates, 
                and detection result in world coordinates if marker detected
    return: det_world_coord (List[x,y,z] or None if not exist)
    '''

    if det_type == 'llm':
        response = read_data(llm_det_file_path)[str(episode_idx)]['result']

        ## marker not detected
        if response is not None:
            det_result_pix_center_coord = response['marker_position']
        else:
            return None
        
        # marker detected
        if det_result_pix_center_coord is not None:

            # ----------------------------------------------------------------------------------------------
            # # get detection coordiantes in world-coordinate
            # img_rgb_downward = get_downward_img_rgb(drone_tool)
            # img_depth_downward = get_downward_img_depth(drone_tool)

            # print('img.shape, ', img_rgb_downward.shape)
            # print('detected result: ', det_result_pix_center_coord)

            # # img_show(img_rgb_downward)

            # drone_pose = drone_tool.get_drone_pose()
            # det_marker_pose_g = convert_coord_pix2realworld(det_result_pix_center_coord, img_rgb_downward, img_depth_downward, drone_pose)
            # -----------------------------------------------------------------------------------------------


            # get detection coordiantes in world-coordinate, where the detected location is near the marker
            img_rgb_downward = get_downward_img_rgb(drone_tool)
            img_depth_downward = get_downward_img_depth(drone_tool)

            # img_show(img_rgb_downward)

            drone_pose = drone_tool.get_drone_pose()
            near_marker_pose_g = convert_coord_pix2realworld(det_result_pix_center_coord, img_rgb_downward, img_depth_downward, drone_pose)


            # set drone position at marker nearby location
            current_drone_height = drone_pose.position.z_val

            if use_fine_grind_det:
                target_drone_height = min(-current_drone_height, 15)
            else:
                target_drone_height = 13


            # target_drone_height = -current_drone_height
            target_pose = near_marker_pose_g[0:2] + [target_drone_height]
            pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
                x_val=drone_pose.orientation.x_val,
                y_val=drone_pose.orientation.y_val,
                z_val=drone_pose.orientation.z_val,
                w_val=drone_pose.orientation.w_val
            ))
            target_yaw = math.degrees(yaw)
            
            drone_tool.reset_drone_pose(target_pose, target_yaw)
            time.sleep(0.5)


            img_rgb_downward = get_downward_img_rgb(drone_tool)
            img_depth_downward = get_downward_img_depth(drone_tool)
            drone_pose = drone_tool.get_drone_pose()
            det_marker_pose_g = detect_marker(img_rgb_downward, img_depth_downward, drone_pose, conf_thres)

            ## further check if the marker can be detected
            if det_marker_pose_g is None and use_fine_grind_det:
                det_marker_pose_g = fine_grained_detection(drone_tool, conf_thres)
        
            # print('current drone height is ', target_pose)
            # img_show(img_rgb_downward)

        else:
            return None  


    elif det_type == 'detector':
        # get detection coordiantes in world-coordinate
        img_rgb_downward = get_downward_img_rgb(drone_tool)
        img_depth_downward = get_downward_img_depth(drone_tool)
        drone_pose = drone_tool.get_drone_pose()
        det_marker_pose_g = detect_marker(img_rgb_downward, img_depth_downward, drone_pose, conf_thres)

    return det_marker_pose_g



def get_drone_pose_trajectory2worldcoord(trajectory_coord):
    """
    input: coordinates recorded in trajectory.json
    output: world coordinates in airsim env
    """
    target_pose = [
        trajectory_coord['position_x'], 
        trajectory_coord['position_y'], 
        -trajectory_coord['position_z']
    ]

    _,_,target_yaw = convert_eularianAngle2orientation(
        trajectory_coord['orientation_x'],
        trajectory_coord['orientation_y'],
        trajectory_coord['orientation_z'],
        trajectory_coord['orientation_w']
    )
            
    # Convert yaw from radians to degrees for reset_drone_pose
    target_yaw_degrees = math.degrees(target_yaw)
    return target_pose, target_yaw_degrees


def judge_success(episode_info, det_marker_pose_g, explore_result, success_dis=2):
    
    if explore_result[-1]['found_marker']:
        distance = update_distance_drone2marker(det_marker_pose_g, episode_info["marker_pose"])
        print(f'found marker, and distance={distance}')
        success = update_success(distance, success_dis)
        print('distance: ', distance, 'success: ', success)
        print('marker pose={}, det_pose={}'.format(episode_info["marker_pose"], det_marker_pose_g))

        if success:
            false_positive = False
        else:
            false_positive = True
    
    else: 
        success = False
        false_positive = False
    
    return success, false_positive





    

def test_collision_one_episode(drone_tool, trajectory):
    for i in range(len(trajectory)):
        pose = [trajectory[i]['position_x'], trajectory[i]['position_y'], -trajectory[i]['position_z']]
        pitch, roll, yaw = convert_eularianAngle2orientation(
            x_val=trajectory[i]['orientation_x'],
            y_val=trajectory[i]['orientation_y'],
            z_val=trajectory[i]['orientation_z'],
            w_val=trajectory[i]['orientation_w']
        )
        # Convert yaw from radians to degrees for reset_drone_pose
        yaw_degrees = math.degrees(yaw)
        drone_tool.reset_drone_pose(pose, yaw_degrees)

        img_depth_forward = get_forward_img_depth(drone_tool)
        img_depth_downward = get_downward_img_depth(drone_tool)

        collision_forward = check_collision(img_depth_forward)
        collision_downward = check_collision(img_depth_downward)
        collision = collision_forward or collision_downward

        if collision:
            return True
    
    return False

        
        
def calculate_NE(goal_pose, stop_pose):
    distance = update_distance_drone2marker(goal_pose, stop_pose)
    return distance



    

def test_one_episode(
        episode_info_all, 
        drone_tool, 
        movement_result_dir, 
        episode_idx, 
        llm_det_file_path, 
        move_type='2D', 
        det_type='detector',
        success_dis_thres=2, conf_thres=0.8,
        use_fine_grind_det=False
    ):

    episode_info = episode_info_all[int(episode_idx)]
    explore_result = get_move_result_explore_history(movement_result_dir, episode_idx)
    trajectory = get_move_result_trajectory(movement_result_dir, episode_idx, move_type)

    setup_episode(episode_info, drone_tool)
    time.sleep(2)

    # evaluate collision
    if episode_info["map_name"] == 'UrbanDistrict':
        collision = False
    else:
        collision = get_move_result_collision(
            movement_result_dir=movement_result_dir, 
            episode_idx=episode_idx, 
            trajectory=trajectory, 
            drone_tool=drone_tool, 
            move_type=move_type
        )

    # set drone at the last position of the trajectory
    target_trajectory_pose = trajectory[-1]
    target_pose, target_yaw_degrees = get_drone_pose_trajectory2worldcoord(target_trajectory_pose)
    drone_tool.reset_drone_pose(target_pose, target_yaw_degrees)
    time.sleep(0.5)


    # evluate SR
    if not collision:
        det_marker_pose_g = get_detection_result_world_coord(
            drone_tool=drone_tool, 
            llm_det_file_path=llm_det_file_path, 
            episode_idx=episode_idx, 
            det_type=det_type, 
            conf_thres=conf_thres,
            use_fine_grind_det=use_fine_grind_det
        )

        if det_marker_pose_g is not None:
            success, false_positive = judge_success(
                episode_info, 
                det_marker_pose_g, 
                explore_result, 
                success_dis=success_dis_thres
            )
        else:
            success, false_positive = False, False

    else:
        success = False
        false_positive = False
        det_marker_pose_g = None

    ## evluate NE
    if det_marker_pose_g is not None and success:
        distance_ne = calculate_NE(episode_info["marker_pose"], det_marker_pose_g)
    else:
        current_pose = get_current_pose(drone_tool)
        distance_ne = calculate_NE(episode_info["marker_pose"], current_pose)
    
    ## evaluate spl
    path_length = compute_path_length(trajectory)
    shortest_path_length = shortest_geodesic_dist(episode_info)
    spl = compute_spl(path_length, shortest_path_length, success)

    coverage = coverage_ratio(trajectory, radius=30.0, fov_side=9.0)


    metrics = {
        'collision': collision,
        'num_steps': len(trajectory),
        'success': success,
        'detect_false_positive': false_positive,
        'distance_NE': distance_ne,
        'path_length': path_length,
        'spl': spl,
        'coverage': coverage
    }

    return metrics





def average_episode_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    对一批 episode 的评估指标取平均。
    - 布尔量先转成 1 / 0
    - 数值 (int / float) 直接平均
    """
    if not all_metrics:
        raise ValueError("all_metrics 为空")

    # 汇总
    sums   = defaultdict(float)
    counts = defaultdict(int)

    for ep in all_metrics:
        for k, v in ep.items():
            # 把 bool 转为 int；其余保持原样
            if isinstance(v, bool):
                v = int(v)
            elif not isinstance(v, numbers.Number):
                raise TypeError(f"键 {k} 的值 {v} 不是数值或布尔类型，无法取平均")

            sums[k]   += v
            counts[k] += 1

    # 计算平均
    return {k: sums[k] / counts[k] for k in sums}
    
    





def run(episode_path, result_path, llm_det_file_path, drone_tool, map_name, save_path, 
        move_type='2D', det_type='detector', success_dis_thres=2, conf_thres=0.8, use_fine_grind_det=False):
    '''
    Inputs:
        move_type: "3D" records collision, while "2D" need to move along the trajectory to check at each time step
        det_type:  "llm" or "detector" to detect marker center position
    Function: load episode_info, explore_history, trajectory, detection result, and calculate metrics
    Return: evaluation result of all episodes 
    '''
    
    evaluation_result = {}
    episode_info_all = read_data(episode_path)
    tested_episode = os.listdir(result_path)
    tested_episode = natsorted(tested_episode)


    for i, episode_name in enumerate(tested_episode):

        print('Processing episode {}'.format(episode_name))

        metric = test_one_episode(
            episode_info_all=episode_info_all, 
            drone_tool=drone_tool, 
            movement_result_dir=result_path, 
            episode_idx=episode_name, 
            llm_det_file_path=llm_det_file_path, 
            move_type=move_type, 
            det_type=det_type,
            success_dis_thres=success_dis_thres, conf_thres=conf_thres,
            use_fine_grind_det = use_fine_grind_det
        )
        evaluation_result[episode_name] = metric


        print('\n')


    json.dump(evaluation_result, open(save_path, "w"))

    avg_metrics = average_episode_metrics(evaluation_result.values())
    print(f'Evaluation result: on {len(tested_episode)} samples, average result is {avg_metrics}')
    
    return evaluation_result





def test_several_sample(episode_path, result_path, llm_det_file_path, drone_tool, map_name, 
        move_type='2D', det_type='detector', success_dis_thres=2, conf_thres=0.8):
    episode_info_all = read_data(episode_path)

    episode_name = [str(i) for i in [143]]
    for episode_idx in episode_name:
        metric = test_one_episode(
            episode_info_all=episode_info_all, 
            drone_tool=drone_tool, 
            movement_result_dir=result_path, 
            episode_idx=episode_idx, 
            llm_det_file_path=llm_det_file_path, 
            move_type=move_type, 
            det_type=det_type,
            success_dis_thres=success_dis_thres, conf_thres=conf_thres,
            use_fine_grind_det=use_fine_grind_det
        )
        print('metric', metric, '\n')

    


def load_evaluation_result(jsonl_path):
    result = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # print(data)
            result.append(data)

    return result




def parse_args():
    parser = argparse.ArgumentParser(description='LLM-agent evaluation')
    parser.add_argument("--client_port", default=41451, type=int, help='client port to connect to AirSim map')
    parser.add_argument('--map_name', default='ModernCityEnvironment', type=str)
    parser.add_argument('--move_type', default='2D', type=str, help='2D/3D, corresponds to different LLM-agent')
    parser.add_argument('--data_type', default='test', type=str, help='test, val_unseen, val_seen, train')
    parser.add_argument('--use_fine_grind_det', default=True, type=bool)
    return parser.parse_args()





if __name__ == '__main__':
    drone_tool = Drone_tool(drone_config)

    args = parse_args()
    client_port = args.client_port
    map_name = args.map_name
    move_type = args.move_type
    data_type = args.data_type
    use_fine_grind_det = args.use_fine_grind_det

    drone_tool.initialize_client(client_port=client_port)

    # Episode metadata
    episode_path = os.path.join(
        PROJECT_ROOT,
        'data_episode_drone',
        data_type,
        map_name,
        f'{data_type}.json'
    )

    # Movement/exploration results generated by discovery_llm.py
    # Example: LLM_agent/logs/Nav_5patch_ModernCityEnvironment/
    result_path = os.path.join(
        PROJECT_ROOT,
        'logs',
        f'Nav_5patch_{map_name}'
    )

    # LLM marker localization JSON generated by extract_marker_pose_via_llm.py
    # Example: LLM_agent/logs/marker_center_pose_from_llm_ModernCityEnvironment_2D.json
    llm_det_file_path = os.path.join(
        PROJECT_ROOT,
        'logs',
        f'Localize_{map_name}_{move_type}.json'
    )

    # Evaluation output
    save_folder_path = os.path.join(
        PROJECT_ROOT,
        'logs',
        f'evaluation_{map_name}_{move_type}'
    )
    os.makedirs(save_folder_path, exist_ok=True)

    save_path = os.path.join(
        save_folder_path,
        f'evaluation_result_{map_name}_{move_type}.json'
    )

    print('episode_path:', episode_path)
    print('result_path:', result_path)
    print('llm_det_file_path:', llm_det_file_path)
    print('save_path:', save_path)
    print('use_fine_grind_det:', use_fine_grind_det)

    if not os.path.isfile(episode_path):
        raise FileNotFoundError(f'Episode file not found: {episode_path}')

    if not os.path.isdir(result_path):
        raise FileNotFoundError(f'Result directory not found: {result_path}')

    if not os.path.isfile(llm_det_file_path):
        raise FileNotFoundError(f'LLM detection file not found: {llm_det_file_path}')

    evaluation_result = run(
        episode_path=episode_path,
        result_path=result_path,
        llm_det_file_path=llm_det_file_path,
        drone_tool=drone_tool,
        map_name=map_name,
        save_path=save_path,
        move_type=move_type,
        det_type='llm',
        success_dis_thres=2,
        conf_thres=0.8,
        use_fine_grind_det=use_fine_grind_det
    )