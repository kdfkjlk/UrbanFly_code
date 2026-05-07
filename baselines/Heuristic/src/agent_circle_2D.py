import math
import os
import json
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import Optional, List
import argparse

from Env.test_env import Test_Env
from Env.read_data import load_one_map_data



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))


def load_scenario_info(map_name, mode="test", data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "DATA")

    data_dir = os.path.abspath(data_dir)

    print(f"[INFO] Loading data from: {data_dir}")
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Map: {map_name}")

    data = load_one_map_data(data_dir, map_name, mode)

    if data is None:
        expected_file = os.path.join(data_dir, mode, map_name, f"{mode}.json")
        raise FileNotFoundError(f"Cannot find scenario file: {expected_file}")

    return data



## generate spiral path ------------------------------------------------------------------------------
def generate_spiral_coverage_path(z, 
                                  R=35.0, 
                                  coverage_factor=0.4):
    """
    生成覆盖半径为 R 的圆区域的螺旋扫描轨迹。

    参数:
        z (float): agent 的高度 (m)。
        R (float, optional): 目标覆盖圆的半径，默认为 30 m。
        coverage_factor (float, optional): 视野覆盖范围相对于高度的比例，默认为 0.75。

    返回:
        List[List[float]]: 轨迹点列表，每个点为 [x, y]，以圆心 (0,0) 为原点。
    """
    # 计算视野覆盖半径
    # coverage_radius = coverage_factor * z
    coverage_radius = min(5, coverage_factor * z)
    
    # 螺旋参数 b 确保相邻一圈的径向间距 ≈ 2*coverage_radius
    b = coverage_radius / math.pi
    
    # 最大扫描角度，对应 r = b * theta_max = R
    theta_max = R / b  #  = R * π / coverage_radius
    
    # 设置角度步长，保证相邻点的弧长 <= coverage_radius
    # 弧长 ≈ r * dθ，当 r 接近 R 时最严苛：dθ <= coverage_radius / R
    dtheta = coverage_radius / R
    
    path = []
    # 首先添加圆心
    path.append([0.0, 0.0])
    
    theta = dtheta
    while theta <= theta_max:
        r = b * theta
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        path.append([x, y])
        theta += dtheta
    
    return path



def translate_spiral_coverage_path(z, x0, y0, R=30.0, coverage_factor=0.75):
    """
    生成以 (x0, y0) 为中心的覆盖半径为 R 的圆区域的螺旋扫描轨迹。

    参数:
        z (float): agent 的高度 (m)。
        x0 (float): 起始中心点 X 坐标。
        y0 (float): 起始中心点 Y 坐标。
        R (float, optional): 目标覆盖圆的半径，默认为 30 m。
        coverage_factor (float, optional): 视野覆盖范围相对于高度的比例，默认为 0.75。

    返回:
        List[List[float]]: 平移后的轨迹点列表，每个点为 [x, y]。
    """
    # 先生成以 (0,0) 为中心的螺旋轨迹
    raw_path = generate_spiral_coverage_path(z, R, coverage_factor)
    
    # 将每个点平移到 (x0, y0)
    translated_path = [[x0 + dx, y0 + dy] for dx, dy in raw_path]
    return translated_path





## generate zigzag path ------------------------------------------------------------------------------

def translate_zigzag_coverage_path(
        z: float,
        x0: float,
        y0: float,
        R: float = 30.0,
        coverage_factor: float = 0.75
    ):
    """
    将 zig-zag 轨迹平移到以 (x0, y0) 为圆心的作业区。
    """
    raw_path = generate_zigzag_center_start(z, R, coverage_factor)
    return [[x0 + dx, y0 + dy] for dx, dy in raw_path]



# 生成以圆心起飞、首条穿过圆心的 Zig-zag 轨迹
def generate_zigzag_center_start(
        z: float,
        R: float = 30.0,
        coverage_factor: float = 0.75,
        start_left: bool = True
    ) -> List[List[float]]:
    """
    以圆心起飞、首条穿过圆心的 zig-zag 轨迹。
    连续航点间距 ≤ coverage_radius，确保观测连续覆盖。
    返回坐标仍以 (0,0) 为圆心，可直接交给 translate_zigzag_coverage_path 平移。
    """
    # ---------- 基本参数 ----------
    coverage_radius = min(5.0, coverage_factor * z)   # 与 spiral 一致，最大 5 m
    strip_step      = 2.0 * coverage_radius           # 相邻条带中心间距

    # ---------- ① 生成未细分的“理想路径” ----------
    rows = [0.0]                                      # 条带中心 y 序列
    offset = strip_step
    while offset <= R - coverage_radius + 1e-6:
        rows.append(-offset)
        if offset <= R - coverage_radius - 1e-6:
            rows.append(offset)
        offset += strip_step
    rows = [r for r in rows if abs(r) <= R - coverage_radius + 1e-6]

    raw = [[0.0, 0.0]]                                # 起飞点 (圆心)
    half_w0 = R                                       # y=0 时交点 ±R

    if start_left:
        raw.extend([[-half_w0, 0.0], [ half_w0, 0.0]])
        going_right = True
    else:
        raw.extend([[ half_w0, 0.0], [-half_w0, 0.0]])
        going_right = False

    for y in rows[1:]:
        half_w = math.sqrt(max(R**2 - y**2, 0.0))
        if going_right:
            raw.extend([[ half_w, y], [-half_w, y]])
        else:
            raw.extend([[-half_w, y], [ half_w, y]])
        going_right = not going_right

    # ---------- ② 细分：步长 ≤ coverage_radius ----------
    path: List[List[float]] = [raw[0]]
    max_step = coverage_radius

    for i in range(1, len(raw)):
        x_prev, y_prev = path[-1]
        x_tgt,  y_tgt  = raw[i]
        dx, dy = x_tgt - x_prev, y_tgt - y_prev
        dist   = math.hypot(dx, dy)

        if dist <= max_step + 1e-6:
            path.append([x_tgt, y_tgt])
        else:
            n_seg = int(math.ceil(dist / max_step))
            for k in range(1, n_seg + 1):
                frac = min(k * max_step / dist, 1.0)
                path.append([x_prev + dx * frac,
                             y_prev + dy * frac])

    return path


# 绘制路径并在线段上添加方向箭头
def plot_zigzag_with_arrows(path,
                            R: float = 30.0,
                            arrow_interval: int = 1,
                            ax: Optional[plt.Axes] = None):
    """
    绘制 zig-zag 轨迹：
    * path      : [[x,y], ...] 序列
    * R         : 作业区半径，仅用于画外圈
    * arrow_interval : 每隔多少条线段画一个箭头 (≥1)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # 圆形作业区边界
    ax.add_patch(plt.Circle((0, 0), R, fill=False, lw=1.2, color='black'))

    # 绘制带箭头的折线
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        # 主线
        ax.plot([x1, x2], [y1, y2], color='orange', lw=1.8)

        # 箭头
        if i % arrow_interval == 0:
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle='-|>',  # 小三角箭头
                                    mutation_scale=10,
                                    color='orange',
                                    lw=0)              # 仅箭头，不额外描边
            ax.add_patch(arrow)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Zig-zag Coverage Path with Direction Arrows')



## -----------------------------------------------------------------------------------



def compute_path_length(path):
    """
    计算给定轨迹 path 的总路径长度。

    参数:
        path (List[List[float]]): 轨迹点列表，每个点为 [x, y]。

    返回:
        float: 路径长度（米）。
    """
    if not path or len(path) < 2:
        return 0.0

    total_length = 0.0
    for i in range(1, len(path)):
        x0, y0 = path[i-1]
        x1, y1 = path[i]
        dx = x1 - x0
        dy = y1 - y0
        total_length += math.hypot(dx, dy)
    return total_length




def evaluate(airsim_env, xy_trajectory, current_z, det_conf_thres, img_height):

    for step, tra in enumerate(xy_trajectory):
        target_pose = tra + [current_z]
        airsim_env.drone_tool.reset_drone_pose(target_pose, init_yaw=0)

        done, _ = airsim_env._check_done()

        if done:  ## find marker
            rgb_image = airsim_env._get_img()
            depth_image = airsim_env._get_depth_img()
            drone_pose = airsim_env.drone_tool.get_drone_pose()

            if rgb_image.shape[0] != img_height or depth_image.shape[0] != img_height:
                # rgb_image = np.zeros((224, 224, 3), dtype=np.uint8) 
                # depth_image = 100.0 * np.ones((480, 640, 1), dtype=np.float32)
                continue


            _, det_marker_pose_g, _ = airsim_env.det_model.detect_marker(rgb_image, depth_image, drone_pose, conf_thres=det_conf_thres)
            
            airsim_env.det_marker_pose_g = det_marker_pose_g
        time.sleep(0.1)  # 等待检测结果稳定


        metrics = airsim_env.update_metrics()
        metrics['done'] = True if done else False
        metrics['num_steps'] = step
        metrics['path_length'] = compute_path_length(xy_trajectory[:step + 1])
        # print(metrics)

        if done:
            break
        
    return metrics



def evaluate_with_collision(airsim_env, xy_trajectory, current_z, det_conf_thres, img_height):
    
    final_collision = False

    for step, tra in enumerate(xy_trajectory):
        # Calculate yaw angle to target position
        if step == 0:
            # For the first waypoint, use current pose as reference
            current_pose = airsim_env._get_agent_pose()
            prev_x, prev_y = current_pose[0], current_pose[1]
        else:
            # Use previous trajectory point as reference
            prev_x, prev_y = xy_trajectory[step-1][0], xy_trajectory[step-1][1]
        
        target_x, target_y = tra[0], tra[1]
        
        # Calculate yaw angle (direction from current to target)
        dx = target_x - prev_x
        dy = target_y - prev_y
        target_yaw = math.degrees(math.atan2(dy, dx))
        

        target_pose = tra + [current_z]
        airsim_env.drone_tool.reset_drone_pose(target_pose, init_yaw=target_yaw)
        collision_detected = airsim_env.check_collision()
        if collision_detected:
            final_collision = True



        # Final collision check before proceeding
        if final_collision:
            print("Collision detected during flight!")
            metrics = airsim_env.update_metrics()
            metrics['done'] = False
            metrics['num_steps'] = step
            metrics['path_length'] = compute_path_length(xy_trajectory[:step + 1])
            metrics['collision'] = final_collision
            terminate_pose = airsim_env.drone_tool.get_drone_pose()
            end_pose = [terminate_pose.position.x_val, terminate_pose.position.y_val, terminate_pose.position.z_val]
            metrics['last_drone_pose'] = end_pose
            return metrics

        done, _ = airsim_env._check_done()
        if done:  ## find marker
            rgb_image = airsim_env._get_img()
            depth_image = airsim_env._get_depth_img()
            drone_pose = airsim_env.drone_tool.get_drone_pose()

            if rgb_image.shape[0] != img_height or depth_image.shape[0] != img_height:
                # rgb_image = np.zeros((224, 224, 3), dtype=np.uint8) 
                # depth_image = 100.0 * np.ones((480, 640, 1), dtype=np.float32)
                continue

            _, det_marker_pose_g, _ = airsim_env.det_model.detect_marker(rgb_image, depth_image, drone_pose, conf_thres=det_conf_thres)
            
            airsim_env.det_marker_pose_g = det_marker_pose_g
        time.sleep(0.1)  # 等待检测结果稳定

        metrics = airsim_env.update_metrics()
        metrics['done'] = True if done else False
        metrics['num_steps'] = step
        metrics['path_length'] = compute_path_length(xy_trajectory[:step + 1])
        metrics['collision'] = final_collision  # No collision if we reach this point
        terminate_pose = airsim_env.drone_tool.get_drone_pose()
        end_pose = [terminate_pose.position.x_val, terminate_pose.position.y_val, terminate_pose.position.z_val]
        metrics['last_drone_pose'] = end_pose
        # print(metrics)

        if done:
            break
        
    return metrics
        


def make_serializable(obj):
    # numpy 标量
    if isinstance(obj, np.generic):
        return obj.item()
    # dict/list 递归
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(x) for x in obj]
    return obj

def append_result_jsonl(results: dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 先转换
    serializable_results = make_serializable(results)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(serializable_results, ensure_ascii=False) + '\n')

    



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--map_name", type=str, default="ModernCityEnvironment")
    parser.add_argument("--flytype", type=str, default="spiral", choices=["spiral", "zigzag"])
    parser.add_argument("--mode", type=str, default="test", choices=["train", "val_unseen", "test"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--det_conf_thres", type=float, default=0.8)
    parser.add_argument("--img_height", type=int, default=480)
    args = parser.parse_args()


    map_name = args.map_name
    flytype = args.flytype
    data = load_scenario_info(map_name, mode=args.mode, data_dir=args.data_dir)
    print('Number of episode: ', len(data))

    det_conf_thres = args.det_conf_thres
    airsim_env = Test_Env(max_num_step=500, det_conf_thres=det_conf_thres)
    
    
    for idx, episode_info in enumerate(data):

        airsim_env.reset(episode_info)

        xy_pos = airsim_env._get_agent_pose()
        agent_z = airsim_env.get_agent_height()
        print('agent_z: ', agent_z)

        x0, y0 = xy_pos[0], xy_pos[1]   # 比如 agent 初始在 (50, -10)

        if flytype == 'spiral':
            trajectory = translate_spiral_coverage_path(agent_z, x0, y0)
        elif flytype == 'zigzag':
            trajectory = translate_zigzag_coverage_path(agent_z, x0, y0)

        print('Length of trajectory: ', len(trajectory))

        if map_name == 'UrbanDistrict':
            metrics = evaluate(airsim_env, trajectory, agent_z, det_conf_thres, img_height=480)
        else:
            metrics = evaluate_with_collision(airsim_env, trajectory, agent_z, det_conf_thres, img_height=480)

        result = {
            'id': idx,
            'map_name': map_name,
            'episode_info': episode_info,
            'metrics': metrics
        }
        print(result)

        if args.output_dir is None:
            output_dir = os.path.join(BASELINE_DIR, 'logs', 'agent_2D')
        else:
            output_dir = args.output_dir

        output_path = os.path.join(output_dir, f'evaluate_results_{map_name}_{flytype}.json')
        append_result_jsonl(results=result, file_path=output_path)
        
        print(idx, metrics)
    


    


