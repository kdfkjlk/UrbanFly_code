import torch
import numpy as np
import argparse
import os
import time
import json
from Components.evaluation_env import DummyEnv_obstacle_airsimmode_evaluate
from Components.policy import RL_Explore_Policy
# from Components.Env_airsim.env_oprator import open_map, close_map
from Components.Env_airsim.env_utils import (
    get_downward_img_rgb, get_downward_img_depth, get_current_pose, 
    get_current_rotation, update_distance_drone2marker, update_success, 
    # setup_episode
)

# Import yolo11 detector
from Obj_Detect.yolo11_detector import detector


def parse_args():
    parser = argparse.ArgumentParser(description="AirSim Environment Evaluation")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--env_mode", default="dummy_obstacle", type=str, help="Environment mode")
    parser.add_argument("--obstacle_mode", default="airsim", type=str, help="Obstacle mode: airsim")
    parser.add_argument("--turn_penalty", default=False, type=bool, help="Turn penalty flag")

    parser.add_argument("--model_name", default="best_model.pth", type=str, help="Model filename")
    parser.add_argument("--render", default=False, type=bool, help="Whether to render during evaluation")
    parser.add_argument("--max_steps", default=400, type=int, help="Maximum steps per episode")
    parser.add_argument("--conf_thres", default=0.8, type=float, help="YOLO confidence threshold")
    parser.add_argument("--success_dis", default=2.0, type=float, help="Success distance threshold")
    parser.add_argument("--num_processes", default=1, type=int, help="Number of processes (should be 1 for evaluation)")

    parser.add_argument("--client_port", default=41452, type=int, help="Client_port for AirSim map connection")
    parser.add_argument("--train_mode", default="val_unseen", type=str, help="select mode: train, val, test")
    parser.add_argument("--train_stage", default=6, type=int, help="Training stage to evaluate")
    parser.add_argument("--model_path", type=str, help="Path to model directory")
    parser.add_argument("--content_dir", type=str, help="Test dataset path")
    parser.add_argument("--map_name", default="IndustrialArea", type=str, help="Map name to test")
    parser.add_argument("--results_dir", type=str, help="Results save directory")
    

    return parser.parse_args()



def load_test_episodes(test_data_dir, map_name, mode):
    """
    Load test episodes from the specified map
    """
    test_file_path = os.path.join(test_data_dir, map_name, f"{mode}.json")
    # test_file_path = os.path.join(test_data_dir, map_name, "val_unseen.json")
    
    if not os.path.exists(test_file_path):
        print(f"Error: Test file not found at {test_file_path}")
        return None
    
    with open(test_file_path, 'r') as f:
        episodes = json.load(f)
    
    print(f"Loaded {len(episodes)} episodes from {map_name}")
    return episodes



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




def save_trajectory(trajectory_data, map_name, episode_id, results_dir):
    """
    Save trajectory data to JSON file
    """
    map_dir = os.path.join(results_dir, map_name)
    episode_dir = os.path.join(map_dir, f"{episode_id}")
    os.makedirs(episode_dir, exist_ok=True)
    
    trajectory_file = os.path.join(episode_dir, "trajectory.json")
    with open(trajectory_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2)



def save_explore_result(explore_result, map_name, episode_id, results_dir):
    """
    Save exploration results to JSON file
    """
    map_dir = os.path.join(results_dir, map_name)
    episode_dir = os.path.join(map_dir, f"{episode_id}")
    os.makedirs(episode_dir, exist_ok=True)
    
    explore_file = os.path.join(episode_dir, "explore_result.json")
    with open(explore_file, 'w') as f:
        json.dump(explore_result, f, indent=2)


def evaluate_single_episode(env, policy, device, episode_info, episode_id, args):
    """
    Evaluate a single episode
    """
    print(f"\n=== Episode {episode_id} ===")
    
    # Setup episode in environment
    setup_episode(episode_info, env.drone_tool)
    
    obs, info = env.reset()
    
    # Convert observations to torch tensors
    for k, v in obs.items():
        v = torch.tensor(v, dtype=torch.float32)
        v = v.unsqueeze(0)  # Add batch dimension
        obs[k] = v.to(device)

    # Custom done condition for evaluation - no need for env.step's done
    done = False
    step_count = 0
    collision_detected = False
    found_marker = False
    success = False
    env_done = False
    
    # Initialize trajectory and collision data
    trajectory_data = []
    
    # Record initial position
    initial_pose = get_current_pose(env.drone_tool)
    initial_rotation = get_current_rotation(env.drone_tool)
    trajectory_data.append({
        "position_x": initial_pose[0],
        "position_y": initial_pose[1], 
        "position_z": initial_pose[2],
        "orientation_x": initial_rotation[0],
        "orientation_y": initial_rotation[1],
        "orientation_z": initial_rotation[2],
        "orientation_w": initial_rotation[3]
    })


    while not done and step_count < args.max_steps:
        with torch.no_grad():
            _, action, _ = policy.act(obs, deterministic=True)
            action = action.item()

        obs, _, env_done, info = env.step({'action': action})
        if env_done:
            done = True
            break
        
        # Convert observations to torch tensors
        for k, v in obs.items():
            v = torch.tensor(v, dtype=torch.float32)
            v = v.unsqueeze(0)  # Add batch dimension
            obs[k] = v.to(device)

        step_count += 1
        
        # Record current position after step
        current_pose = get_current_pose(env.drone_tool)
        current_rotation = get_current_rotation(env.drone_tool)
        trajectory_data.append({
            "position_x": current_pose[0],
            "position_y": current_pose[1],
            "position_z": current_pose[2],
            "orientation_x": current_rotation[0],
            "orientation_y": current_rotation[1],
            "orientation_z": current_rotation[2],
            "orientation_w": current_rotation[3]
        })
        
        # Check for collision - this is our primary done condition
        if info.get("collision", False):
            collision_detected = True
            done = True  # End episode on collision
            print(f"  Collision detected at step {step_count}! Episode ended.")
            break
        
        # Use YOLO11 to detect marker
        img_rgb_downward = get_downward_img_rgb(env.drone_tool)
        img_depth_downward = get_downward_img_depth(env.drone_tool)
        drone_pose = env.drone_tool.get_drone_pose()
        
        detect_result_post, det_marker_pose_g, detect_result_conf = detector.detect_marker(
            img_rgb_downward, img_depth_downward, drone_pose, args.conf_thres
        )
        
        if det_marker_pose_g:
            found_marker = True
            done = True

            distance = update_distance_drone2marker(det_marker_pose_g[:3], episode_info["marker_pose"])
            if update_success(distance, args.success_dis):
                success = True
                print('find marker')
            break

    
    if not found_marker:
        current_pose = get_current_pose(env.drone_tool)
        distance = update_distance_drone2marker(current_pose[:2], episode_info["marker_pose"][0:2])

    
    if collision_detected:
        reason_for_end = "collision" 
    elif found_marker:
        reason_for_end = "found_marker" 
    elif env_done:
        reason_for_end = "env_done"
    else:
        reason_for_end = "max_steps"
    
    explore_result = {
        "episode_id": episode_id,
        "episode_info": episode_info,
        "steps": step_count,
        "success": success,
        "collision": collision_detected,
        "found_marker": found_marker,
        "distance_to_marker": distance,
        "reason_for_end": reason_for_end
    }

    # Save trajectory and collision data
    save_trajectory(trajectory_data, args.map_name, episode_id, args.results_dir)
    save_explore_result(explore_result, args.map_name, episode_id, args.results_dir)
    
    return explore_result






def main():

    ## 2D algorithms evaluation-----------------------------------------------------




    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test episodes
    episodes = load_test_episodes(args.content_dir, args.map_name, args.train_mode)
    if episodes is None:
        return
    
    # Initialize environment for AirSim
    print("Initializing AirSim environment...")
    env = DummyEnv_obstacle_airsimmode_evaluate(args)
    
    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space
    
    # Initialize policy network
    print("Initializing policy network...")
    policy = RL_Explore_Policy(obs_space, action_space, args).to(device)
    
    # Load pretrained model
    model_file_path = args.model_path
    
    if os.path.exists(model_file_path):
        print(f"Loading pretrained model from: {model_file_path}")
        state_dict = torch.load(model_file_path, map_location=lambda storage, loc: storage)
        policy.load_state_dict(state_dict)
        policy.eval()
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file not found at {model_file_path}")
        return

    # Open AirSim map
    # print(f"Opening AirSim map: {args.map_name}")
    # open_map(args.map_name)
    # time.sleep(10)  # Wait for map to load

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Start evaluation on all episodes
    print(f"\nStarting evaluation on {len(episodes)} episodes...")
    
    for i, episode_info in enumerate(episodes):
        result = evaluate_single_episode(env, policy, device, episode_info, i, args)
        print(result)
            
        


if __name__ == "__main__":
    main()
