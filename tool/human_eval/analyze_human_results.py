import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_flight_logs(log_dir):
    log_dir = Path(log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    paths = []
    for path in log_dir.rglob("*.json"):
        if path.name.startswith("flight_log_"):
            paths.append(path)

    return sorted(paths)


def calculate_path_length(trajectory):
    path_length = 0.0

    for i in range(1, len(trajectory)):
        p1 = np.array(trajectory[i - 1]["drone_position"], dtype=float)
        p2 = np.array(trajectory[i]["drone_position"], dtype=float)
        path_length += np.linalg.norm(p2 - p1)

    return float(path_length)


def calculate_init_distance_to_goal(log_data):
    marker_xy = np.array(log_data["marker_pose"][:2], dtype=float)
    drone_xy = np.array(log_data["drone_pose"][:2], dtype=float)
    return float(np.linalg.norm(marker_xy - drone_xy))


def calculate_distance_to_goal(log_data):
    marker_pos = np.array(log_data["marker_pose"], dtype=float)
    last_pos = np.array(log_data["trajectory"][-1]["drone_position"], dtype=float)

    # Human-eval logs store drone z in AirSim coordinates, where flying height is negative.
    # Marker z is already stored in the evaluation coordinate convention.
    # Therefore, flip the drone z before computing 3D distance.
    last_pos[-1] = -last_pos[-1]

    return float(np.linalg.norm(last_pos - marker_pos))


def calculate_horizontal_distance_to_goal(log_data):
    marker_xy = np.array(log_data["marker_pose"][:2], dtype=float)
    last_xy = np.array(log_data["trajectory"][-1]["drone_position"][:2], dtype=float)
    return float(np.linalg.norm(marker_xy - last_xy))


def extract_collision(log_data):
    if bool(log_data.get("collision", False)):
        return 1.0

    for step in log_data.get("trajectory", []):
        if bool(step.get("is_collide", False)) or bool(step.get("collided", False)):
            return 1.0

    return 0.0


def calculate_search_success(log_data):
    """
    This follows the old human-eval convention:
    success = 1 if the participant has entered the landing stage,
    usually by pressing Space after visually finding the marker.
    """
    trajectory = log_data.get("trajectory", [])
    if not trajectory:
        return 0.0

    if trajectory[-1].get("flight_stage") == "land":
        return 1.0

    for step in trajectory:
        if step.get("flight_stage") == "land":
            return 1.0

    return 0.0


def calculate_land_success(log_data, distance_to_goal, is_collide, success_distance=2.0):
    """
    Landing success is based on final distance to marker and collision state.
    """
    if distance_to_goal <= success_distance and is_collide == 0.0:
        return 1.0
    return 0.0


def calculate_spl(log_data, land_success):
    shortest_path_length = calculate_init_distance_to_goal(log_data)
    actual_path_length = calculate_path_length(log_data["trajectory"])

    if land_success != 1.0:
        return 0.0

    if max(actual_path_length, shortest_path_length) == 0:
        return 0.0

    return float(shortest_path_length / max(actual_path_length, shortest_path_length))


def analyze_one_log(log_path, success_distance=2.0):
    log_data = load_json(log_path)

    if "trajectory" not in log_data or len(log_data["trajectory"]) == 0:
        raise ValueError(f"No trajectory found in {log_path}")

    is_collide = extract_collision(log_data)
    distance_to_goal = calculate_distance_to_goal(log_data)
    horizontal_distance_to_goal = calculate_horizontal_distance_to_goal(log_data)
    path_length = calculate_path_length(log_data["trajectory"])
    success = calculate_search_success(log_data)
    land_success = calculate_land_success(
        log_data,
        distance_to_goal=distance_to_goal,
        is_collide=is_collide,
        success_distance=success_distance,
    )
    spl = calculate_spl(log_data, land_success)

    metrics = {
        "trajectory_file": str(log_path),
        "map_name": log_data.get("map_name", "unknown"),
        "success": success,
        "is_collide": is_collide,
        "distance_to_goal": distance_to_goal,
        "horizontal_distance_to_goal": horizontal_distance_to_goal,
        "path_length": path_length,
        "spl": spl,
        "land_success": land_success,
        "steps_taken": len(log_data["trajectory"]),
        "total_time": float(log_data["trajectory"][-1].get("time_s", 0.0)),
    }

    return metrics


def average_metrics(metrics_list):
    if not metrics_list:
        return {}

    numeric_keys = [
        "success",
        "is_collide",
        "distance_to_goal",
        "horizontal_distance_to_goal",
        "path_length",
        "spl",
        "land_success",
        "steps_taken",
        "total_time",
    ]

    avg = {}
    for key in numeric_keys:
        values = [m[key] for m in metrics_list if key in m]
        avg[key] = float(np.mean(values)) if values else float("nan")

    return avg


def print_summary(metrics_list, avg):
    print("=" * 80)
    print("Human Evaluation Metrics")
    print("=" * 80)
    print(f"Evaluated episodes: {len(metrics_list)}")
    print("-" * 80)

    print(f"Success:                    {avg.get('success', float('nan')):.4f}")
    print(f"Land Success:               {avg.get('land_success', float('nan')):.4f}")
    print(f"Collision Rate:             {avg.get('is_collide', float('nan')):.4f}")
    print(f"Distance to Goal:           {avg.get('distance_to_goal', float('nan')):.4f}")
    print(f"Horizontal Distance to Goal:{avg.get('horizontal_distance_to_goal', float('nan')):.4f}")
    print(f"Path Length:                {avg.get('path_length', float('nan')):.4f}")
    print(f"SPL:                        {avg.get('spl', float('nan')):.4f}")
    print(f"Steps Taken:                {avg.get('steps_taken', float('nan')):.4f}")
    print(f"Total Time:                 {avg.get('total_time', float('nan')):.4f}")
    print("=" * 80)


def save_results(metrics_list, avg, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "num_episodes": len(metrics_list),
        "average_metrics": avg,
        "episode_metrics": metrics_list,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved analysis results to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/human_eval",
        help="Directory containing human evaluation flight logs.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="logs/human_eval_metrics.json",
        help="Path to save the merged metric results.",
    )
    parser.add_argument(
        "--success_distance",
        type=float,
        default=2.0,
        help="Distance threshold for landing success.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log_paths = find_flight_logs(args.log_dir)
    if not log_paths:
        raise RuntimeError(f"No flight_log_*.json files found under: {args.log_dir}")

    metrics_list = []
    for log_path in log_paths:
        try:
            metrics = analyze_one_log(
                log_path,
                success_distance=args.success_distance,
            )
            metrics_list.append(metrics)
        except Exception as e:
            print(f"[Warning] Failed to process {log_path}: {e}")

    avg = average_metrics(metrics_list)
    print_summary(metrics_list, avg)
    save_results(metrics_list, avg, args.output_path)


if __name__ == "__main__":
    main()