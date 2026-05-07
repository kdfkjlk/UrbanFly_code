import argparse
import json
import math
import numbers
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path.cwd()
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "logs"


def load_json(file_path: Path):
    with open(file_path, "r") as f:
        return json.load(f)


def natural_key(value):
    name = value.name if isinstance(value, Path) else str(value)
    return int(name) if name.isdigit() else name


def calculate_path_length(trajectory):
    path_length = 0.0
    for i in range(1, len(trajectory)):
        pos1 = np.array([
            trajectory[i - 1]["position_x"],
            trajectory[i - 1]["position_y"],
            trajectory[i - 1]["position_z"],
        ])
        pos2 = np.array([
            trajectory[i]["position_x"],
            trajectory[i]["position_y"],
            trajectory[i]["position_z"],
        ])
        path_length += float(np.linalg.norm(pos2 - pos1))
    return path_length


def calculate_init_distance_2d(episode_result):
    drone_pos = np.array(episode_result["episode_info"]["drone_pose"][:2])
    marker_pos = np.array(episode_result["episode_info"]["marker_pose"][:2])
    return float(np.linalg.norm(marker_pos - drone_pos))


def calculate_spl(success, shortest_path_length, path_length, epsilon=1e-6):
    if not success:
        return 0.0
    return float(shortest_path_length / (max(path_length, shortest_path_length) + epsilon))


def extract_episode_metrics(episode_result, trajectory):
    found_marker = bool(episode_result["found_marker"])
    success = bool(episode_result["success"])
    collision = bool(episode_result["collision"])
    false_positive = int(found_marker and not success)

    path_length = calculate_path_length(trajectory)
    shortest_path_length = calculate_init_distance_2d(episode_result)

    return {
        "num_steps": episode_result["steps"],
        "success": int(success),
        "collision": int(collision),
        "found_marker": int(found_marker),
        "false_positive": false_positive,
        "distance_to_marker": episode_result["distance_to_marker"],
        "path_length": path_length,
        "spl": calculate_spl(success, shortest_path_length, path_length),
    }


def average_metrics(metrics_list):
    if not metrics_list:
        raise ValueError("No valid episode results were found.")

    sums = defaultdict(float)
    counts = defaultdict(int)

    for episode_metrics in metrics_list:
        for key, value in episode_metrics.items():
            if isinstance(value, bool):
                value = int(value)
            if not isinstance(value, numbers.Number):
                raise ValueError(f"Unsupported metric type for '{key}': {type(value)}")
            sums[key] += value
            counts[key] += 1

    return {key: sums[key] / counts[key] for key in sums}


def find_episode_dirs(split_dir: Path, map_names=None):
    """
    Find all episode folders under the split directory and merge them for total evaluation.

    Supported structures:
      logs/E2E_2D/test/UrbanDistrict/0/explore_result.json
      logs/E2E_2D/test/0/explore_result.json

    If map_names is provided, only folders under those map names are used.
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Result split directory not found: {split_dir}")

    search_roots = []
    if map_names:
        for map_name in map_names:
            map_dir = split_dir / map_name
            if not map_dir.exists():
                raise FileNotFoundError(f"Map result directory not found: {map_dir}")
            search_roots.append(map_dir)
    else:
        search_roots.append(split_dir)

    episode_dirs = []
    for root in search_roots:
        for result_path in root.rglob("explore_result.json"):
            episode_dir = result_path.parent
            trajectory_path = episode_dir / "trajectory.json"
            if trajectory_path.exists():
                episode_dirs.append(episode_dir)

    return sorted(set(episode_dirs), key=lambda p: tuple(natural_key(part) for part in p.relative_to(split_dir).parts))


def collect_total_results(result_root, move_type, eval_type="test", map_names=None):
    split_dir = Path(result_root) / f"E2E_{move_type}" / eval_type
    episode_dirs = find_episode_dirs(split_dir, map_names=map_names)

    all_metrics = []
    skipped = []

    for episode_dir in episode_dirs:
        result_path = episode_dir / "explore_result.json"
        trajectory_path = episode_dir / "trajectory.json"

        try:
            episode_result = load_json(result_path)
            trajectory = load_json(trajectory_path)
            all_metrics.append(extract_episode_metrics(episode_result, trajectory))
        except Exception as exc:
            skipped.append((str(episode_dir), str(exc)))

    return all_metrics, skipped, episode_dirs


def print_total_summary(move_type, eval_type, metrics, skipped, episode_dirs):
    avg = average_metrics(metrics)

    print("=" * 72)
    print(f"Total evaluation results: E2E_{move_type} / {eval_type}")
    print(f"Merged episode folders: {len(episode_dirs)}")
    print(f"Evaluated episodes: {len(metrics)}")
    if skipped:
        print(f"Skipped invalid episode folders: {len(skipped)}")
    print("-" * 72)

    ordered_keys = [
        "success",
        "spl",
        "distance_to_marker",
        "false_positive",
        "collision",
        "found_marker",
        "num_steps",
        "path_length",
    ]

    for key in ordered_keys:
        if key in avg:
            print(f"{key}: {avg[key]:.6f}")

    print("=" * 72)

    if skipped:
        print("Skipped folders:")
        for folder, reason in skipped:
            print(f"  - {folder}: {reason}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze total E2E evaluation results by merging all episode folders under logs/E2E_*/test."
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=str(DEFAULT_RESULT_ROOT),
        help="Root log directory. Default: ../logs relative to this script.",
    )
    parser.add_argument(
        "--move_type",
        type=str,
        default="3D",
        choices=["2D", "3D"],
        help="Evaluation mode to analyze.",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="test",
        help="Evaluation split, e.g., test or val_unseen.",
    )
    parser.add_argument(
        "--map_names",
        nargs="*",
        default=None,
        help="Optional map names. If omitted, all episode folders under the split are merged.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics, skipped, episode_dirs = collect_total_results(
        result_root=args.result_root,
        move_type=args.move_type,
        eval_type=args.eval_type,
        map_names=args.map_names,
    )
    print_total_summary(args.move_type, args.eval_type, metrics, skipped, episode_dirs)
