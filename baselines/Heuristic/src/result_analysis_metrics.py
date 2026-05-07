import os
import json
import argparse
import math
import csv


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_bool_int(x):
    return 1 if bool(x) else 0


def is_nan(x):
    try:
        return math.isnan(float(x))
    except Exception:
        return False


def mean(values):
    values = [v for v in values if v is not None and not is_nan(v)]
    if len(values) == 0:
        return float("nan")
    return sum(values) / len(values)


def euclidean_2d(a, b):
    if not isinstance(a, list) or not isinstance(b, list):
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def infer_flytype_from_filename(filename):
    basename = os.path.basename(filename)
    if "_spiral" in basename:
        return "spiral"
    if "_zigzag" in basename:
        return "zigzag"
    return "unknown"


def read_one_result_file(file_path):
    records = []
    flytype = infer_flytype_from_filename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)
            episode_info = raw.get("episode_info", {}) or {}
            metrics = raw.get("metrics", {}) or {}

            marker_pose = episode_info.get("marker_pose")
            init_drone_pose = episode_info.get("drone_pose")
            last_drone_pose = metrics.get("last_drone_pose")

            gt_distance = euclidean_2d(marker_pose, init_drone_pose)
            final_distance = euclidean_2d(marker_pose, last_drone_pose)

            success = safe_bool_int(metrics.get("success", 0))
            done = safe_bool_int(metrics.get("done", 0))
            collision = safe_bool_int(metrics.get("collision", 0))

            path_length = safe_float(metrics.get("path_length"), default=0.0)
            num_steps = safe_float(metrics.get("num_steps"), default=0.0)
            time_consumed = safe_float(metrics.get("time_consumed"), default=0.0)

            if final_distance is not None:
                navigation_error = final_distance
            else:
                navigation_error = safe_float(metrics.get("distance_drone2marker"), default=float("nan"))

            if gt_distance is not None and path_length > 0:
                spl = success * gt_distance / max(gt_distance, path_length)
            else:
                spl = float("nan")

            false_detection = 1 if done == 1 and success == 0 else 0

            records.append({
                "source_file": os.path.basename(file_path),
                "line_id": line_id,
                "id": raw.get("id", raw.get("episode_id", line_id)),
                "map_name": raw.get("map_name", episode_info.get("map_name", "unknown")),
                "flytype": flytype,
                "success": success,
                "done": done,
                "collision": collision,
                "false_detection": false_detection,
                "path_length": path_length,
                "num_steps": num_steps,
                "time_consumed": time_consumed,
                "navigation_error": navigation_error,
                "spl": spl,
            })

    return records


def load_results(result_dir, flytype):
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")

    result_files = []
    for filename in os.listdir(result_dir):
        if not filename.startswith("evaluate_results_"):
            continue
        if not filename.endswith(".json"):
            continue
        if f"_{flytype}.json" not in filename:
            continue
        result_files.append(os.path.join(result_dir, filename))

    if not result_files:
        raise FileNotFoundError(
            f"No result files found for flytype='{flytype}' in: {result_dir}"
        )

    all_records = []
    for file_path in sorted(result_files):
        print(f"[INFO] Loading {file_path}")
        all_records.extend(read_one_result_file(file_path))

    return all_records


def summarize(records):
    return {
        "num_episodes": len(records),
        "num_maps": len(set(r["map_name"] for r in records)),
        "success_rate": mean([r["success"] for r in records]),
        "spl": mean([r["spl"] for r in records]),
        "navigation_error": mean([r["navigation_error"] for r in records]),
        "path_length": mean([r["path_length"] for r in records]),
        "num_steps": mean([r["num_steps"] for r in records]),
        "time_consumed": mean([r["time_consumed"] for r in records]),
        "done_rate": mean([r["done"] for r in records]),
        "false_detection_rate": mean([r["false_detection"] for r in records]),
        "collision_rate": mean([r["collision"] for r in records]),
    }


def format_value(v):
    if isinstance(v, float):
        if is_nan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def print_summary(summary, method, flytype, result_dir):
    print("\n" + "=" * 80)
    print(f"Heuristic {method} - {flytype} - average over all maps")
    print("=" * 80)
    print(f"Result dir: {result_dir}")
    print("-" * 80)

    for key, value in summary.items():
        print(f"{key}: {format_value(value)}")


def save_csv(summary, output_csv, method, flytype):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    row = {
        "method": method,
        "flytype": flytype,
        **summary,
    }

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"\n[INFO] Saved CSV summary to: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--flytype", type=str, required=True, choices=["spiral", "zigzag"])
    parser.add_argument("--method", type=str, default="unknown")
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    records = load_results(args.result_dir, args.flytype)
    summary = summarize(records)

    print_summary(summary, args.method, args.flytype, args.result_dir)

    if args.output_csv is not None:
        save_csv(summary, args.output_csv, args.method, args.flytype)


if __name__ == "__main__":
    main()