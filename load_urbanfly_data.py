#!/usr/bin/env python3
"""
Lightweight loader and inspection utility for the UrbanFly dataset.

This script downloads or reads the UrbanFly episode metadata and reports:
- number of JSON files
- number of episodes by split
- total number of episodes
- example fields from one episode
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_REPO_ID = "dfjkalfj/UrbanFly_dataset"


def download_dataset(output_dir: Path, repo_id: str = DEFAULT_REPO_ID) -> None:
    """Download only the DATA JSON files from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for downloading. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="DATA/**/*.json",
        local_dir=str(output_dir),
    )

    print(f"Downloaded DATA JSON files to: {output_dir}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_records(data: Any) -> List[Dict[str, Any]]:
    """
    Return episode records from common JSON structures.

    UrbanFly files are expected to be lists of episode dictionaries.
    This function also supports dictionary wrappers such as:
    {"episodes": [...]}, {"scenarios": [...]}, or {"data": [...]}.
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("episodes", "scenarios", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        return [data]

    return []


def inspect_data(data_root: Path) -> Tuple[int, Dict[str, int], List[Tuple[str, int]], Dict[str, Any]]:
    if not data_root.exists():
        raise FileNotFoundError(f"DATA root does not exist: {data_root}")

    json_files = sorted(data_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {data_root}")

    total = 0
    by_split: Dict[str, int] = defaultdict(int)
    by_file: List[Tuple[str, int]] = []
    example_record: Dict[str, Any] = {}

    for path in json_files:
        data = load_json(path)
        records = extract_records(data)
        n = len(records)

        rel = path.relative_to(data_root)
        split = rel.parts[0] if len(rel.parts) > 1 else "root"

        total += n
        by_split[split] += n
        by_file.append((str(rel), n))

        if not example_record and records:
            example_record = records[0]

    return total, dict(by_split), by_file, example_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect UrbanFly dataset metadata.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download DATA JSON files from Hugging Face before inspection.",
    )
    parser.add_argument(
        "--repo_id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo ID. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--output_dir",
        default="UrbanFly_dataset",
        help="Directory for downloaded files. Default: UrbanFly_dataset",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Path to DATA directory. If omitted, uses <output_dir>/DATA.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.download:
        download_dataset(output_dir=output_dir, repo_id=args.repo_id)

    data_root = Path(args.data_root) if args.data_root else output_dir / "DATA"

    total, by_split, by_file, example = inspect_data(data_root)

    print("\n===== UrbanFly DATA inspection =====")
    print(f"DATA root: {data_root}")
    print(f"JSON files: {len(by_file)}")

    print("\n===== Episodes by split =====")
    for split, count in sorted(by_split.items()):
        print(f"{split}: {count}")

    print(f"\nTotal episodes: {total}")

    print("\n===== Files =====")
    for file_path, count in by_file:
        print(f"{file_path}: {count}")

    if example:
        print("\n===== Example episode fields =====")
        for key, value in example.items():
            value_repr = repr(value)
            if len(value_repr) > 120:
                value_repr = value_repr[:117] + "..."
            print(f"{key}: {value_repr}")
    else:
        print("\nNo example episode found.")


if __name__ == "__main__":
    main()
