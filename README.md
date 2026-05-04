# UrbanFly Benchmark Code

This repository provides dataset access and inspection utilities for UrbanFly, a simulation-based benchmark for UAV target acquisition in urban pre-landing scenarios.

## Dataset

UrbanFly consists of two Hugging Face repositories:

- Dataset metadata and episode files: https://huggingface.co/datasets/dfjkalfj/UrbanFly_dataset
- AirSim/Unreal Engine environment archives: https://huggingface.co/datasets/dfjkalfj/UrbanFly_envs

The dataset repository contains the episode metadata used to define UAV target-acquisition tasks. The environment repository contains the packaged AirSim/Unreal Engine maps required to run the benchmark.

## Installation

```bash
pip install huggingface_hub
```

## Usage

Download and inspect the UrbanFly episode files:

```bash
python load_urbanfly_data.py --download --output_dir UrbanFly_dataset
```

Inspect an already downloaded dataset:

```bash
python load_urbanfly_data.py --data_root UrbanFly_dataset/DATA
```

## Expected Dataset Structure

```text
UrbanFly_dataset/
└── DATA/
    ├── train/
    ├── val_seen/
    ├── val_unseen/
    └── test/
```

Each episode file contains scenario metadata such as UAV initial pose, marker/target pose, map name, time-of-day condition, and weather parameters.

## License

The dataset is released under CC BY-NC 4.0.
