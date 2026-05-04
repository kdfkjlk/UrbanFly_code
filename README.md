# UrbanFly Benchmark Code

This repository provides dataset access and inspection utilities for UrbanFly, a simulation-based benchmark for UAV target acquisition in urban pre-landing scenarios.

## Dataset

UrbanFly consists of two Hugging Face repositories:

- Dataset metadata and episode files:  
  https://huggingface.co/datasets/dfjkalfj/UrbanFly_dataset

- AirSim/Unreal Engine environment archives:  
  https://huggingface.co/datasets/dfjkalfj/UrbanFly_envs

The dataset repository contains the episode metadata used to define UAV target-acquisition tasks. The environment repository contains the packaged AirSim/Unreal Engine maps required to run the benchmark.

## Current Contents

- `load_urbanfly_data.py`: downloads or loads UrbanFly episode files, counts episodes by split, and prints basic field information.

## Installation

```bash
pip install huggingface_hub
