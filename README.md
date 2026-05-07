# UrbanFly Benchmark Code

This repository provides baseline implementations and utility scripts for **UrbanFly**, a simulation-based benchmark for UAV target acquisition in urban pre-landing scenarios.

UrbanFly evaluates whether an autonomous UAV can search a local urban area and acquire a designated landing target before the final precision-landing stage. The benchmark is built on AirSim and Unreal Engine environments and provides episode metadata, packaged maps, baseline agents, and evaluation scripts.

## Dataset and Environments

UrbanFly consists of two Hugging Face repositories:

- Dataset metadata and episode files: [UrbanFly_dataset](https://huggingface.co/datasets/dfjkalfj/UrbanFly_dataset)
- AirSim/Unreal Engine environment archives: [UrbanFly_envs](https://huggingface.co/datasets/dfjkalfj/UrbanFly_envs)

The dataset repository contains the episode metadata used to define UAV target-acquisition tasks. The environment repository contains the packaged AirSim/Unreal Engine maps required to run the benchmark.

## Repository Structure

Expected structure:

```text
UrbanFly/
├── DATA/
│   ├── train/
│   ├── val_unseen/
│   └── test/
│       └── MAP_NAME/
│           └── test.json
├── Envs/
│   └── MAP_NAME/
│       ├── AirSimEnv.sh
│       └── ...
├── baselines/
│   ├── E2E/
│   ├── Heuristic/
│   └── LLM_agent/
├── tool/
│   └── human_eval/
├── README.md
└── .gitignore
```

`DATA/` contains the UrbanFly episode metadata.

`Envs/` contains the packaged AirSim/Unreal Engine maps.

`baselines/` contains baseline evaluation code, including heuristic, end-to-end, and LLM-agent baselines.

`tool/human_eval/` contains utilities for human evaluation.

## Installation

Each baseline folder provides its own `README.md` and `requirements.txt`. Please install the dependencies required by the baseline you want to run.

For example:

```bash
cd baselines/Heuristic
pip install -r requirements.txt
```

## Running Baselines

Before running AirSim-based evaluation, manually launch the corresponding packaged environment:

```bash
cd Envs/MAP_NAME
sh AirSimEnv.sh --settings=/path/to/baselines/BASELINE_NAME/src/settings.json
```

Then enter the corresponding baseline folder and run its scripts. For example:

```bash
cd baselines/Heuristic
bash scripts/eval_heuristic_2D.sh
bash scripts/eval_heuristic_3D.sh
bash scripts/calculate_metrics.sh
```

Please refer to each baseline folder for detailed configuration, dependencies, and evaluation commands.

## Expected Dataset Structure

```text
UrbanFly_dataset/
└── DATA/
    ├── train/
    ├── val_unseen/
    └── test/
        └── MAP_NAME/
            └── test.json
```

Each episode file contains scenario metadata such as UAV initial pose, marker/target pose, map name, time-of-day condition, weather condition, and other task attributes.

## License

The dataset is released under CC BY-NC 4.0.

## Notes

This repository provides code and lightweight utilities. Large assets, including full dataset files, packaged simulation environments, and model weights, should be downloaded separately from the corresponding Hugging Face repositories.
