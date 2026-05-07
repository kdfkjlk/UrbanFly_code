# Heuristic Baseline for UrbanFly

This folder provides the heuristic baseline evaluation code for the UrbanFly benchmark. The baseline connects to an already running AirSim environment, executes predefined spiral or zigzag search trajectories in 2D or 3D, saves episode-level logs, and calculates total evaluation metrics.

# Getting started

## Step1: install dependencies

```bash
conda create -n UrbanFly_Heuristic python=3.8
conda activate UrbanFly_Heuristic
pip install -r requirements.txt
```

and install AirSim from its official website.

## Prepare the simulation environment

## Directory Structure

Expected structure:

```text
UrbanFly/
├── DATA/
│   └── test/
│       └── MAP_NAME/
│           └── test.json
├── Envs/
│   └── MAP_NAME/
│       ├── AirSimEnv.sh
│       └── ...
└── baselines/
    └── Heuristic/
        ├── Env/
        ├── Obj_Detect/
        │   ├── weights/
        │   │   └── best.pt
        │   └── yolo11_detector.py
        ├── scripts/
        │   ├── eval_heuristic_2D.sh
        │   ├── eval_heuristic_3D.sh
        │   └── calculate_metrics.sh
        ├── src/
        │   ├── agent_circle_2D.py
        │   ├── agent_circle_3D.py
        │   ├── result_analysis_metrics.py
        │   └── settings.json
        ├── logs/
        ├── requirements.txt
        └── README.md
```

`Heuristic/` contains the baseline code.

`DATA/` contains the UrbanFly episode data, and can be downloaded from [UrbanFly_dataset](https://huggingface.co/datasets/dfjkalfj/UrbanFly_dataset).

`Envs/` contains the AirSim simulation environments, and can be downloaded from [UrbanFly_envs](https://huggingface.co/datasets/dfjkalfj/UrbanFly_envs).

`Obj_Detect/weights/` contains the marker detection model weights.

## Running

You need a working AirSim + Unreal Engine environment. The evaluation scripts assume that the AirSim map has already been opened.

Launch one packaged AirSim map first. For example:

```bash
cd UrbanFly/Envs/MAP_NAME
sh AirSimEnv.sh --settings=/path/to/UrbanFly/baselines/Heuristic/src/settings.json
```

Before running the heuristic baseline, edit the map name and search type inside the corresponding script. The search type can be `spiral` or `zigzag`.

Run the 2D heuristic baseline:

```bash
cd UrbanFly/baselines/Heuristic
bash scripts/eval_heuristic_2D.sh
```

Run the 3D heuristic baseline:

```bash
cd UrbanFly/baselines/Heuristic
bash scripts/eval_heuristic_3D.sh
```

Calculate metrics after evaluation:

```bash
cd UrbanFly/baselines/Heuristic
bash scripts/calculate_metrics.sh
```