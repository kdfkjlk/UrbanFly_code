# E2E Baseline for UrbanFly

This folder provides the end-to-end (E2E) baseline evaluation code for the UrbanFly benchmark. The baseline connects to an already running AirSim environment, executes the trained 2D or 3D policy, saves episode-level logs, and calculates total evaluation metrics.


# Getting started

## Step1: install dependencies

```bash
conda create -n UrbanFly_E2E python=3.8
conda activate UrbanFly_E2E
pip install -r requirements.txt
```
and install Airsim from its official website.


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
    └── E2E/
        ├── Components/
        ├── Obj_Detect/
        |   |—— weights/
        |       |——best_model.pth
        ├── scripts/
        │   ├── eval_E2E_2D.sh
        │   ├── eval_E2E_3D.sh
        │   └── calculate_metrics.sh
        ├── src/
        │   ├── eval_airsim_E2E_2D.py
        │   ├── eval_airsim_E2E_3D.py
        │   ├── result_analysis.py
        |   |—— settings.json
        │   └── weights/
        │       ├── best_model_2D.pth
        │       └── best_model_3D.pth
        ├── logs/
        ├── requirements.txt
        └── README.md
        
```

`E2E/` contains the baseline code. 
`DATA/` contains the UrbanFly episode data, and can be downloaded from [UrbanFly_dataset](https://huggingface.co/datasets/dfjkalfj/UrbanFly_dataset).
`Envs/` contains the Airsim simulation environments, and can be downloaded from [UrbanFly_envs](https://huggingface.co/datasets/dfjkalfj/UrbanFly_envs).
`weights/` can be downloaded from . [UrbanFly_weights](https://huggingface.co/datasets/dfjkalfj/Urbanly_weights).



## Running

You need a working AirSim + Unreal Engine environment. The evaluation scripts assume that the AirSim map has already been opened. You need to run the command "sh AirSimEnv.sh --settings = settings json file path (src / settings.json)" in the terminal under one packaged Airsim map file folder.

Run the 2D E2E baseline:

```bash
bash scripts/eval_E2E_2D.sh
```

Run the 3D E2E baseline:

```bash
bash scripts/eval_E2E_3D.sh
```

Calculate Metrics

After evaluation, calculate total metrics with:

```bash
bash scripts/calculate_metrics.sh
```

