# UrbanFly: A Benchmark for UAV Pre-landing Target Acquisition in Urban Environments

---

## Notes

This repository provides the evaluation code and baseline implementations for UrbanFly. The dataset files, simulation environments, and large model weights are hosted separately and can be downloaded for running the benchmark.

## Content

- Introduction
- Getting Started
- Usage
- Baselines
- Acknowledgment

## Introduction

Unmanned aerial vehicle (UAV) delivery has gained increasing attention for fast and flexible service in dense urban environments, yet current systems remain constrained by fragile positioning assumptions and predefined landing infrastructure, limiting deployment to arbitrary urban destinations. We present UrbanFly, a large-scale benchmark for pre-landing target acquisition in autonomous UAV delivery under coupled perception, geometry, and environmental challenges. UrbanFly contains 10,201 episodes across 17 near-photorealistic urban maps, generated through a systematic construction pipeline covering diverse spatial configurations, operating conditions, and environmental variations. Evaluations with representative autonomous agents and human operation reveal the difficulty of coordinating aerial navigation, local exploration, and endpoint acquisition in complex scenarios， highlighting pre-landing stage as a critical bottleneck for scalable UAV delivery. 
  UrbanFly aims to advance research on robust pre-landing aerial autonomy in complex real-world environments.

## Getting Started

### Step1: Install dependencies

Each baseline has its own dependencies. Please enter the corresponding baseline folder for detailed README and installation requirements.

AirSim should also be installed following its official instructions.

### Step2: Prepare the simulation environments

The AirSim/Unreal Engine environments can be downloaded from:

[UrbanFly_envs](https://huggingface.co/datasets/dfjkalfj/UrbanFly_envs)

After downloading, place the environments under:

```text
Envs/
├── ModernCityEnvironment/
├── UrbanDistrict/
├── AbandonCity_PostSoviet/
└── ...
```

Each map folder should contain an executable script such as:

```text
AirSimEnv.sh
```

### Step3: Prepare the dataset files

The UrbanFly episode metadata can be downloaded from:

[UrbanFly_dataset](https://huggingface.co/datasets/dfjkalfj/UrbanFly_dataset)

The dataset directory should be structured as follows:

```text
DATA/
├── train/
|—— val_seen/
├── val_unseen/
└── test/
    └── MAP_NAME/
        └── test.json
```

## Project Directory Structure

Your workspace directory should be structured as follows:

```text
UrbanFly/
├── DATA/
├── Envs/
├── baselines/
│   ├── E2E/
│   ├── Heuristic/
│   └── LLM_agent/
├── tool/
│   └── human_eval/
├── README.md
└── .gitignore
```

## Usage

### 1. Launch the AirSim environment

Before running any baseline, first open one packaged AirSim map. For example:

```bash
cd Envs/ModernCityEnvironment
sh AirSimEnv.sh --settings=/path/to/settings.json
```

The settings file should correspond to the baseline being evaluated.

### 2. Run the baseline

```bash
cd baselines/Heuristic
bash scripts/eval_heuristic_2D.sh
bash scripts/eval_heuristic_3D.sh
bash scripts/calculate_metrics.sh
```

Before running, edit the map name and search type inside the corresponding script. The search type can be `spiral` or `zigzag`.

### 3. Run the E2E baseline

```bash
cd baselines/E2E
bash scripts/eval_E2E_2D.sh
bash scripts/eval_E2E_3D.sh
bash scripts/calculate_metrics.sh
```

Before running, edit the map name and evaluation setting inside the corresponding script.

### 4. Run the LLM-agent baseline

```bash
cd baselines/LLM_agent
```

Please refer to the README inside `baselines/LLM_agent/` for detailed instructions.

## Baselines

UrbanFly currently provides the following baselines:

- **Heuristic baseline**: predefined spiral and zigzag search trajectories.
- **E2E baseline**: trained end-to-end policies for 2D and 3D navigation.
- **LLM-agent baseline**: language-model-based high-level decision-making agent.

Each baseline folder contains its own scripts, requirements, logs, and README.


## Acknowledgment

This project is built upon AirSim and Unreal Engine simulation environments, and AerialVLN. We thank the open-source UAV navigation and embodied AI communities for their contributions.
