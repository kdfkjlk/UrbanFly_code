# LLM-Agent for UAV Marker Search

This repository provides an LLM-based UAV navigation pipeline for marker search in AirSim environments. The agent uses onboard camera observations to explore the environment, identify a target marker, localize it in the final view, and evaluate navigation performance.

## Repository Structure

```text
LLM_agent/
├── airsim/                    # Customized AirSim Python client
├── configs/                   # API and AirSim settings
├── data_episode_drone/        # Episode metadata
├── evaluation/                # Evaluation scripts and utilities
├── llm_localization/          # LLM marker localization scripts
├── llm_nav/                   # LLM navigation scripts
├── obj_detect/                # Marker detector
├── logs/                      # Runtime outputs
├── requirements.txt
└── README.md
```

## Main Pipeline

The pipeline has three stages.

```text
1. LLM navigation
2. LLM marker localization
3. Evaluation
```

### Stage 1: LLM Navigation

Run:

```bash
python llm_nav/discovery_llm.py
```

This stage runs the LLM-based UAV agent in AirSim. It saves the navigation results to:

```text
logs/Nav_5patch_ModernCityEnvironment/
```

Each episode folder contains:

```text
explore.json
trajectory.json
collision.json
image_down_*.jpg
image_forward_*.jpg
```

### Stage 2: LLM Marker Localization

Run:

```bash
python llm_localization/extract_marker_pose_via_llm.py
```

This stage reads the final downward-facing image from each episode and asks the LLM to estimate the marker center in pixel coordinates.

The output is saved to:

```text
logs/marker_center_pose_from_llm_ModernCityEnvironment_2D.json
```

### Stage 3: Evaluation

Run:

```bash
python evaluation/evaluate_llm_det.py \
  --client_port 41451 \
  --map_name ModernCityEnvironment \
  --move_type 2D \
  --data_type test
```

The evaluation result is saved to:

```text
logs/evaluation/evaluation_result_ModernCityEnvironment_2D.json
```

The main metrics include:

```text
success
detect_false_positive
distance_NE
path_length
spl
collision
coverage
num_steps
```

## Installation

Create the environment:

```bash
conda create -n UrbenFly_LLM python=3.10 -y
conda activate UrbenFly_LLM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If AirSim reports a Tornado-related error, install the compatible version:

```bash
pip install "tornado==4.5.3"
```

## API Configuration

Add your OpenAi api key into LLM_agent/configs/api_config.py


## AirSim Settings

Two AirSim settings files are provided:

```text
configs/settings_nav.json    # for LLM navigation
configs/settings_eval.json   # for evaluation
```

Before running the three stages (LLM navigation, LLM marker localization, and evaluation),
Open a terminal in the map environment map
Run the command: sh AirSimEnv.sh --settings = path of the settings JSON file (depends on the task)



## Data

Episode metadata should be organized as:

```text
data_episode_drone/
└── test/
    └── ModernCityEnvironment/
        └── test.json
```

Other maps should follow the same structure:

```text
data_episode_drone/test/<MapName>/test.json
```


## Yolo11 Detector:
The weights can be downloaded from [UrbanFly_weights](https://huggingface.co/datasets/dfjkalfj/Urbanly_weights), and then put at LLM_agent/Obj_detect/Weights/best.pt.


## Notes

This repository includes a customized AirSim Python client under:

```text
airsim/
```

The navigation code depends on:

```python
from airsim.scenario_generation.scenario_manager import ScenarioManager
```

Therefore, users should use the local `airsim/` package included in this repository.


## License

This project includes a customized AirSim Python client. Please ensure the original AirSim license and attribution requirements are respected when redistributing the code.
