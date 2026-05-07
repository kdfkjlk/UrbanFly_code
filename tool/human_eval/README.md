# Human Evaluation Tool

This tool provides a human-operation interface for collecting human pilot trajectories in UrbanFly. It connects to an already running AirSim environment, displays real-time visual observations, accepts keyboard controls, and saves the recorded trajectory.

## Interface

The interface includes:

- forward RGB view
- downward RGB view
- forward depth view
- countdown timer
- feedback window
- task control panel

Participants first press `Return` to activate the episode. After visually identifying the marker, they press `Space` to mark the search phase as complete, and then continue controlling the UAV for tasks.

## Setup

Create a conda environment and install dependencies:

```bash
conda create -n urbanfly_human python=3.8
conda activate urbanfly_human
pip install -r requirements.txt
```


## AirSim Launch

Start the target AirSim map manually before running the tool:

```bash
cd cd /path/to/UrbanFly/Envs/FactoryDistrict
bash AirSimEnv.sh --settings /path/to/settings.example.json
```

The UAV configurations can be custimized in settings.example file, and the provided settings.json is an example. 
After launching the Airsim map, then wait until the AirSim/Unreal window is fully loaded.


## Run

From the human evaluation tool directory:

```bash
cd UrbanFly_code/tools/human_eval
```

Run:

```bash
python run_human_eval.py \
  --scenario_file scenarios/samples.json \
  --output_dir logs/human_eval \
  --manual_env
```

To run selected scenarios only:

```bash
python run_human_eval.py \
  --scenario_file scenarios/samples.json \
  --sample_indices configs/sample_indices.json \
  --output_dir logs/human_eval \
  --manual_env
```

The sample index file should be a JSON list:

```json
[0, 3, 8, 12]
```

The provided `samples.json` contains scenarios from `FactoryDistrict`, so please launch the corresponding AirSim map before running the tool, or change to other sample episode data.



## Controls

| Key | Action |
|---|---|
| `Return` | Activate episode |
| `Space` | Mark search completion |
| `Up Arrow` | Move forward |
| `Left Arrow` | Move left |
| `Right Arrow` | Move right |
| `W` | Ascend |
| `S` | Descend |
| `A` | Turn left |
| `D` | Turn right |
| `Esc` | End episode manually, if enabled |


## Output

Flight logs are saved to logs file, and each log contains the scenario metadata and the recorded UAV trajectory, including position, orientation, flight stage, timestamp, key press, collision state, and success state.


## Analyze results

```bash
python analyze_human_results.py \
  --log_dir logs/human_eval \
  --output_path logs/human_eval_metrics.json
```


## Notes

This version assumes manual AirSim launch. Automated map startup and shutdown will be added in a future update.


An episode is considered successful when the UAV reaches the marker within the configured landing threshold (2 meters). In the current implementation, `is_end=True` indicates successful landing, while collision, timeout, or manual quit should be treated as unsuccessful termination cases.