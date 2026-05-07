import argparse
import itertools
import json
import os
import time
from datetime import datetime

from human_eval_tool.client_RL import drone
from human_eval_tool.env_operator import simulator_handeler
from human_eval_tool.human_eval_tool import Human_eval_tool
from human_eval_tool.record_tool import Sim_state


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def generate_save_path(output_dir):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(output_dir, f"flight_log_{current_time}.json")


def connect_simulator(env_path=None, manual_env=True, timeout=180):
    if not manual_env:
        if env_path is None:
            raise ValueError("env_path must be provided when manual_env=False.")
        simulator_handeler.check_running_sh_file()
        simulator_handeler.run_shell_script(env_path)

    start = time.time()
    while time.time() - start < timeout:
        try:
            drone.initialize_client()
            print("Connected to AirSim successfully.")
            return True
        except Exception as e:
            print(f"Waiting for simulator connection... {e}")
            time.sleep(2)

    return False


def main(args):
    ensure_dir(args.output_dir)

    data_all = read_json(args.scenario_file)

    if args.sample_indices is not None:
        sample_indices = read_json(args.sample_indices)
        data_selected = [data_all[i] for i in sample_indices]
    else:
        data_selected = data_all

    data_selected.sort(key=lambda x: x["map_name"])
    grouped_data = itertools.groupby(data_selected, key=lambda x: x["map_name"])

    for map_name, group in grouped_data:
        env_path = os.path.join(args.env_root, map_name, "AirSimEnv.sh") if args.env_root else None

        print(f"Current map: {map_name}")
        if args.manual_env:
            print("Manual environment mode is enabled.")
            print("Please make sure the corresponding AirSim map is already running.")
        else:
            print(f"Launching environment from: {env_path}")

        connection_success = connect_simulator(
            env_path=env_path,
            manual_env=args.manual_env,
            timeout=args.timeout,
        )

        if not connection_success:
            raise RuntimeError(f"Failed to connect to AirSim for map: {map_name}")

        for data in group:
            print("data =", data)

            save_path = generate_save_path(args.output_dir)

            sim_state = Sim_state(init_data=data)
            human_eval_tool = Human_eval_tool()

            human_eval_tool.main(
                data,
                save_path,
                drone,
                sim_state,
            )

            del sim_state
            del human_eval_tool
            

        if not args.manual_env:
            simulator_handeler.close_sh_file()
            time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario_file", type=str, required=True)
    parser.add_argument("--env_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./logs")
    parser.add_argument("--sample_indices", type=str, default=None)
    parser.add_argument("--max_refly", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)

    parser.add_argument(
        "--manual_env",
        action="store_true",
        help="Use an already running AirSim environment instead of launching AirSimEnv.sh.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())