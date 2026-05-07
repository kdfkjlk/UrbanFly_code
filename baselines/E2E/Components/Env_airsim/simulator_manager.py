try:
    from .env_utils import (
        interpret_action,
        get_current_pose,
        get_current_rotation,
        setup_episode,
    )
except ImportError:
    from env_utils import (
        interpret_action,
        get_current_pose,
        get_current_rotation,
        setup_episode,
    )


class Simulator_Manager:
    def __init__(
        self,
        max_num_step,
        step_size=2,
        step_size_z=1,
        turn_angle=90,
        action_mapping_Simenv2Airsimenv=None,
    ):
        self.max_num_step = max_num_step
        self.step_size = step_size
        self.step_size_z = step_size_z
        self.turn_angle = turn_angle
        self.action_mapping_Simenv2Airsimenv = action_mapping_Simenv2Airsimenv

    def setup_new_episode(self, episode_info, drone_tool):
        setup_episode(episode_info, drone_tool)

    def step(self, action, drone_tool):
        current_position = get_current_pose(drone_tool)
        current_rotation = get_current_rotation(drone_tool)

        action = action["action"]
        if self.action_mapping_Simenv2Airsimenv is not None:
            action = self.action_mapping_Simenv2Airsimenv.get(action, action)

        new_pose = interpret_action(
            action,
            current_position,
            current_rotation,
            self.step_size,
            self.step_size_z,
            self.turn_angle,
        )
        drone_tool.set_drone_pose(new_pose)