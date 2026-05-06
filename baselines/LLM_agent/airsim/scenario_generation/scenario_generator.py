from ..client import MultirotorClient, VehicleClient
from ..types import *
from .utils import sample_position_on_circle, validate_marker_position
from ..utils import to_quaternion
from .components.config import ACTOR_TYPE, BP_TYPE
from .components.weather import Weather
from .components.actor import Actor
from .components.time import Time
from .components.marker import Marker
import random
from .scenario import Scenario
from .scenario_manager import ScenarioManager




def generate_random_scenario(client, x_range, y_range, radius, z_dist=None, tp_marker_id=0):
    # set a random marker position
    while True:
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        z = client.simGetGroundHeight(x, y)
        marker_position = Vector3r(x, y, z)
        if validate_marker_position(client, marker_position):
            break

    tp_marker = Marker(tp_marker_id, None, Pose(marker_position), 'tp')

    # set a target gps position within the raidus range of the marker position
    # gps_pose = Pose(Vector3r(0, 10, 0))
    in_radius = random.uniform(0, radius)
    x, y = sample_position_on_circle(marker_position, in_radius)
    if not z_dist:
        z_dist = random.uniform(5, 50)
    gps_pose = Pose(Vector3r(x, y, z - z_dist), to_quaternion(0, 0, math.radians(random.uniform(-90, 90))))

    fp_markers= []
    drone_start_pose = Pose(Vector3r(0, 0, 0))
    drone_start_ground_height = client.simGetGroundHeight(drone_start_pose.position.x_val, drone_start_pose.position.y_val)
    drone_start_pose.position.z_val = drone_start_ground_height - 0.5
    weather = Weather(*([random.uniform(0, 0.3)] * 9))
    time = Time(random.random(), random.random())
    actors = []

    scenario = Scenario(tp_marker, fp_markers, drone_start_pose, gps_pose, radius, actors, weather, time)
    return scenario


def test_scenario_generator(sim_mode='cv'):
    if sim_mode == 'cv':
        client = VehicleClient()
    else:
        client = MultirotorClient()
    mgr = ScenarioManager(client, sim_mode=sim_mode)

    scenario = generate_random_scenario(client, [0, 100], [0, 100], 10)

    mgr.set_scenario(scenario)
    mgr.load_scenario()


if __name__ == '__main__':
    client = MultirotorClient()
    mgr = ScenarioManager(client, sim_mode='cv')

    scenario = generate_random_scenario(client, [0, 100], [0, 100], 10)

    mgr.set_scenario(scenario)
    mgr.load_scenario()

    print(scenario)