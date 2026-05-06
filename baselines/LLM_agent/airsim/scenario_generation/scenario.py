import json
import random
from .components.actor import Actor
from .components.weather import Weather
from .components.time import Time
from .components.marker import Marker
from ..types import *
from .components.utils import pose_to_dict, pose_from_vec, pose_from_dict

class Scenario(object):
    def __init__(self, tp_marker=None, fp_markers=[], drone_start_pose=None, gps_pose=None, radius=None, actors=[], weather=None, time=None):
        self.tp_marker = tp_marker
        self.fp_markers = fp_markers
        self.drone_start_pose = drone_start_pose
        self.gps_pose = gps_pose
        self.radius = radius # marker sample radius
        self.actors = actors
        self.weather = weather
        self.time = time
    

    def init(self, tp_marker, fp_markers, drone_start_pose, gps_pose, radius, actors, weather, time):
        self.tp_marker = tp_marker
        self.fp_markers = fp_markers
        self.drone_start_pose= drone_start_pose
        self.gps_pose = gps_pose
        self.radius = radius
        self.actors = actors
        self.weather = weather
        self.time = time

    def to_json(self, save_path=None):
        json_dict = {}
        json_dict['tp_marker'] = self.tp_marker.to_dict()
        json_dict['drone_start_pose'] = pose_to_dict(self.drone_start_pose)
        json_dict['gps_pose'] = pose_to_dict(self.gps_pose)
        json_dict['radius'] = self.radius   
        json_dict['weather'] = self.weather.to_dict()
        json_dict['time'] = self.time.to_dict()

        json_dict['fp_markers'] = [marker.to_dict() for marker in self.fp_markers]
        json_dict['actors'] = [actor.to_dict() for actor in self.actors]

        if save_path:
            with open(save_path, 'w') as json_file:
                json.dump(json_dict, json_file)

        return json_dict


    def to_vec_dict(self):
        vec_dict = {}
        vec_dict['tp_marker'] = self.tp_marker.to_vec()
        vec_dict['drone_start_pose'] = self.drone_start_pose.to_vec()
        vec_dict['gps_pose'] = self.gps_pose.to_vec()
        vec_dict['radius'] = self.radius
        vec_dict['weather'] = self.weather.to_vec()
        vec_dict['time'] = self.time.to_vec()

        vec_dict['fp_markers'] = [marker.to_vec() for marker in self.fp_markers]
        vec_dict['actors'] = [actor.to_vec() for actor in self.actors]
        
        return vec_dict
    
    def load_from_vec_dict(self, vec_dict):
        self.tp_marker = Marker.from_vec(vec_dict['tp_marker'])
        self.drone_start_pose = pose_from_vec(vec_dict['drone_start_pose'])
        self.gps_pose = pose_from_vec(vec_dict['gps_pose'])

        self.radius = vec_dict['radius']

        self.weather = Weather.from_vec(vec_dict['weather'])
        self.time = Time.from_vec(vec_dict['time'])

        self.fp_markers = [Marker.from_vec(vec_data) for vec_data in vec_dict['fp_markers']]
        
        # Assuming the actor vector has 8 values: type, start_x, start_y, start_z, start_angle, end_x, end_y, end_z, speed
        self.actors = []
        for actor_vec in vec_dict['actors']:
            # actor_type = actor_vec[0]
            # start_pose = Pose(*actor_vec[1:5])
            # end_pose = Pose(*actor_vec[5:9])
            # speed = actor_vec[9]
            self.actors.append(Actor.from_vec(actor_vec))

    def load_from_json(self, json_path):
        if type(json_path) == str:
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
        else:
            json_dict = json_path
        self.tp_marker = Marker.from_dict(json_dict['tp_marker'])
        self.drone_start_pose = pose_from_dict(json_dict['drone_start_pose'])
        self.gps_pose = pose_from_dict(json_dict['gps_pose'])
        self.radius = json_dict['radius']
        self.weather = Weather.from_dict(json_dict['weather'])
        self.time = Time.from_dict(json_dict['time'])

        self.fp_markers = [Marker.from_dict(marker_data) for marker_data in json_dict['fp_markers']]
        self.actors = [Actor.from_dict(actor_data) for actor_data in json_dict['actors']]      
    
    def load_from_msg(self, scenario_msg):
        self.tp_marker = Marker.from_msg(scenario_msg.tp_marker)
        self.drone_start_pose = Pose.from_msg(scenario_msg.drone_start_pose)
        self.gps_pose = Pose.from_msg(scenario_msg.gps_pose)
        self.radius = scenario_msg.radius
        self.weather = Weather.from_msg(scenario_msg.weather)
        self.time = Time.from_msg(scenario_msg.time)

        self.fp_markers = [Marker.from_msg(marker_msg) for marker_msg in scenario_msg.fp_markers]
        self.actors = [Actor.from_msg(actor_msg) for actor_msg in scenario_msg.actors]      

    def mutate(self, mutation_rate=0.3):
        self.tp_marker.mutate(mutation_rate)
        self.drone_start_pose.mutate(mutation_rate)
        self.gps_pose.mutate(mutation_rate)
        self.weather.mutate(mutation_rate)
        self.time.mutate(mutation_rate)
        flag = random.random()
        if flag <= mutation_rate:
            self.radius += random.uniform(-3, 3)

        for marker in self.fp_markers:
            marker.mutate(mutation_rate)
        
        # mutate actors
        flag = random.random() # whether mutate the number of actors
        if flag <= mutation_rate:
            num_actors = len(self.actors) + random.randint(-2, 2)
        else:
            num_actors = len(self.actors)

        if num_actors <= 0:
            self.actors = []
        elif num_actors <= len(self.actors):
            self.actors = self.actors[:num_actors]
            for i in range(num_actors):
                self.actors[i].mutate()
        else:
            for i in range(len(self.actors)):
                self.actors[i].mutate()
            
            # for i in range(num_actors - len(self.actors)):
            #     self.actors.append(sample_actor(self.gps_pose))
        
        # check numbers of different types of actors
        # actor_types = {}



    @classmethod
    def crossover(cls, scenario_1, scenario_2):
        # tp_marker_crossover
        tp_marker_1, tp_marker_2 = Marker.crossover(scenario_1.tp_marker, scenario_1.tp_marker)
        drone_start_pose_1, drone_start_pose_2 = Pose.crossover(scenario_1.drone_start_pose, scenario_2.drone_start_pose)
        gps_pose_1, gps_pose_2 = Pose.crossover(scenario_1.gps_pose, scenario_2.gps_pose)
        radius_1, radius_2 = scenario_2.radius, scenario_1.radius
        weather_1, weather_2 = Weather.crossover(scenario_1.weather, scenario_2.weather)
        time_1, time_2 = Time.crossover(scenario_1.time, scenario_2.time)

        fp_markers_1, fp_markers_2 = scenario_2.fp_markers, scenario_1.fp_markers
        actors_1, actors_2 = scenario_2.actors, scenario_1.actors

        new_scenario_1 = Scenario()
        new_scenario_2 = Scenario()
        new_scenario_1.init(tp_marker_1, fp_markers_1, drone_start_pose_1, gps_pose_1, radius_1, actors_1, weather_1, time_1)
        new_scenario_2.init(tp_marker_2, fp_markers_2, drone_start_pose_2, gps_pose_2, radius_2, actors_2, weather_2, time_2)

        return new_scenario_1, new_scenario_2


    def __str__(self):
        return str(self.to_json())

