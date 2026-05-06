import time
import airsim
import numpy as np
import math

# try:
#     from agents.drone.drone_config import drone_config
# except:
#     from drone_config import drone_config


drone_config = {
    "sim_mode": "cv",
    "host_ip": '127.0.0.1',
    "drone_name": '',
    'camera_names': {'external': {'downward': 'seg_cam', 'forward': 'forward_check'},
                     'internal': {'downward': 'bottom_center', 'forward':'front_center'}},
    'help_camera_name': 'seg_help_cam'
}



class Drone_tool():
    def __init__(self, drone_config):
        self.drone_config = drone_config
        self.sim_mode = drone_config['sim_mode']
        self.drone_name = drone_config['drone_name']

        self.camera_name = drone_config['camera_names']
        self.help_camera_name = drone_config['help_camera_name']

        # self.client = airsim.VehicleClient(drone_config['host_ip'])
        # self.client.simEnableWeather(True)
        self.client = None

    
    def change_drone_name(self, drone_name):
        self.drone_name = drone_name


    def initialize_client(self, client_port):
        print(f"Initializing client on port {client_port}")

        # self.client = airsim.VehicleClient(ip="10.6.36.90", port=client_port)

        self.client = airsim.VehicleClient(ip="127.0.0.1", port=client_port)  # type: ignore
        self.client.confirmConnection()
        print('set enable weather')
        self.client.simEnableWeather(True)
        print('find and destroy marker')
        time.sleep(3.5)
        self.find_and_destroy_markers()


    def set_drone_pose(self, pose):
        pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            x_val=pose.orientation.x_val,
            y_val=pose.orientation.y_val,
            z_val=pose.orientation.z_val,
            w_val=pose.orientation.w_val
        ))

        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(pose.position.x_val, pose.position.y_val, pose.position.z_val),
            airsim.to_quaternion(pitch, roll, yaw)),
            True,
            vehicle_name=self.drone_name)
        
        time.sleep(0.05)
        return self.client.simGetVehiclePose(self.drone_name)


    def reset_drone_pose(self, init_pose, init_yaw):
        x, y, z = init_pose[0], init_pose[1], init_pose[2]
        pitch, roll = 0, 0
        self.client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(x, y, -z), airsim.to_quaternion(pitch, roll, math.radians(init_yaw))),
            True,
            vehicle_name=self.drone_name)
        return self.client.simGetVehiclePose(self.drone_name)


    def get_drone_pose(self):
        return self.client.simGetVehiclePose(self.drone_name)


    def spawn_marker(self):

        self.client.simSpawnObject('Marker0',
                                   'bp_marker0',
                                   airsim.Pose(airsim.Vector3r(5, 0, 0)),
                                   airsim.Vector3r(1, 1, 1),
                                   False, True)

    def set_marker_pose(self, marker_pose, marker_name='Marker0'):
        orientation = airsim.to_quaternion(math.radians(0), math.radians(0), math.radians(0))
        x, y, z = marker_pose[0], marker_pose[1], marker_pose[2]
        position = airsim.Vector3r(x, y, -z)
        pose = airsim.Pose(position_val=position, orientation_val=orientation)
        self.client.simSetObjectPose(marker_name, pose)
        marker_pose = self.client.simGetObjectPose(marker_name)
        return marker_pose



    def check_drone_collision(self, drone_pose):
        def judge_collision(image):
            image = np.clip(image, 0, 255)
            # image = np.array(image, dtype=np.uint8)
            img_collision_result = (image / 255 < 0.004).sum() / image.flatten().shape[0]
            # print('aaaaaaaaaaa', img_collision_result)
            collision = True if img_collision_result > 0.1 else False
            return collision

        self.set_drone_pose(drone_pose)
        image = self.get_current_image(image_type='depth', external=False, angle='forward', help_cam=False)
        collide = judge_collision(image)
        return collide


    

    def get_current_image(self, image_type, external, angle, help_cam=False):

        if not help_cam:
            if external:
                if angle == 'downward':
                    camera_name = self.camera_name['external']['downward']
                elif angle == 'forward':
                    camera_name = self.camera_name['external']['forward']
            else:
                if angle == 'downward':
                    camera_name = self.camera_name['internal']['downward']
                elif angle == 'forward':
                    camera_name = self.camera_name['internal']['forward']
        else:
            camera_name = self.help_camera_name
            external = True


        def get_image(image_type, external, camera_name):

            if image_type == 'scene':
                image_type = airsim.ImageType.Scene
                responses = self.client.simGetImages([airsim.ImageRequest(camera_name, image_type, False, False)], 
                                                     vehicle_name=self.drone_name,
                                                     external=external)
                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                image = img1d.reshape(response.height, response.width, 3)

            elif image_type == 'depth':

                # if camera_name in ['seg_cam', 'downward_custom', 'forward_downward_custom', 'seg_help_cam', 'forward_check']:
                if camera_name in ['bottom_center', 'front_center']:
                    image_type = airsim.ImageType.DepthPlanar
                    responses = self.client.simGetImages([airsim.ImageRequest(camera_name, image_type, True, False)], 
                                                         vehicle_name=self.drone_name,
                                                         external=external)

                # elif camera_name in ['forward_check']:
                #     image_type = airsim.ImageType.DepthVis
                #     responses = self.client.simGetImages([airsim.ImageRequest(camera_name, image_type, True, False)], external=external)

                # elif camera_name in ['forward_downward_custom']:
                #     image_type = airsim.ImageType.DepthPerspective
                #     # responses = self.client.simGetImages([airsim.ImageRequest(camera_name, image_type, False, True)], external=external)
                #     responses = self.client.simGetImages([airsim.ImageRequest(camera_name, image_type, True, False)], external=external)

                response = responses[0]
                img1d = np.array(response.image_data_float, dtype=float)
                image = img1d.reshape(response.height, response.width, -1)

            elif image_type == 'segmentation':
                image_type = airsim.ImageType.Segmentation
                responses = self.client.simGetImages([airsim.ImageRequest(camera_name, image_type, False, False)], 
                                                     vehicle_name=self.drone_name,
                                                     external=external)
                response = responses[0]
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                image = img1d.reshape(response.height, response.width, 3)

            return image, response


        time_sleep_cnt = 0
        while time_sleep_cnt <= 20:
            try:
                # image, response = get_image(image_type, external, camera_name)
                image, _ = get_image(image_type, external, camera_name)
                break

            except:
                time_sleep_cnt += 1
                image = None
                print('Error when getting image {} from camera {} with angle {}, time_sleep_cnt: {}'.format(image_type, camera_name, angle, time_sleep_cnt))
                time.sleep(3)


        # image, response = get_image(image_type, external, camera_name)
        # if response.width == 0 or response.height == 0:
        #     if image_type in ['scene', 'semantic']:
        #         image = np.zeros((480, 640, 3), dtype=int)
        #     else:
        #         image = 100 * np.ones((480, 640, 1), dtype=float)

        if image is None:
            print('Use default image, due to incorrect response from Airsim Map Rendering')
            if image_type in ['scene', 'segmentation']:
                image = np.zeros((480, 640, 3), dtype=np.uint8)  # 全黑RGB图

            elif image_type == 'depth':
                image = 100.0 * np.ones((480, 640, 1), dtype=np.float32)  # 深度100米

        return image



    def set_weather(self, params):
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, min(1, params[0]))
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Roadwetness, min(1, params[1]))
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Snow, min(1, params[2]))
        self.client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow, min(1, params[3]))
        self.client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, min(1, params[4]))
        self.client.simSetWeatherParameter(airsim.WeatherParameter.RoadLeaf, min(1, params[5]))
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Dust, min(1, params[6]))
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, min(1, params[7]))
        # if self.sim_mode != 'cv':
        #     wind = airsim.Vector3r(min(5, params[8 ] *5), min(5, params[8 ] *5), 0)
        #     self.client.simSetWind(wind)

    def set_time_of_day(self, params):
        hour, minute = params[0], params[1]
        hour_time = int(hour * 24)
        min_time = int(minute * 60)
        sim_time = '2023-09-06 {}:{}:00'.format(hour_time, min_time)
        self.client.simSetTimeOfDay(True, start_datetime=sim_time, is_start_datetime_dst=True, celestial_clock_speed=1, update_interval_secs=1, move_sun=True)

    def find_and_destroy_markers(self):
        # 获取当前所有对象的名称
        object_names = self.client.simListSceneObjects()
        marker_objects = [name for name in object_names if 'marker' in name.lower()]
        # print(object_names)
        if len(marker_objects) == 0:
            print('spawn marker')
            self.spawn_marker()  ## 添加marker

        # # 删除所有找到的 marker 对象
        # for marker in marker_objects:
        #     self.client.simDestroyObject(marker)
        #     print(f"Destroyed object: {marker}")




# drone = Drone_tool(drone_config)
# client = drone.client


if __name__ == '__main__':

    drone = Drone_tool(drone_config)
    
    drone.initialize_client(41451)

    # img_up, img_down = drone.get_current_extra_depth_img()
    # print(img_up.shape, img_down.shape)

    img = drone.get_current_image(image_type='scene', external=False, angle='forward', help_cam=False)
    print(img.shape)

    # response = drone.get_multi_image()
    # print(type(response))



    # img = drone.get_current_image(image_type='depth', external=False, angle='downward', help_cam=False)
    # print(img.shape, img.max(), img.min())


    drone.reset_drone_pose(init_pose=[0,0,15], init_yaw=0)

    



