import airsim
import math
import numpy as np
import time
import random

from Env.read_data import load_one_map_data
from Env.client import Drone_tool, drone_config




import sys
import os

# Go up two levels: from Env/ → Liang/ → baselines/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from Obj_Detect.yolo11_detector import detector






class Test_Env():
    def __init__(self, max_num_step, step_size=2, turn_angle=30, action_mapping_Simenv2Airsimenv=None, det_conf_thres=0.7):
        self.num_step = 0
        self.max_num_step = max_num_step

        self.step_size = step_size
        self.turn_angle = turn_angle
        self.dis_success = 2

        self.drone_tool = Drone_tool(drone_config)
        self.drone_tool.initialize_client(client_port=41451)

        self.det_model = detector
        self.det_conf_thres = det_conf_thres

        self.action_mapping_Simenv2Airsimenv = action_mapping_Simenv2Airsimenv

        # self.agent = 


    def reset(self, episode_info):
        self.episode_info = episode_info

        shortest_path_length = self._setup_episode(episode_info)

        self.shortest_path_length = shortest_path_length

        self.num_step = 0
        self.success = 0
        self.agent_success = 0
        self.path_length = 0
        self.det_marker_pose_g = None

        self.time_start = time.time()



    def _setup_episode(self, episode_info):

        marker_pose = episode_info['marker_pose']
        self.drone_tool.set_marker_pose(episode_info['marker_pose'])
        drone_pose = episode_info['drone_pose']
        self.drone_tool.reset_drone_pose(init_pose=drone_pose[0:-1], init_yaw=drone_pose[-1])

        self.drone_tool.set_time_of_day(episode_info['time'])
        self.drone_tool.set_weather(episode_info['weather'])

        time.sleep(1)

        shortest_path = np.linalg.norm(
            np.array(marker_pose[:2]) - np.array(drone_pose[:2])
        )

        return shortest_path


    
    def step(self, action):

        infos = {}

        print('1111111111111111', action)

        if self.action_mapping_Simenv2Airsimenv is not None:
            action = self.action_mapping_Simenv2Airsimenv[action]
        
        print('22222222222222222222', action)

        # do_action
        new_pose = self.interpret_action(action)
        self.drone_tool.set_drone_pose(new_pose)

        if action == 1:
            self.path_length += self.step_size
        
        self.num_step += 1

        self.collision = self.check_collision()
        find_marker, _ = self._check_done()


        if find_marker:
            rgb_image = self._get_img()
            depth_image = self._get_depth_img()
            drone_pose = self.drone_tool.get_drone_pose()
            _, det_marker_pose_g, _ = self.det_model.detect_marker(rgb_image, depth_image, drone_pose, conf_thres=self.det_conf_thres)
            
            self.det_marker_pose_g = det_marker_pose_g


        done = find_marker or self.collision
        metrics = self.update_metrics()
        metrics['find_marker'] = True if find_marker else False
        print(metrics)
        
        return done, metrics



    def _get_agent_pose(self):
        agent_state = self.drone_tool.get_drone_pose()
        x = agent_state.position.x_val   ## x left positive
        y = agent_state.position.y_val 
        return [x,y]
    
    def get_agent_height(self):
        agent_state = self.drone_tool.get_drone_pose()
        z = -agent_state.position.z_val
        return z
    
    def get_forward_img_rgb(self):
        rgb_image_forward = self.drone_tool.get_current_image(image_type='scene', external=False, angle='forward')
        img = np.clip(rgb_image_forward, 0, 255)
        return img
    
    def get_forward_img_depth(self):
        depth_image_forward = self.drone_tool.get_current_image(image_type='depth', external=False, angle='forward')
        img = np.clip(depth_image_forward, 0, 255)
        return img

    
    def _get_img(self):
        rgb_image_downward = self.drone_tool.get_current_image(image_type='scene', external=False, angle='downward')
        img = np.clip(rgb_image_downward, 0, 255)
        # img = np.array(img, dtype=np.uint8)
        return img
    

    def _get_depth_img(self):
        depth_image_downward = self.drone_tool.get_current_image(image_type='depth', external=False, angle='downward')
        img = np.clip(depth_image_downward, 0, 255)
        return img
    
    def judge_collision(self, depth_image):
        '''depth_image: shape (H, W) or (1, H, W)  '''

        if depth_image.ndim == 3:
            depth_image = depth_image[0]  # shape: (H, W)
        
        # 计算距离小于阈值的像素数量
        # depth_image = depth_image / 255.0
        depth_image = np.clip(depth_image, 0, 255)

        close_pixels = depth_image < 1    # 1 meter
        close_pixel_count = np.sum(close_pixels)
        total_pixels = depth_image.size
        
        # 计算比例
        close_ratio = close_pixel_count / total_pixels
        
        # 如果超过阈值比例的像素距离过近，则判定为碰撞
        return close_ratio > 0.1
    

    def check_collision(self):
        # done due to collision
        depth_forward = self.get_forward_img_depth()
        depth_downward = self._get_depth_img()
        collision_forward = self.judge_collision(depth_forward)
        collision_downward = self.judge_collision(depth_downward)

        if collision_forward or collision_downward:
            print('Collision detected!')
            return True
        else:
            return False


    def _check_done(self):
        # done due to finding the marker
        rgb_down = self._get_img()

        detect_result_post, detect_result_conf = self.det_model.only_detect_marker_bbox(rgb_down, conf_thres=self.det_conf_thres)

        if detect_result_conf >= self.det_conf_thres:
            return True, detect_result_post
        else:
            return False, None
    

    def _update_success(self):

        dis_det2marker = 50

        if self.det_marker_pose_g is not None:
            marker_pose = self.episode_info['marker_pose']

            dis_det2marker = np.linalg.norm(
                np.array(self.det_marker_pose_g) - np.array(marker_pose)
            )

            if dis_det2marker < self.dis_success:
                self.success = 1
            
        return dis_det2marker

        
        

    
    def _update_distance_drone2marker(self):
        drone_pose = self._get_agent_pose()
        marker_pose = self.episode_info['marker_pose']

        # print('pose in airsim: ', drone_pose, marker_pose)

        dis_drone2marker = np.linalg.norm(
            np.array(marker_pose[:2]) - np.array(drone_pose[:2])
        )

        return dis_drone2marker

    
    
    def update_metrics(self):
        dis_det2marker = self._update_success()
        # distance_drone2marker = self._update_distance_drone2marker()

        time_consumed = round(time.time() - self.time_start, 2)

        metrics = {
            'success': self.success,
            'path_length': self.path_length,
            'time_consumed': time_consumed,
            'num_steps': self.num_step,
            'distance_drone2marker': dis_det2marker,
            # 'collision': self.collision,
        }
        
        return metrics


    def interpret_action(self, action):

        drone_pose = self.drone_tool.get_drone_pose()
        current_position = np.array([drone_pose.position.x_val, drone_pose.position.y_val, drone_pose.position.z_val])
        current_rotation = np.array([
            drone_pose.orientation.x_val, drone_pose.orientation.y_val, drone_pose.orientation.z_val, drone_pose.orientation.w_val
        ])

        # if action == AirsimActions.MOVE_FORWARD:
        if action == 1:
            print('action: forward')

            pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
                x_val=current_rotation[0],
                y_val=current_rotation[1],
                z_val=current_rotation[2],
                w_val=current_rotation[3]
            ))
            pitch = 0
            roll = 0

            unit_x = 1 * math.cos(pitch) * math.cos(yaw)
            unit_y = 1 * math.cos(pitch) * math.sin(yaw)
            unit_z = 1 * math.sin(pitch) * (-1)
            unit_vector = np.array([unit_x, unit_y, unit_z])
            assert unit_z == 0

            # new_position = np.array(current_position) + unit_vector * AirsimActionSettings.FORWARD_STEP_SIZE
            new_position = np.array(current_position) + unit_vector * self.step_size
            new_rotation = current_rotation.copy()


        # elif action == AirsimActions.TURN_LEFT:
        elif action == 2:
            print('action: turn left')

            pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
                x_val=current_rotation[0],
                y_val=current_rotation[1],
                z_val=current_rotation[2],
                w_val=current_rotation[3]
            ))
            pitch = 0
            roll = 0

            new_pitch = pitch
            new_roll = roll
            new_yaw = yaw - math.radians(self.turn_angle)
            if float(new_yaw * 180 / math.pi) < -180:
                new_yaw = math.radians(360) + new_yaw

            new_position = current_position.copy()
            new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
            new_rotation = [
                new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
            ]


        # elif action == AirsimActions.TURN_RIGHT:
        elif action == 3:
            print('action: turn right')

            pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
                x_val=current_rotation[0],
                y_val=current_rotation[1],
                z_val=current_rotation[2],
                w_val=current_rotation[3]
            ))
            pitch = 0
            roll = 0

            new_pitch = pitch
            new_roll = roll
            new_yaw = yaw + math.radians(self.turn_angle)
            if float(new_yaw * 180 / math.pi) > 180:
                new_yaw = math.radians(-360) + new_yaw

            new_position = current_position.copy()
            new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
            new_rotation = [
                new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
            ]


        # elif action == AirsimActions.GO_UP:
        elif action == 4:
            print('action: ascend')

            # pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            #     x_val=current_rotation[0],
            #     y_val=current_rotation[1],
            #     z_val=current_rotation[2],
            #     w_val=current_rotation[3]
            # ))
            # pitch = 0
            # roll = 0

            unit_vector = np.array([0, 0, -1])

            # new_position = np.array(current_position) + unit_vector * AirsimActionSettings.UP_DOWN_STEP_SIZE
            new_position = np.array(current_position) + unit_vector * self.step_size_z
            new_rotation = current_rotation.copy()


        # elif action == AirsimActions.GO_DOWN:
        elif action == 5:
            print('action: descend')
            # pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
            #     x_val=current_rotation[0],
            #     y_val=current_rotation[1],
            #     z_val=current_rotation[2],
            #     w_val=current_rotation[3]
            # ))
            # pitch = 0
            # roll = 0

            unit_vector = np.array([0, 0, -1])

            # new_position = np.array(current_position) + unit_vector * AirsimActionSettings.UP_DOWN_STEP_SIZE * (-1)
            new_position = np.array(current_position) + unit_vector * self.step_size_z * (-1)
            new_rotation = current_rotation.copy()

        
        
        else:
            new_position = current_position.copy()
            new_rotation = current_rotation.copy()


        new_pose = airsim.Pose(
            position_val=airsim.Vector3r(
                x_val=new_position[0],
                y_val=new_position[1],
                z_val=new_position[2]
            ),
            orientation_val=airsim.Quaternionr(
                x_val=new_rotation[0],
                y_val=new_rotation[1],
                z_val=new_rotation[2],
                w_val=new_rotation[3]
            )
        )
        return new_pose
        

    
    def get_action_from_agent(self):
        action = random.choice([1, 2, 3])
        return action
    
    
    def test(self, episode_info):
        self.reset(episode_info)

        while self.num_step < self.max_num_step:
            self.num_step += 1
            
            action = self.get_action_from_agent()
            done = self.step(action)

            if done:
                self.agent_success = 1
                break
        
        self.time_consumed = round(time.time() - self.time_start, 2)

        return self.num_step, self.success, self.agent_success, self.path_length, self.time_consumed




if __name__ == "__main__":

    dataset_dir = '/home/mqdronelabserver/Documents/Yjh/baselines/Test/data/episode_drone'
    mode = 'test'

    test_env = Test_Env(max_num_step=400)


    map_names = 'Urbandistract'
    data = load_one_map_data(dataset_dir, map_names, mode)
    print(len(data), data[0])



    metrics = test_env.test(episode_info = data[0])
    print(metrics)