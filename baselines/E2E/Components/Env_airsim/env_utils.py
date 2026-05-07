import airsim
import numpy as np
import math
import time



def interpret_action(action, current_position, current_rotation, step_size=1, step_size_z=1, turn_angle=90):
    '''
    param:
        action index: action index (int, 1 forward, 2 turn left, 3 turn right, 4 ascend, 5 descend, others: stop or hover)
        current_position: [x, y, z], z is negative in airsim
        current_rotation: [x, y, z, w]
        step_size: step size
        step_size_z: step size in z direction
        turn_angle: turn angle  

    return:
        new_pose: airsim.Pose

    '''
    current_position = np.array(current_position) 
    current_rotation = np.array(current_rotation)

    pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
        x_val=current_rotation[0],
        y_val=current_rotation[1],
        z_val=current_rotation[2],
        w_val=current_rotation[3]
    ))
    pitch = 0
    roll = 0

    # forward
    if action == 1:
        unit_x = 1 * math.cos(pitch) * math.cos(yaw)
        unit_y = 1 * math.cos(pitch) * math.sin(yaw)
        unit_z = 1 * math.sin(pitch) * (-1)
        unit_vector = np.array([unit_x, unit_y, unit_z])
        assert unit_z == 0

        new_position = np.array(current_position) + unit_vector * step_size
        new_rotation = current_rotation.copy()

    # turn left
    elif action == 2:
        new_position = current_position.copy()

        new_pitch = pitch
        new_roll = roll
        new_yaw = yaw - math.radians(turn_angle)
        if float(new_yaw * 180 / math.pi) < -180:
            new_yaw = math.radians(360) + new_yaw
        new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
        new_rotation = [
                new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
            ]

    # turn right
    elif action == 3:
        new_position = current_position.copy()

        new_pitch = pitch
        new_roll = roll
        new_yaw = yaw + math.radians(turn_angle)
        if float(new_yaw * 180 / math.pi) > 180:    
            new_yaw = math.radians(-360) + new_yaw
        new_rotation = airsim.to_quaternion(new_pitch, new_roll, new_yaw)
        new_rotation = [
                new_rotation.x_val, new_rotation.y_val, new_rotation.z_val, new_rotation.w_val
            ]
    
    # ascend
    elif action == 4:
        unit_vector = np.array([0, 0, -1])
        new_position = np.array(current_position) + unit_vector * step_size_z
        new_rotation = current_rotation.copy()

    # descend
    elif action == 5:
        unit_vector = np.array([0, 0, 1])
        new_position = np.array(current_position) + unit_vector * step_size_z
        new_rotation = current_rotation.copy()

    # stop or hover
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



def update_path_length(action, current_path_length, step_size=1, step_size_z=1):
    if action == 1: # forward
        current_path_length += step_size
    
    elif action == 4 or action == 5: # ascend or descend
        current_path_length += step_size_z

    return current_path_length



def update_distance_drone2marker(current_position, marker_position):
    '''
    param:
        current_position: list, [x, y, z]
        marker_position: list, [x, y, z]

    return:
        distance: float
    '''
    
    distance = np.linalg.norm(np.array(current_position) - np.array(marker_position))
    return distance



def update_success(distance_drone2marker, success_distance=2):
    success = False
    if distance_drone2marker < success_distance:
        success = True
    return success



def get_current_pose(drone_tool):
    agent_state = drone_tool.get_drone_pose()
    x = agent_state.position.x_val   ## x left positive
    y = agent_state.position.y_val 
    z = agent_state.position.z_val
    return [x,y,z]

def get_current_rotation(drone_tool):
    agent_state = drone_tool.get_drone_pose()
    rotation = [agent_state.orientation.x_val, 
                agent_state.orientation.y_val, 
                agent_state.orientation.z_val, 
                agent_state.orientation.w_val]
    return rotation


def get_agent_pose(drone_tool):
    agent_state = drone_tool.get_drone_pose()
    x = agent_state.position.x_val   ## x left positive
    y = agent_state.position.y_val 
    return [x,y]
    
def get_agent_height(drone_tool):
    agent_state = drone_tool.get_drone_pose()
    z = -agent_state.position.z_val
    return z
    
def get_forward_img_rgb(drone_tool):
    rgb_image_forward = drone_tool.get_current_image(image_type='scene', external=False, angle='forward')
    img = np.clip(rgb_image_forward, 0, 255)
    return img
    
def get_forward_img_depth(drone_tool):
    depth_image_forward = drone_tool.get_current_image(image_type='depth', external=False, angle='forward')
    img = np.clip(depth_image_forward, 0, 255)
    return img
    
def get_downward_img_rgb(drone_tool):
    rgb_image_downward = drone_tool.get_current_image(image_type='scene', external=False, angle='downward')
    img = np.clip(rgb_image_downward, 0, 255)
    return img

def get_downward_img_depth(drone_tool):
    depth_image_downward = drone_tool.get_current_image(image_type='depth', external=False, angle='downward')
    img = np.clip(depth_image_downward, 0, 255)
    return img



def setup_episode(episode_info, drone_tool):
    # marker_pose = episode_info['marker_pose']
    drone_pose = episode_info['drone_pose']
    # episode_time = episode_info['time']
    # weather = episode_info['weather']

    # drone_tool.set_marker_pose(marker_pose)
    drone_tool.reset_drone_pose(init_pose=drone_pose[0:-1], init_yaw=drone_pose[-1])
    # drone_tool.set_time_of_day(episode_time)
    # drone_tool.set_weather(weather)

    time.sleep(1)





    
    

    
    
    





