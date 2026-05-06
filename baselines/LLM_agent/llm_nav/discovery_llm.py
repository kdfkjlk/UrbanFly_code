import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)




import airsim as airsim
import numpy as np
from airsim.scenario_generation.scenario_manager import ScenarioManager
import cv2
import json
import base64
from openai import OpenAI

from configs.api_config import *


from io import BytesIO
from PIL import Image
import time
import math
import cairosvg
import os
from scipy.spatial.transform import Rotation as R
from load_env import load_scenario, analyze_image_regions

from prompt_doc import (
    PROMPT_TEMPLATE_8region,
    PROMPT_TEMPLATE_5region,
    PROMPT_TEMPLATE_3action,
    PROMPT_TEMPLATE_5action,
    PROMPT_TEMPLATE_5region_3D
)


# Initialize AirSim client and scenario manager
client = airsim.VehicleClient()
client.confirmConnection()
sm = ScenarioManager(client)
search_space = [60, 60, 0]




openai_client = OpenAI(
    api_key=api_key
)

# Function to encode image to base64 for API
def encode_image(image_array):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    # Save to BytesIO object
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    # Get the byte data and encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def encode_svg_marker(svg_path):
    # Create a temporary PNG file
    temp_png = "temp_marker.png"
    
    # Convert SVG to PNG using cairosvg
    cairosvg.svg2png(url=svg_path, write_to=temp_png)
    
    # Read the PNG and encode to base64
    with open(temp_png, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Clean up the temporary file
    if os.path.exists(temp_png):
        os.remove(temp_png)
        
    return encoded_string





def move_drone_to_region(new_position, new_orientation):

    # Set the new pose
    client.simSetVehiclePose(
        airsim.Pose(
            airsim.Vector3r(
                new_position[0], 
                new_position[1], 
                new_position[2]
            ), 
            new_orientation
        ), 
        ignore_collision=True
    )


def get_position_orientation_list(client_pose):
    pose = {
        'position_x': client_pose.position.x_val,
        'position_y': client_pose.position.y_val,
        'position_z': client_pose.position.z_val,
        'orientation_x': client_pose.orientation.x_val,
        'orientation_y': client_pose.orientation.y_val,
        'orientation_z': client_pose.orientation.z_val,
        'orientation_w': client_pose.orientation.w_val,
    }
    return pose    



def compute_target_orientation(current_pos, current_ori, target_pos):
    """
    计算agent从当前位置朝向目标位置所需的yaw角，并返回目标位置的orientation（四元数）

    Args:
        current_pos (airsim.Vector3r): 当前坐标
        current_ori (airsim.Quaternionr): 当前四元数方向
        target_pos (airsim.Vector3r): 目标坐标

    Returns:
        new_orientation (airsim.Quaternionr): 面向目标位置的 orientation
    """

    # 1. 先获取当前位置和目标位置的x, y坐标
    dx = target_pos[0] - current_pos.x_val
    dy = target_pos[1] - current_pos.y_val

    # 2. 当前 orientation 转为 pitch, roll, yaw
    r = R.from_quat([current_ori.x_val, current_ori.y_val, current_ori.z_val, current_ori.w_val])
    pitch, roll, current_yaw = r.as_euler('xyz', degrees=False)  # 注意airsim是用'xyz'顺序

    # 3. 计算目标方向的 yaw（在xy平面上，atan2）
    target_yaw = np.arctan2(dy, dx)

    # 4. 构造新方向（保持 pitch 和 roll 不变，只改变 yaw）
    new_r = R.from_euler('xyz', [pitch, roll, target_yaw], degrees=False)
    new_quat = new_r.as_quat()

    # 5. 返回新的 orientation 四元数
    return airsim.Quaternionr(new_quat[0], new_quat[1], new_quat[2], new_quat[3])


def interprete_action(current_pose, result, region_centers, move_format, step_size=2):

    if move_format == '8patch':  # 8 boxes
        target_region_id = int(result["next_region"])-1

        new_pose_x, new_pose_y = region_centers[target_region_id][0:2]
        new_pose_z = current_pose.position.z_val
        
        new_position = [new_pose_x, new_pose_y, new_pose_z]
        new_orientation = current_pose.orientation

    
    elif move_format == '5patch': 
        target_region_id = int(result["next_region"])-1

        new_pose_x, new_pose_y = region_centers[target_region_id][0:2]
        new_pose_z = current_pose.position.z_val
        
        new_position = [new_pose_x, new_pose_y, new_pose_z]

        new_orientation = compute_target_orientation(current_pose.position, current_pose.orientation, new_position)
        new_orientation = new_orientation


    elif move_format == '3action':  # forward, turn left/right
        target_region_id = int(result["next_action"])

        curr_pitch, curr_roll, curr_yaw = airsim.to_eularian_angles(current_pose.orientation)
        # set default value, to prevent invalid target_region_id output from LLM
        yaw_delta = 0
        new_position = [
            current_pose.position.x_val,
            current_pose.position.y_val,
            current_pose.position.z_val
        ]

        if target_region_id == 1:
            # yaw_delta = 0
            curr_pitch = 0
            curr_roll = 0

            unit_x = 1 * math.cos(curr_pitch) * math.cos(curr_yaw)
            unit_y = 1 * math.cos(curr_pitch) * math.sin(curr_yaw)
            unit_z = 1 * math.sin(curr_pitch) * (-1)
            assert unit_z == 0

            new_position = [current_pose.position.x_val + unit_x * step_size, 
                            current_pose.position.y_val + unit_y * step_size, 
                            current_pose.position.z_val + unit_z * step_size]

        elif target_region_id == 2:
            yaw_delta = math.radians(90) * (-1)
        
        elif target_region_id == 4:
            yaw_delta = math.radians(90)


        new_yaw = curr_yaw + yaw_delta
        new_orientation = airsim.to_quaternion(curr_pitch, curr_roll, new_yaw)

    
    elif move_format == '5action':  # forward, turn left/right
        target_region_id = int(result["next_action"])

        curr_pitch, curr_roll, curr_yaw = airsim.to_eularian_angles(current_pose.orientation)
        # set default value, to prevent invalid target_region_id output from LLM
        yaw_delta = 0
        new_position = [
            current_pose.position.x_val,
            current_pose.position.y_val,
            current_pose.position.z_val
        ]

        if target_region_id == 1:    # turn left 45 degree + forward step
            yaw_delta = math.radians(45) * (-1)
            new_yaw = curr_yaw + yaw_delta

            curr_pitch = 0
            curr_roll = 0

            unit_x = 1 * math.cos(curr_pitch) * math.cos(curr_yaw)
            unit_y = 1 * math.cos(curr_pitch) * math.sin(curr_yaw)
            unit_z = 1 * math.sin(curr_pitch) * (-1)
            assert unit_z == 0

            new_position = [current_pose.position.x_val + unit_x * step_size, 
                            current_pose.position.y_val + unit_y * step_size, 
                            current_pose.position.z_val + unit_z * step_size]
            new_orientation = airsim.to_quaternion(curr_pitch, curr_roll, new_yaw)

        
        elif target_region_id == 2:  # turn left 45 degree + forward step
            # yaw_delta = 0
            curr_pitch = 0
            curr_roll = 0

            unit_x = 1 * math.cos(curr_pitch) * math.cos(curr_yaw)
            unit_y = 1 * math.cos(curr_pitch) * math.sin(curr_yaw)
            unit_z = 1 * math.sin(curr_pitch) * (-1)
            assert unit_z == 0

            new_position = [current_pose.position.x_val + unit_x * step_size, 
                            current_pose.position.y_val + unit_y * step_size, 
                            current_pose.position.z_val + unit_z * step_size]
            new_yaw = curr_yaw + yaw_delta
            new_orientation = airsim.to_quaternion(curr_pitch, curr_roll, new_yaw)

        
        elif target_region_id == 3:   # turn right 45 degree + forward step
            yaw_delta = math.radians(45)
            new_yaw = curr_yaw + yaw_delta

            curr_pitch = 0
            curr_roll = 0

            unit_x = 1 * math.cos(curr_pitch) * math.cos(curr_yaw)
            unit_y = 1 * math.cos(curr_pitch) * math.sin(curr_yaw)
            unit_z = 1 * math.sin(curr_pitch) * (-1)
            assert unit_z == 0

            new_position = [current_pose.position.x_val + unit_x * step_size, 
                            current_pose.position.y_val + unit_y * step_size, 
                            current_pose.position.z_val + unit_z * step_size]
            new_orientation = airsim.to_quaternion(curr_pitch, curr_roll, new_yaw)


        elif target_region_id == 4:  ## turn left
            yaw_delta = math.radians(90) * (-1)

            new_yaw = curr_yaw + yaw_delta
            new_orientation = airsim.to_quaternion(curr_pitch, curr_roll, new_yaw)
        
        elif target_region_id == 6:  ## turn right
            yaw_delta = math.radians(90)

            new_yaw = curr_yaw + yaw_delta
            new_orientation = airsim.to_quaternion(curr_pitch, curr_roll, new_yaw)

    
    elif move_format == '5patch_3D': 

        if int(result["next_region"]) in [1,2,3,4,5,6]:
            target_region_id = int(result["next_region"])-1

            new_pose_x, new_pose_y = region_centers[target_region_id][0:2]
            new_pose_z = current_pose.position.z_val
        
            new_position = [new_pose_x, new_pose_y, new_pose_z]

            new_orientation = compute_target_orientation(current_pose.position, current_pose.orientation, new_position)
            new_orientation = new_orientation
        

        elif int(result["next_region"]) in [7,8]:
            target_region_id = int(result["next_region"])

            curr_pitch, curr_roll, curr_yaw = airsim.to_eularian_angles(current_pose.orientation)
            # set default value, to prevent invalid target_region_id output from LLM
            yaw_delta = 0


            if target_region_id == 7:  ## ascend
                new_position = [current_pose.position.x_val, 
                            current_pose.position.y_val, 
                            current_pose.position.z_val + (-1) * step_size]
        
            elif target_region_id == 8:  ## descend
                new_position = [current_pose.position.x_val, 
                            current_pose.position.y_val, 
                            current_pose.position.z_val + 1 * step_size]


            new_yaw = curr_yaw + yaw_delta
            new_orientation = airsim.to_quaternion(curr_pitch, curr_roll, new_yaw)


    return new_position, new_orientation



def judge_collision(depth_image):
        '''depth_image: shape (H, W) or (1, H, W)  '''

        if depth_image.ndim == 3:
            depth_image = depth_image[0]  # shape: (H, W)
        
        # 计算距离小于阈值的像素数量
        depth_image = np.clip(depth_image, 0, 255.0)
        # depth_image = depth_image / 255.0

        close_pixels = depth_image < 1    # 1 meter
        close_pixel_count = np.sum(close_pixels)
        total_pixels = depth_image.size
        
        # 计算比例
        close_ratio = close_pixel_count / total_pixels
        
        # 如果超过阈值比例的像素距离过近，则判定为碰撞
        return close_ratio > 0.1


def main(move_format, save_dir, map_name, step_size, data_split='test'):
    data_path = os.path.join(PROJECT_ROOT, 'data_episode_drone', data_split, map_name, f'{data_split}.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    # Path to the SVG marker file
    svg_marker_path = os.path.join(PROJECT_ROOT, 'llm_nav', 'marker.svg')
    
    # Encode the SVG marker once
    encoded_marker = encode_svg_marker(svg_marker_path)


    for i in range(0, len(data)):
        load_scenario(client, data[i])
        os.makedirs(f"{save_dir}/{i}", exist_ok=True)

    
        # Initialize movement history
        movement_history = []
        movement_reason_history = []
        explore_history = []
        trajectory = []
        found_marker = False
        max_steps = 100  # Maximum number of steps to prevent infinite loops
    
        for step in range(max_steps):
            # Get current image from downward camera
            image_downward, camera_pose = sm.get_current_scene(camera_name="bottom_center", image_type=0, external=False)
            current_pose = client.simGetVehiclePose()  ## radians

            client_pose = get_position_orientation_list(current_pose)
            trajectory.append(client_pose)
        
            # Display the current view
            # cv2.imshow("Drone View", image)
            # cv2.waitKey(1)
            # save image
            # cv2.imwrite(f"scenarios/{i}/image_{step+1}.jpg", image)
            cv2.imwrite(f"{save_dir}/{i}/image_down_{step+1}.jpg", image_downward)

        
            # Encode current camera image 
            encoded_image_downward = encode_image(image_downward)


            ## forwrd RGB image ------------------------------------------------------------------
            image_forward, camera_pose = sm.get_current_scene(camera_name="front_center", image_type=0, external=False)
            cv2.imwrite(f"{save_dir}/{i}/image_forward_{step+1}.jpg", image_forward)
            encoded_image_forward = encode_image(image_forward)
            ## forwrd RGB image ------------------------------------------------------------------

        
            # Create prompt with movement history
            # prompt = PROMPT_TEMPLATE.format(movement_history=movement_history, movement_reason_history=movement_reason_history)

            if move_format == '8patch':
                prompt = PROMPT_TEMPLATE_8region.format(
                    movement_history=movement_history, 
                    movement_reason_history=movement_reason_history
                )
            
            elif move_format == '5patch':
                prompt = PROMPT_TEMPLATE_5region.format(
                    movement_history=movement_history, 
                    movement_reason_history=movement_reason_history
                )

            elif move_format == '3action':
                prompt = PROMPT_TEMPLATE_3action.format(
                    step_size = step_size,
                    movement_history=movement_history, 
                    movement_reason_history=movement_reason_history
                )
            
            elif move_format == '5action':
                prompt = PROMPT_TEMPLATE_5action.format(
                    step_size = step_size,
                    movement_history=movement_history, 
                    movement_reason_history=movement_reason_history
                )
            
            elif move_format == "5patch_3D":
                prompt = PROMPT_TEMPLATE_5region_3D.format(
                    step_size = step_size,
                    movement_history=movement_history, 
                    movement_reason_history=movement_reason_history
                )
            

            
            try:
            
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{encoded_marker}"
                                    }
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_image_downward}"
                                    }
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_image_forward}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_completion_tokens=512,
                    timeout=60
                )
                

            
                # Parse the response
                # try:
                response_text = response.choices[0].message.content
                # Extract JSON from response (in case there's additional text)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                # Log the result
                print(f"Step {step+1}: {result}")
                
                # Update movement history
                if move_format in ['8patch', '5patch', '5patch_3D']:
                    movement_history.append(f"Region {result['next_region']}")
                    
                elif move_format in ['3action', '5action']:
                    movement_history.append(f"Region {result['next_action']}")
            


                movement_reason_history.append(result['reasoning'])
                explore_history.append(result)
                # Check if marker is found
                if result["found_marker"]:
                    found_marker = True
                    print(f"Marker found! Reasoning: {result['reasoning']}")
                    print(f"Confidence: {result.get('confidence', 'Not specified')}")
                    # json.dump(explore_history, open(f"scenarios/{i}/explore.json", "w"))
                    json.dump(explore_history, open(f"{save_dir}/{i}/explore.json", "w"))
                    json.dump(trajectory, open(f"{save_dir}/{i}/trajectory.json", "w"))
                    json.dump('no_collision', open(f"{save_dir}/{i}/collision.json", "w"))

                    # image_downward, camera_pose = sm.get_current_scene(camera_name="bottom_center", image_type=0, external=False)
                    # cv2.imwrite(f"scenarios/{i}/image_success.jpg", image)
                    # cv2.imwrite(f"{save_dir}/{i}/image_success.jpg", image)
                    break
                    
                ## depth image
                # depth_image_forward, _ = sm.get_current_scene(camera_name='0', image_type=1, image_encoding='rgb', external=False)
                # depth_image_forward, _ = sm.get_current_scene(camera_name='front_center', image_type=1, image_encoding='rgb', external=False)

                # collision = judge_collision(depth_image_forward)
                # if collision:
                #     print('collision')
                #     json.dump(explore_history, open(f"{save_dir}/{i}/explore.json", "w"))
                #     json.dump(trajectory, open(f"{save_dir}/{i}/trajectory.json", "w"))
                #     json.dump('has_collision', open(f"{save_dir}/{i}/collision.json", "w"))
                #     break

                
                if move_format in ['5patch_3D']:
                    depth_image_downward, _ = sm.get_current_scene(camera_name='bottom_center', image_type=1, image_encoding='rgb', external=False)
                    collision_downward = judge_collision(depth_image_downward)

                    depth_image_forward, _ = sm.get_current_scene(camera_name='front_center', image_type=1, image_encoding='rgb', external=False)
                    collision_forward = judge_collision(depth_image_forward)

                    collision = collision_downward or collision_forward

                    if collision:
                        print('collision')
                        json.dump(explore_history, open(f"{save_dir}/{i}/explore.json", "w"))
                        json.dump(trajectory, open(f"{save_dir}/{i}/trajectory.json", "w"))
                        json.dump('has_collision', open(f"{save_dir}/{i}/collision.json", "w"))
                        break
                
                else:
                    depth_image_forward, _ = sm.get_current_scene(camera_name='front_center', image_type=1, image_encoding='rgb', external=False)

                    collision = judge_collision(depth_image_forward)
                    if collision:
                        print('collision')
                        json.dump(explore_history, open(f"{save_dir}/{i}/explore.json", "w"))
                        json.dump(trajectory, open(f"{save_dir}/{i}/trajectory.json", "w"))
                        json.dump('has_collision', open(f"{save_dir}/{i}/collision.json", "w"))
                        break



                # Move the drone according to the selected region
                _, _, region_centers, _ = analyze_image_regions(image_downward, camera_pose, current_pose)
                # move_drone_to_region(region_centers[int(result["next_region"])-1], current_pose)

                # new_position, new_orientation = interprete_action(current_pose, result, region_centers, move_format)
                # move_drone_to_region(new_position, new_orientation)

                if move_format in ['8patch', '5patch', '5patch_3D']:
                    _, _, region_centers, _ = analyze_image_regions(image_downward, camera_pose, current_pose)
                    new_position, new_orientation = interprete_action(current_pose, result, region_centers, move_format)
                
                elif move_format in ['3action', '5action']:
                    new_position, new_orientation = interprete_action(current_pose, result, None, move_format, step_size=step_size)

                
                move_drone_to_region(new_position, new_orientation)

                # current_pose = client.simGetVehiclePose()
                # client_pose = get_position_orientation_list(current_pose)
                # trajectory.append(client_pose)
            

                if step >= max_steps-1:
                    json.dump(explore_history, open(f"{save_dir}/{i}/explore.json", "w"))
                    json.dump(trajectory, open(f"{save_dir}/{i}/trajectory.json", "w"))
                    json.dump('no_collision', open(f"{save_dir}/{i}/collision.json", "w"))


                
            # except Exception as e:
            #     print(f"Error processing LLM response: {e}")
            #     print(f"Raw response: {response.choices[0].message.content}")
            #     continue


            except Exception as e:
                print("=" * 80)
                print("Error type:", type(e).__name__)
                print("Error message:", repr(e))
                if "response" in locals() and response is not None:
                    print("Raw response:", response)
                print("=" * 80)
                continue

            # time.sleep(1)
    
    # Final result
    if found_marker:
        print("Mission successful! Marker was found.")
    else:
        print("Maximum steps reached. No marker found.")
    
    # Clean up
    # cv2.destroyAllWindows()



if __name__ == "__main__":

    move_format = '5patch' 
    step_size = 5  
    map_name = 'ModernCityEnvironment'
   
    save_dir = str('logs/Nav_' +move_format + '_' + map_name)

    main(move_format=move_format, save_dir=save_dir, map_name=map_name, step_size=step_size)