import json
import airsim
import math
import numpy as np
import cv2
from airsim.scenario_generation.scenario_manager import ScenarioManager
import time



def load_scenario(client, data):
    marker_pose = data['marker_pose']
    drone_pose = data['drone_pose']
    weather = data['weather']
    time = data['time']


    print(client.simSetObjectPose('marker0', airsim.Pose(airsim.Vector3r(marker_pose[0], marker_pose[1], -marker_pose[2]))))
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(drone_pose[0], drone_pose[1], -drone_pose[2]),  airsim.to_quaternion(0, 0, math.radians(drone_pose[3]))),ignore_collision=True)
    
    client.simEnableWeather(True)
    client.simSetWeatherParameter(airsim.WeatherParameter.Rain, min(weather[0], 1))
    client.simSetWeatherParameter(airsim.WeatherParameter.Roadwetness, min(weather[1], 1))
    client.simSetWeatherParameter(airsim.WeatherParameter.Snow, min(weather[2], 1))
    client.simSetWeatherParameter(airsim.WeatherParameter.RoadSnow, min(weather[3], 1))
    client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, min(weather[4], 1))
    client.simSetWeatherParameter(airsim.WeatherParameter.RoadLeaf, min(weather[5], 1))
    client.simSetWeatherParameter(airsim.WeatherParameter.Dust, min(weather[6], 1))
    client.simSetWeatherParameter(airsim.WeatherParameter.Fog, min(weather[7], 1))


    hour, minute = time[0], time[1]
    hour_time = int(hour * 24)
    min_time = int(minute * 60)
    sim_time = "2023-09-06 {}:{}:00".format(hour_time, min_time)
    client.simSetTimeOfDay(True, sim_time)

def get_camera_intrinsics(image_width, image_height, fov_degrees=90):
    """
    Get camera intrinsic parameters. You can modify these based on your actual camera settings.
    
    Args:
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        fov_degrees: Field of view in degrees (default: 90)
    
    Returns:
        K: Camera intrinsic matrix (3x3)
    """
    fov_radians = math.radians(fov_degrees)
    focal_length = image_width / (2 * math.tan(fov_radians / 2))
    
    # Camera intrinsic matrix
    K = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ])
    
    return K

def split_image_to_regions(image, grid_size=3):
    """
    Split image into a grid of regions.
    
    Args:
        image: Input image (numpy array)
        grid_size: Size of the grid (default: 3 for 3x3 grid)
    
    Returns:
        regions: List of image regions
        region_centers: List of (x, y) pixel coordinates for each region's center
    """
    height, width = image.shape[:2]
    region_height = height // grid_size
    region_width = width // grid_size
    
    regions = []
    region_centers = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate region boundaries
            y_start = i * region_height
            y_end = (i + 1) * region_height if i < grid_size - 1 else height
            x_start = j * region_width
            x_end = (j + 1) * region_width if j < grid_size - 1 else width
            
            # Extract region
            region = image[y_start:y_end, x_start:x_end]
            regions.append(region)
            
            # Calculate center of the region
            center_x = (x_start + x_end) / 2
            center_y = (y_start + y_end) / 2
            region_centers.append((center_x, center_y))
    
    return regions, region_centers

def pixel_to_world_coordinates(pixel_coords, camera_intrinsics, camera_pose, height_above_ground):
    """
    Convert pixel coordinates to 3D world coordinates for a downward-facing camera.
    
    Args:
        pixel_coords: List of (x, y) pixel coordinates
        camera_intrinsics: Camera intrinsic matrix K (3x3)
        camera_pose: Camera pose from AirSim
        height_above_ground: Height of the camera above the ground
    
    Returns:
        world_coords: List of 3D world coordinates
    """
    world_coords = []
    
    # Extract camera position
    cam_pos = np.array([camera_pose.position.x_val, camera_pose.position.y_val, camera_pose.position.z_val])
    cam_orientation = camera_pose.orientation
    
    # Debug: print camera pose info
    # print(f"Camera position: {cam_pos}")
    # print(f"Camera orientation (quaternion): w={cam_orientation.w_val:.3f}, x={cam_orientation.x_val:.3f}, y={cam_orientation.y_val:.3f}, z={cam_orientation.z_val:.3f}")
    
    # Convert quaternion to rotation matrix
    q = np.array([cam_orientation.w_val, cam_orientation.x_val, cam_orientation.y_val, cam_orientation.z_val])
    R = quaternion_to_rotation_matrix(q)
    
    # print(f"Rotation matrix:\n{R}")
    
    # Camera intrinsics
    K_inv = np.linalg.inv(camera_intrinsics)
    
    # print(f"Camera intrinsics:\n{camera_intrinsics}")
    # print(f"Height above ground: {height_above_ground}")
    
    for i, (pixel_x, pixel_y) in enumerate(pixel_coords):
        # Convert pixel to normalized camera coordinates
        pixel_homogeneous = np.array([pixel_x, pixel_y, 1])
        cam_coords_normalized = K_inv @ pixel_homogeneous
        
        # For a downward-facing camera, create a ray in camera coordinates
        # In camera coordinates, Z typically points forward, so for downward camera
        # we need to handle the coordinate system correctly
        
        # Create ray direction in camera coordinates (assuming camera looks down)
        # The ray goes from camera center through the pixel
        ray_direction_cam = np.array([cam_coords_normalized[0], cam_coords_normalized[1], 1])
        ray_direction_cam = ray_direction_cam / np.linalg.norm(ray_direction_cam)
        
        # Transform ray direction to world coordinates
        ray_direction_world = R @ ray_direction_cam
        
        # Ground plane intersection
        # The ground is at z = cam_pos[2] - height_above_ground
        ground_z = cam_pos[2] - height_above_ground
        
        # Ray parametric equation: point = cam_pos + t * ray_direction_world
        # For ground intersection: cam_pos[2] + t * ray_direction_world[2] = ground_z
        if abs(ray_direction_world[2]) > 1e-6:  # Avoid division by zero
            t = (ground_z - cam_pos[2]) / ray_direction_world[2]
            world_point = cam_pos + t * ray_direction_world
            
            # print(f"Pixel {i+1} ({pixel_x:.1f}, {pixel_y:.1f}):")
            # print(f"  Cam coords normalized: {cam_coords_normalized}")
            # print(f"  Ray direction (cam): {ray_direction_cam}")
            # print(f"  Ray direction (world): {ray_direction_world}")
            # print(f"  t parameter: {t}")
            # print(f"  World point: {world_point}")
            
            world_coords.append(world_point)
        else:
            # Ray is parallel to ground, use camera position projected down
            world_point = np.array([cam_pos[0], cam_pos[1], ground_z])
            world_coords.append(world_point)
    
    return world_coords





# def simple_downward_projection(pixel_coords, camera_pose, height_above_ground, image_width, image_height, fov_degrees=90):
#     """
#     Simplified projection for downward-facing camera assuming camera is pointing straight down.
#     This is a more direct approach that should work better for downward cameras.
    
#     Args:
#         pixel_coords: List of (x, y) pixel coordinates
#         camera_pose: Camera pose from AirSim
#         height_above_ground: Height of the camera above the ground
#         image_width: Width of the image in pixels
#         image_height: Height of the image in pixels
#         fov_degrees: Field of view in degrees
    
#     Returns:
#         world_coords: List of 3D world coordinates
#     """
#     world_coords = []
    
#     # Camera position
#     cam_pos = np.array([camera_pose.position.x_val, camera_pose.position.y_val, camera_pose.position.z_val])
    
#     # Calculate the ground coverage (assuming camera points straight down)
#     fov_radians = math.radians(fov_degrees)
#     ground_width = 2 * height_above_ground * math.tan(fov_radians / 2)
#     ground_height = ground_width * (image_height / image_width)
    
#     # Calculate world coordinates for each pixel
#     for pixel_x, pixel_y in pixel_coords:
#         # Convert pixel coordinates to normalized coordinates (-1 to 1)
#         norm_x = (pixel_x - image_width / 2) / (image_width / 2)
#         norm_y = (pixel_y - image_height / 2) / (image_height / 2)
        
#         # Calculate world offset from camera position
#         world_offset_x = norm_x * (ground_width / 2)
#         world_offset_y = norm_y * (ground_height / 2)

#         # For AirSim coordinate system, we need to be careful about axis orientation
#         # Assuming standard NED (North-East-Down) coordinate system
#         world_x = cam_pos[0] + world_offset_x
#         world_y = cam_pos[1] + world_offset_y
#         world_z = cam_pos[2] - height_above_ground  # Ground level
        
#         world_coords.append(np.array([world_x, world_y, world_z]))
    
#     return world_coords




def simple_downward_projection(pixel_coords, camera_pose, current_pose, height_above_ground, image_width, image_height, fov_degrees=90):
    """
    Simplified projection for downward-facing camera assuming camera is pointing straight down.
    This is a more direct approach that should work better for downward cameras.
    
    Args:
        pixel_coords: List of (x, y) pixel coordinates
        camera_pose: Camera pose from AirSim
        height_above_ground: Height of the camera above the ground
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        fov_degrees: Field of view in degrees
    
    Returns:
        world_coords: List of 3D world coordinates
    """
    world_coords = []
    
    # Camera position
    cam_pos = np.array([camera_pose.position.x_val, camera_pose.position.y_val, camera_pose.position.z_val])
    pitch, roll, cam_yaw = airsim.to_eularian_angles(current_pose.orientation)
    
    # Calculate the ground coverage (assuming camera points straight down)
    fov_radians = math.radians(fov_degrees)
    ground_width = 2 * height_above_ground * math.tan(fov_radians / 2)
    ground_height = ground_width * (image_height / image_width)
    
    # Calculate world coordinates for each pixel
    for pixel_x, pixel_y in pixel_coords:
        # Convert pixel coordinates to normalized coordinates (-1 to 1)
        norm_x = (pixel_x - image_width / 2) / (image_width / 2)
        norm_y = (pixel_y - image_height / 2) / (image_height / 2)
        
        # Calculate world offset from camera position
        world_offset_x = norm_x * (ground_width / 2)
        world_offset_y = norm_y * (ground_height / 2)

        ## convert coordinates from img coord to airsim coord (without yaw)
        airsim_offset_x = -world_offset_y
        airsim_offset_y = world_offset_x

        # 旋转到世界 NED 坐标系
        dx_world =  airsim_offset_x * math.cos(cam_yaw) - airsim_offset_y * math.sin(cam_yaw)
        dy_world =  airsim_offset_x * math.sin(cam_yaw) + airsim_offset_y * math.cos(cam_yaw)
        


        # For AirSim coordinate system, we need to be careful about axis orientation
        # Assuming standard NED (North-East-Down) coordinate system
        # world_x = cam_pos[0] + world_offset_x
        # world_y = cam_pos[1] + world_offset_y
        # world_z = cam_pos[2] - height_above_ground  # Ground level

        world_x = cam_pos[0] + dx_world
        world_y = cam_pos[1] + dy_world
        world_z = cam_pos[2] - height_above_ground   # 地面
        
        world_coords.append(np.array([world_x, world_y, world_z]))
    
    return world_coords

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion as [w, x, y, z]
    
    Returns:
        R: 3x3 rotation matrix
    """
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R

def visualize_regions(image, region_centers, world_coords):
    """
    Visualize the image regions and their centers.
    
    Args:
        image: Input image
        region_centers: List of (x, y) pixel coordinates
        world_coords: List of 3D world coordinates
    """
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Draw region centers and world coordinates
    for i, ((px, py), world_coord) in enumerate(zip(region_centers, world_coords)):
        # Draw circle at region center
        cv2.circle(vis_image, (int(px), int(py)), 10, (0, 255, 0), -1)
        
        # Add text with region number and world coordinates
        # text = f"R{i+1}: ({world_coord[0]:.2f}, {world_coord[1]:.2f}, {world_coord[2]:.2f})"
        # cv2.putText(vis_image, text, (int(px-50), int(py-20)), 
                #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_image

def analyze_image_regions(image, camera_pose, current_pose, height_above_ground=10.0):
    """
    Complete analysis of image regions with 3D world coordinates.
    
    Args:
        image: Input image from AirSim
        camera_pose: Camera pose from AirSim
        height_above_ground: Estimated height above ground in meters
    
    Returns:
        regions: List of image regions
        region_centers: List of pixel coordinates for region centers
        world_coords: List of 3D world coordinates for region centers
        visualization: Visualization image with annotations
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Split image into regions
    regions, region_centers = split_image_to_regions(image, grid_size=3)
    
    # print("=== Debugging Coordinate Transformation ===")
    
    # Try the complex method first (for comparison)
    # camera_intrinsics = get_camera_intrinsics(width, height, fov_degrees=90)
    # world_coords_complex = pixel_to_world_coordinates(region_centers, camera_intrinsics, 
    #                                                 camera_pose, height_above_ground)
    
    # print("\n=== Using Simplified Method ===")
    # Use simplified method
    world_coords = simple_downward_projection(region_centers, camera_pose, current_pose, height_above_ground, 
                                            width, height, fov_degrees=90)
    
    # print("Simplified method results:")
    # for i, ((px, py), world_coord) in enumerate(zip(region_centers, world_coords)):
    #     print(f"Region {i+1}: Pixel ({px:.1f}, {py:.1f}) -> World ({world_coord[0]:.2f}, {world_coord[1]:.2f}, {world_coord[2]:.2f})")
    
    # Create visualization
    visualization = visualize_regions(image, region_centers, world_coords)
    
    return regions, region_centers, world_coords, visualization






def calculate_step_size(camera_height_m,
                                 hfov_deg=90,
                                 image_width_px=640,
                                 image_height_px=480):
    """
    计算：相机竖直向下俯拍时，从网格中心 (1,1) → 前方网格 (0,1) 中心，所需的水平位移（米）。

    参数
    ----
    camera_height_m : float 相机离地面的垂直高度（米）
    hfov_deg : float **水平**视场角 (HFOV)。若只有对角 / 垂直 FOV，请先换算。
    image_width_px, image_height_px : int 图像分辨率，默认 640×480

    返回
    ----
    step_m : float
        需要向前移动的距离（米）
    """


    fov_radians = math.radians(hfov_deg)
    ground_width = 2 * camera_height_m * math.tan(fov_radians / 2)
    ground_height = ground_width * (image_height_px / image_width_px)

    step_size = ground_height / 3
    return step_size





# ----------------- 示例 -----------------
if __name__ == "__main__":
    H   = 10     # 相机高度 50 m
    d = calculate_step_size(H)
    print(f"相机需向前移动 {d:.2f} m 才能到达 (0,1) patch 中心")





# if __name__ == "__main__":
#     with open('data_episode_drone/test/UrbanDistrict/test.json', 'r') as f:
#         data = json.load(f)

#     print(data[0])
#     client = airsim.VehicleClient()
#     load_scenario(client, data[6])
#     sm = ScenarioManager(client)
#     image, camera_pose = sm.get_current_scene(camera_name="bottom_center", image_type=0, external=False)

#     print("Image shape:", image.shape)
#     print("Camera pose:", camera_pose)
#     current_pose = client.simGetVehiclePose()

#     # Analyze image regions and get 3D world coordinates
#     regions, region_centers, world_coords, visualization = analyze_image_regions(image, camera_pose)

#     # Print results
#     print("\nRegion Analysis Results:")
#     print("=" * 50)
#     for i, ((px, py), (wx, wy, wz)) in enumerate(zip(region_centers, world_coords)):
#         print(f"Region {i+1}:")
#         print(f"  Pixel center: ({px:.1f}, {py:.1f})")
#         print(f"  World coordinates: ({wx:.2f}, {wy:.2f}, {wz:.2f})")
#         print()

#     # Save visualization
#     cv2.imwrite('regions_visualization.png', visualization)
#     print("Visualization saved as 'regions_visualization.png'")

#     # Save individual regions
#     for i, region in enumerate(regions):
#         cv2.imwrite(f'region_{i+1}.png', region)
#     print(f"Individual regions saved as 'region_1.png' to 'region_9.png'")

#     time.sleep(5)
#     # move drone to region 1 center
#     client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(world_coords[8][0], world_coords[8][1], current_pose.position.z_val),  current_pose.orientation),ignore_collision=True)

