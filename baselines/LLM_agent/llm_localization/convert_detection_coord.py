import numpy as np
import math


def bilinear_interpolate(im, x, y):
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, im.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y0, x1]
    Ic = im[y1, x0]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id



def get_depth_value(u_rgb, v_rgb, depth_image, rgb_size, depth_size):
    """
    将 RGB 像素坐标 (u_rgb, v_rgb) 映射到深度图上，并做双线性插值。
    
    参数:
        u_rgb, v_rgb: 在 RGB 图像中的像素坐标
        depth_image: 深度图，shape=(H_d, W_d) 或 (H_d, W_d, C)
        rgb_size:  (H_r, W_r) RGB 图像的尺寸
        depth_size: (H_d, W_d) 深度图的尺寸

    返回:
        depth_value: 插值后的深度值
    """
    H_r, W_r = rgb_size
    H_d, W_d = depth_size

    # 根据图像尺寸动态计算相机内参
    f_rgb_x = W_r / 2.0
    c_rgb_x = W_r / 2.0
    f_rgb_y = H_r / 2.0
    c_rgb_y = H_r / 2.0

    f_depth_x = W_d / 2.0
    c_depth_x = W_d / 2.0
    f_depth_y = H_d / 2.0
    c_depth_y = H_d / 2.0

    # 归一化 RGB 像素坐标
    x_norm = (u_rgb - c_rgb_x) / f_rgb_x
    y_norm = (v_rgb - c_rgb_y) / f_rgb_y

    # 映射到深度图像坐标
    u_depth = x_norm * f_depth_x + c_depth_x
    v_depth = y_norm * f_depth_y + c_depth_y

    # Clamp 防止越界
    u_depth = np.clip(u_depth, 0, W_d - 1)
    v_depth = np.clip(v_depth, 0, H_d - 1)

    # 双线性插值获取深度
    depth_value = bilinear_interpolate(depth_image, u_depth, v_depth)
    return depth_value





def pixel_to_pos(detect_result):
    P = [1332.8958740234375, 0.0, 320.0, 0.0, 0.0, 1332.8958740234375, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # f_x = 1332.8958740234375
    # c_x = 320
    # f_y = 1332.8958740234375
    # c_y = 240

    f_x = 320
    c_x = 320
    f_y = 320
    c_y = 240

    # detect_result = detect_result.detach().cpu().numpy()

    # center_x = (detect_result[0] + detect_result[2]) // 2
    # center_y = (detect_result[1] + detect_result[3]) // 2
    center_x = detect_result[0]
    center_y = detect_result[1]
    z = detect_result[2]

    x_world = (center_x - c_x) * (z) / f_x
    y_world = (center_y - c_y) * (z) / f_y
    return [x_world, y_world]



def pixel_to_pos_img224(detect_result):
    f_x = 112
    c_x = 112
    f_y = 112
    c_y = 112

    center_x = detect_result[0]
    center_y = detect_result[1]
    z = detect_result[2]

    x_world = (center_x - c_x) * (z) / f_x
    y_world = (center_y - c_y) * (z) / f_y
    return [x_world, y_world]



def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians



def coord_convert(obj_position_c, drone_position, drone_orientation):
    _, _, yaw = euler_from_quaternion(drone_orientation.x_val, drone_orientation.y_val,
                                        drone_orientation.z_val, drone_orientation.w_val)
    yaw_init_mat = np.array([[0, 1],[-1, 0]])
    yaw_angle_mat = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    obj_position_delta = np.matmul(obj_position_c, np.matmul(yaw_angle_mat, yaw_init_mat))
    # obj_position_delta = np.matmul(obj_position_c, yaw_angle_mat)
    obj_position_g = [obj_position_delta[0] + drone_position.x_val, obj_position_delta[1] + drone_position.y_val]
    return obj_position_g




def convert_coord_pix2realworld(pix_coord, rgb_image, depth_image, drone_pose):
    center_x, center_y = pix_coord[0], pix_coord[1]
    drone_position, drone_orientation = drone_pose.position, drone_pose.orientation

    height_diff = get_depth_value(center_x, center_y, depth_image, rgb_image.shape[0:2], depth_image.shape[0:2])[0]
    marker_center_pose = [center_x, center_y, height_diff]

    det_marker_pose_c = pixel_to_pos(marker_center_pose)
    # det_marker_pose_c = pixel_to_pos_img224(marker_center_pose)

    det_marker_pose_g = coord_convert(det_marker_pose_c, drone_position, drone_orientation=drone_orientation)
    print('obj_position_g', det_marker_pose_g)

    det_marker_pose_g = list(det_marker_pose_g) + [-drone_position.z_val - height_diff]  # (x,y,z)

    return det_marker_pose_g