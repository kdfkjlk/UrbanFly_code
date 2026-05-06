import math
import numpy as np


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


