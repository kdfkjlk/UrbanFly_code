import numpy as np
import torch
import os

from .coord_convert_utils import coord_convert, pixel_to_pos, pixel_to_pos_img224
from ultralytics import YOLO


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





class Detector():
    def __init__(self, name='yolo11'):
        if name == 'yolo11':
            self.det_model = self.load_model()
            if torch.cuda.is_available():
                self.det_model = self.det_model.cuda()


    def load_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'weights', 'best.pt')
        model = YOLO(file_path)
        return model


    def detect(self, img_np):
        pred = self.det_model(img_np, verbose=False)[0] ## e.g., torch.Size([1, 18908, 85]): 18908 bounding boxes are detected
        return pred


    def only_detect_marker_bbox(self, rgb_image, conf_thres=0.6):
        '''
        only detect the marker bbox, and return the bbox coordinates in pixel-coordinates and confidence
        '''
        detect_result = self.detect(rgb_image)  ## detect_result: [torch[xyxy, conf, cls], ...], len = # detections
        # detect_result.show()
        detect_result = detect_result.boxes.data

        detect_result_post, detect_result_conf = None, 0
        filtered_tensor = detect_result[detect_result[:, -1] == 0]

        if len(filtered_tensor) == 1:
            result = filtered_tensor[0]  ## filtered_tensor size: (1,6), result: (6,)
            if result[-2] > conf_thres:
                detect_result_post = result[0:4]
                detect_result_conf = result[-2].item()
        elif len(filtered_tensor) > 1:
            _, max_idx = filtered_tensor[:, -2].max(0)  # max(0) 表示在第 0 维（行）上找最大值
            result = filtered_tensor[max_idx]
            if result[-2] > conf_thres:
                detect_result_post = result[0:4]
                detect_result_conf = result[-2].item()
        
        return detect_result_post, detect_result_conf
    



    def detect_marker(self, rgb_image, depth_image, drone_pose, conf_thres=0.6):
        '''
        detect, and then convert the detected pixel to global position in world-coordinate
        '''
        detect_result = self.detect(rgb_image)  ## detect_result: [torch[xyxy, conf, cls], ...], len = # detections
        # detect_result.show()
        detect_result = detect_result.boxes.data

        detect_result_post, det_marker_pose_g, detect_result_conf = None, False, 0
        filtered_tensor = detect_result[detect_result[:, -1] == 0]

        if len(filtered_tensor) == 1:
            result = filtered_tensor[0]  ## filtered_tensor size: (1,6), result: (6,)
            if result[-2] > conf_thres:
                detect_result_post = result
                detect_result_conf = result[-2].item()
        elif len(filtered_tensor) > 1:
            _, max_idx = filtered_tensor[:, -2].max(0)  # max(0) 表示在第 0 维（行）上找最大值
            result = filtered_tensor[max_idx]
            if result[-2] > conf_thres:
                detect_result_post = result
                detect_result_conf = result[-2].item()

        if detect_result_post != None:
            drone_position, drone_orientation = drone_pose.position, drone_pose.orientation

            result_temp = detect_result_post.detach().cpu().numpy()
            center_x = (result_temp[0] + result_temp[2]) // 2
            center_y = (result_temp[1] + result_temp[3]) // 2

            # height_diff = depth_image.squeeze(-1)[int(center_y), int(center_x)]
            # marker_center_pose = [center_x, center_y, height_diff]
            # height_diff = get_depth_value(center_x, center_y, depth_image)[0]
            height_diff = get_depth_value(center_x, center_y, depth_image, rgb_image.shape[0:2], depth_image.shape[0:2])[0]
            marker_center_pose = [center_x, center_y, height_diff]

            det_marker_pose_c = pixel_to_pos(marker_center_pose)
            # det_marker_pose_c = pixel_to_pos_img224(marker_center_pose)

            det_marker_pose_g = coord_convert(det_marker_pose_c, drone_position, drone_orientation=drone_orientation)
            print('obj_position_g', det_marker_pose_g)

            det_marker_pose_g = list(det_marker_pose_g) + [-drone_position.z_val - height_diff]  # (x,y,z)
        
        return detect_result_post, det_marker_pose_g, detect_result_conf




detector = Detector()






if __name__ == '__main__':

    img = np.random.random((480, 640, 3))
    img = (img * 255).astype(np.uint8)

    pred = detector.detect(img)
    pred.show()

