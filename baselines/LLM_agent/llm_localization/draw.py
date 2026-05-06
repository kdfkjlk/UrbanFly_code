import cv2


def draw_bbox_and_center_on_array(img_array, bbox, center, save_path):

    if bbox is None and center is None:
        pass
    
    elif center is None:
        pass

    elif bbox is None and center is not None:
        img = img_array.copy()  # 避免直接修改原始数据

        # 画红色圆点（中心点）
        cx, cy = center
        cv2.circle(img, (int(cx), int(cy)), radius=4, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(save_path, img)
        print(f"已保存: {save_path}")
        
    else:
        # img_array: np.ndarray, 格式为HWC, dtype=uint8
        img = img_array.copy()  # 避免直接修改原始数据

        # 画红色bounding box
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)

        # 画红色圆点（中心点）
        cx, cy = center
        cv2.circle(img, (int(cx), int(cy)), radius=4, color=(0, 0, 255), thickness=-1)

        # 保存图片
        cv2.imwrite(save_path, img)
        print(f"已保存: {save_path}")




def get_coord_from_request(request, format='center'):
    '''
    Input:
        format: 
            "bbox": includes bbox and center
            "center": only has center position 
    '''

    marker_center_position = request['result']['marker_position']
    if marker_center_position is None:
        return None, None
    else:
        cx, cy = marker_center_position
        center = [cx, cy]

    if format == 'center':
        return None, center

    elif format == 'bbox':
        trial = request['result']['trials']['1']
        xmin, ymin, xmax, ymax = trial['x1'], trial['y1'], trial['x2'], trial['y2']
        bbox = [xmin, ymin, xmax, ymax]
        return bbox, center
    
    