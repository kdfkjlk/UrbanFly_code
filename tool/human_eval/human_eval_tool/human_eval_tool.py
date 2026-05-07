import os
from datetime import datetime
import pygame
import sys
import time
import math
import cv2
import numpy as np
import json
import airsim
import cv2



def create_subscreen(width, height, color):
    subscreen = pygame.Surface((width, height))
    subscreen.fill(color)
    return subscreen

def update_rgbImg_subscreen(img, img_size):
    img1_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))  # 创建 Pygame 表面
    img1_surface = pygame.transform.scale(img1_surface, img_size)
    return img1_surface

def update_depthImg_subscreen(img, img_size):
    alpha = 0.5  # 对比度控制 (0.0-3.0)，数值越小，图像越淡
    beta = 80  # 亮度控制 (0-100)，数值越大，图像越亮
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))  # 创建 Pygame 表面
    img_surface = pygame.transform.scale(img_surface, img_size)
    return img_surface


def update_button_screen(subscreen, subscreen_width, font, sim_state, subscreen_height=40):
    done_color = (100, 255, 100)
    future_color = (255, 100, 100)

    if not sim_state.start_fly:
        rect_colors = [future_color, future_color, future_color]
    else:
        if sim_state.flight_state == 'search':
            rect_colors = [done_color, future_color, future_color]
        elif sim_state.flight_state == 'land':
            rect_colors = [done_color, done_color, future_color]
        elif sim_state.is_end:
            rect_colors = [done_color, done_color, done_color]

    pygame.draw.rect(subscreen, rect_colors[0], (20, 20, subscreen_width - 40, subscreen_height))
    pygame.draw.rect(subscreen, rect_colors[1], (20, 80, subscreen_width - 40, subscreen_height))
    # pygame.draw.rect(subscreen, rect_colors[2], (20, 140, subscreen_width - 40, subscreen_height))

    text_game_start = font.render("Game activated: Return", True, (255, 255, 255))
    text_search = font.render("Search done: Space", True, (255, 255, 255))
    # text_land = font.render("Land done: 9", True, (255, 255, 255))

    text_rect_start = text_game_start.get_rect(center=(20 + (subscreen_width - 40) // 2, 20 + subscreen_height // 2))  # 文字在矩形中居中
    text_rect_search = text_search.get_rect(center=(20 + (subscreen_width - 40) // 2, 80 + subscreen_height // 2))  # 文字在矩形中居中
    # text_rect_land = text_land.get_rect(center=(20 + (subscreen_width - 40) // 2, 140 + subscreen_height // 2))  # 文字在矩形中居中

    subscreen.blit(text_game_start, text_rect_start)
    subscreen.blit(text_search, text_rect_search)
    # subscreen.blit(text_land, text_rect_land)

    return subscreen


def time_counter(start_time, max_time=120):
    if start_time is None:
        return max_time

    elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
    countdown_time = max(0, max_time - elapsed_time)  # 倒计时，确保不为负值
    return countdown_time

# def update_time_countdown_subscreen(time_screen, font, countdown_time, subscreen_size):
#     ## subscreen_size (width, height)
#     countdown_text = font.render(f"Rest time: {countdown_time}s", True, (255, 255, 255))
#     time_screen.fill((0, 50, 150))  # 填充背景色
#     screen_center_pose = (subscreen_size[0] // 2, subscreen_size[1] // 2)
#     time_screen.blit(countdown_text, countdown_text.get_rect(center=screen_center_pose))
#     return time_screen


def update_time_countdown_subscreen(time_screen, font, countdown_time, subscreen_size):
    ## subscreen_size (width, height)

    # 创建两行文本
    rest_landing_time = max(0, countdown_time - 30)
    line1_text = font.render(f"Rest Landing Time: {rest_landing_time}s", True, (255, 255, 255))
    line2_text = font.render(f"Rest Total Time: {countdown_time}s", True, (255, 255, 255))

    time_screen.fill((0, 50, 150))  # 填充背景色
    screen_center_pose = (subscreen_size[0] // 2, subscreen_size[1] // 2)

    # 获取文本的矩形区域
    line1_rect = line1_text.get_rect(center=(screen_center_pose[0], screen_center_pose[1] - 20))  # 上行文本
    line2_rect = line2_text.get_rect(center=(screen_center_pose[0], screen_center_pose[1] + 20))  # 下行文本

    # 将文本绘制到屏幕上
    time_screen.blit(line1_text, line1_rect)
    time_screen.blit(line2_text, line2_rect)

    return time_screen


def update_text_subscreen(text_screen, text, font, subscreen_size):
    ## subscreen_size (width, height)
    dynamic_text = font.render(text, True, (255, 255, 255))
    text_screen.fill((100, 0, 0))  # 填充背景色
    screen_center_pose = (subscreen_size[0] // 2, subscreen_size[1] // 2)
    text_screen.blit(dynamic_text, dynamic_text.get_rect(center=screen_center_pose))
    return text_screen


def get_subscreen_position(pose, subscreen_size, margin, font_size):
    """
    根据给定的子窗口位置代号（行，列），计算子窗口的实际坐标位置。

    参数:
    - subscreen_pose: (row, col), row: 子窗口所在的行数（1 或 2）, col: 子窗口所在的列数（1 到 3）
    - subscreen_size: (subscreen_width, subscreen_height)
    - margin: 子窗口之间的间隔距离

    返回: - 子窗口的左上角坐标 (x, y)
    """
    # 行、列从 1 开始的，所以需要减 1 来调整索引
    x = (pose[1] - 1) * (subscreen_size[0] + margin) + margin  # 计算x坐标
    y = (pose[0] - 1) * (subscreen_size[1] + margin + font_size) + margin + font_size # 计算y坐标
    return (x, y)


def get_title_position(pose, subscreen_size, margin, font_size):
    x = (pose[1] - 1) * (subscreen_size[0] + margin) + margin  # 计算x坐标
    y = (pose[0] - 1) * (subscreen_size[1] + margin + font_size) + margin  # 计算y坐标
    return (x,y)





class Human_eval_tool():
    def __init__(self):
        ## define search range and step size
        self.search_circle_radius = 30

        # self.speed = 0.5
        self.movement_duration = 0.02

        self.ascend_step = self.descend_step = 1
        self.left_step = self.right_step = 1

        self.forward_step = 1

        self.turn_angle = 15
        self.time_limit = 120

        ## settings for plotting
        self.margin = 10
        self.font_space = 20
        self.screen_width, self.screen_height = 1800, 920
        self.subscreen_width = (self.screen_width - 4 * self.margin) // 3  # 每行3个窗口，留出左右空隙
        self.subscreen_height = (self.screen_height - 4 * self.margin) // 2  # 两行，留出上下空隙
        self.subscreen_size = (self.subscreen_width, self.subscreen_height)

        self.success_land_dis = 2


    def _setup_flight(self):
        marker_pose = self.data['marker_pose']
        drone_pose = self.data['drone_pose']
        time = self.data['time']
        weather = self.data['weather']

        self.drone.set_marker_pose(marker_pose)
        self.drone.reset_drone_pose(init_pose=list(drone_pose[0:3]), init_yaw=drone_pose[-1])
        self.drone.set_time_of_day(time)
        self.drone.set_weather(weather)


    def setup_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Drone Trajectory Display')

        self.font = pygame.font.SysFont('Arial', 20)
        self.font_color = (255, 0, 0)

        self.button_subscreen = create_subscreen(self.subscreen_width, self.subscreen_height, (200, 200, 200))  # 固定文字窗口
        self.time_subscreen = create_subscreen(self.subscreen_width, self.subscreen_height, (0, 50, 150))  # 倒计时窗口
        self.dynamic_text_subscreen = create_subscreen(self.subscreen_width, self.subscreen_height,(255, 100, 100))  # 动态提示窗口
        self.full_text_screen = create_subscreen(self.screen_width, self.screen_height, (200, 200, 200))

    # def close_pygame(self):
    #     # pygame.display.quit()  # 确保显示关闭
    #     pygame.event.clear()  # 清除事件队列
    #     pygame.quit()
    #     pygame.time.delay(500)  # 等待片刻，确保所有资源都释放


    def interprete_key(self, key):

        print('input key: ', key)

        drone_pose = self.drone.get_drone_pose()
        current_position = np.array([drone_pose.position.x_val, drone_pose.position.y_val, drone_pose.position.z_val])
        current_rotation = np.array([
            drone_pose.orientation.x_val, drone_pose.orientation.y_val, drone_pose.orientation.z_val,
            drone_pose.orientation.w_val
        ])

        if key == 'forward':
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

            new_position = np.array(current_position) + unit_vector * self.forward_step
            new_rotation = current_rotation.copy()


        elif key == 'turn_left':
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


        elif key == 'turn_right':
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


        elif key == 'ascend':
            pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
                x_val=current_rotation[0],
                y_val=current_rotation[1],
                z_val=current_rotation[2],
                w_val=current_rotation[3]
            ))
            pitch = 0
            roll = 0

            unit_vector = np.array([0, 0, -1])

            new_position = np.array(current_position) + unit_vector * self.ascend_step
            new_rotation = current_rotation.copy()


        elif key == 'descend':
            pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
                x_val=current_rotation[0],
                y_val=current_rotation[1],
                z_val=current_rotation[2],
                w_val=current_rotation[3]
            ))
            pitch = 0
            roll = 0

            unit_vector = np.array([0, 0, -1])

            new_position = np.array(current_position) + unit_vector * self.descend_step * (-1)
            new_rotation = current_rotation.copy()


        elif key == 'move_left':
            pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
                x_val=current_rotation[0],
                y_val=current_rotation[1],
                z_val=current_rotation[2],
                w_val=current_rotation[3]
            ))
            pitch = 0
            roll = 0

            unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
            unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
            unit_vector = np.array([unit_x, unit_y, 0])

            new_position = np.array(current_position) + unit_vector * self.left_step * (-1)
            new_rotation = current_rotation.copy()


        elif key == 'move_right':
            pitch, roll, yaw = airsim.to_eularian_angles(airsim.Quaternionr(
                x_val=current_rotation[0],
                y_val=current_rotation[1],
                z_val=current_rotation[2],
                w_val=current_rotation[3]
            ))
            pitch = 0
            roll = 0

            unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
            unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
            unit_vector = np.array([unit_x, unit_y, 0])

            new_position = np.array(current_position) + unit_vector * self.right_step
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


    def do_action(self, new_pose):
        self.drone.set_drone_pose(new_pose)

        drone_pose = self.drone.get_drone_pose()
        print('drone_current_pose', drone_pose)


    def convert_pose_for_save(self, pose):
        position = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        orientation = [pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val]
        return position, orientation

    def process_depth_img(self, depth_img):
        cliped_img = depth_img.squeeze(-1).clip(min=0, max=255)
        # 将深度图像转换为 8 位无符号整型, 确保图像能正确地表示为灰度
        gray_depth_image = cliped_img.astype(np.uint8)
        return gray_depth_image

    def capture_imgs(self):
        rgb_forward = self.drone.get_current_image(image_type='scene', external=False, angle='forward', help_cam=False)
        depth_forward = self.drone.get_current_image(image_type='depth', external=False, angle='forward', help_cam=False)
        depth_forward = self.process_depth_img(depth_forward)

        rgb_downward = self.drone.get_current_image(image_type='scene', external=False, angle='downward', help_cam=False)
        depth_downward = depth_forward

        # rgb_forward = self.resize_image(rgb_forward, (224,224))
        # depth_forward = self.resize_image(depth_forward, (256,256))
        # rgb_downward = self.resize_image(rgb_downward, (224,224))
        # depth_downward = self.resize_image(depth_downward, (256,256))

        # rgb_forward = rgb_forward[:, :, [1, 2, 0]]
        # rgb_downward = rgb_downward[:, :, [1, 2, 0]]
        return rgb_forward, depth_forward, rgb_downward, depth_downward


    def resize_image(self, img, new_size):
        resized_image = cv2.resize(img, new_size)
        return resized_image


    def text_generator(self):
        text = None

        if self.sim_state.is_collide:
            text = 'Collision! Game Over'
            return text
        elif self.sim_state.is_end:
            text = 'Success Landing, wait for next iteration!'
            return text
        elif self.sim_state.time_is_up:
            text = 'Time is up. Game over!'
            return text


        # Distance between current drone position to destination
        drone_pose = self.sim_state.trajectory[-1]['drone_position']
        # marker_pose = self.sim_state.marker_pose
        dis_drone2marker = math.sqrt((drone_pose[0] - self.sim_state.drone_pose[0])**2 + (drone_pose[1] - self.sim_state.drone_pose[1])**2)
        if dis_drone2marker > self.search_circle_radius:
            text = 'Far away from initial drone position!'
            return text
        # print('uuuuuuuuuuu', drone_pose, self.sim_state.drone_pose, dis_drone2marker)

        if self.sim_state.trajectory[-1]['drone_position'][-1] < -40:
            text = 'Reminder: Drone height is 40m'
        elif self.sim_state.trajectory[-1]['drone_position'][-1] < -30:
            text = 'Reminder: Drone height is already 30m'

        return text


    def update_drone_state(self, log_save_path):
        '''
        update collision_state, time_is_up
        '''
        drone_pose = self.drone.get_drone_pose()
        self.sim_state.is_collide = self.drone.check_drone_collision(drone_pose)

        self.countdown_time = time_counter(self.start_time, self.time_limit)
        self.sim_state.time_is_up = True if self.countdown_time <= 0 else False

        np_drone_pose = np.array(self.sim_state.trajectory[-1]['drone_position'])
        np_drone_pose[-1] = -np_drone_pose[-1]
        np_marker_pose = np.array(self.sim_state.marker_pose)
        dis_3d = math.sqrt(sum(np.square(np_drone_pose - np_marker_pose)))

        if (self.sim_state.is_collide or self.sim_state.is_end or self.sim_state.time_is_up):
            self.sim_state.save_flight_data(log_save_path)

        if dis_3d <= self.success_land_dis:
            self.sim_state.is_end = True
            # self.sim_state.total_time = round(time.time() - self.sim_state.start_time, 2),
            self.sim_state.save_flight_data(log_save_path)


    def render_screen(self):
        rgb_forward, depth_forward, rgb_downward, depth_downward = self.capture_imgs()
        self.screen.fill((255, 255, 255))     ## clean screen

        # plot img and title
        depth_img_surface = update_depthImg_subscreen(depth_forward, self.subscreen_size)
        subscreen_pose = get_subscreen_position((2,1), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(depth_img_surface, subscreen_pose)  # 在屏幕上显示图像
        text_surface = self.font.render('Depth_forward', True, self.font_color)
        text_pose = get_title_position((2,1), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(text_surface, text_pose)

        rgb_forward_img_surface = update_rgbImg_subscreen(rgb_forward, self.subscreen_size)
        subscreen_pose = get_subscreen_position((1, 2), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(rgb_forward_img_surface, subscreen_pose)  # 右侧图片
        text_surface = self.font.render('RGB_forward', True, self.font_color)
        text_pose = get_title_position((1, 2), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(text_surface, text_pose)

        rgb_downward_img_surface = update_rgbImg_subscreen(rgb_downward, self.subscreen_size)
        subscreen_pose = get_subscreen_position((2, 2), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(rgb_downward_img_surface, subscreen_pose)  # 右侧图片
        text_surface = self.font.render('RGB_downward', True, self.font_color)
        text_pose = get_title_position((2, 2), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(text_surface, text_pose)

        ## dynamic text subscreen
        text = self.text_generator()
        text_screen = update_text_subscreen(self.dynamic_text_subscreen, text, self.font, self.subscreen_size)
        subscreen_pose = get_subscreen_position((1, 3), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(text_screen, subscreen_pose)
        text_surface = self.font.render('Text Reminder', True, self.font_color)
        text_pose = get_title_position((1, 3), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(text_surface, text_pose)

        ## countdown_time subscreen
        # countdown_time = time_counter(self.start_time, self.time_limit)
        time_screen = update_time_countdown_subscreen(self.time_subscreen, self.font, self.countdown_time, self.subscreen_size)
        subscreen_pose = get_subscreen_position((1, 1), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(time_screen, subscreen_pose)
        text_surface = self.font.render('Time Countdown', True, self.font_color)
        text_pose = get_title_position((1, 1), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(text_surface, text_pose)

        ## button subscreen
        button_subscreen = update_button_screen(self.button_subscreen, self.subscreen_width, self.font, self.sim_state, subscreen_height=40)
        subscreen_pose = get_subscreen_position((2, 3), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(button_subscreen, subscreen_pose)
        text_surface = self.font.render('Button Reminder', True, self.font_color)
        text_pose = get_title_position((2, 3), self.subscreen_size, self.margin, self.font_space)
        self.screen.blit(text_surface, text_pose)

        pygame.display.flip()



    def control(self, log_save_path):

        while True:

            ## keys to change state
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # pygame.quit()
                    # sys.exit()
                    self.sim_state.is_end = True
                    self.sim_state.save_flight_data(log_save_path)
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:  ## game is activated, begin to search
                        self.sim_state.flight_state = 'search'
                        self.sim_state.start_fly = True
                        self.start_time = pygame.time.get_ticks()  # 重置计时器

                    elif event.key == pygame.K_SPACE:  ## search done
                        self.sim_state.flight_state = 'land'  ## state is search

                    if not self.sim_state.start_fly:
                        continue

            if self.sim_state.is_end or self.sim_state.is_collide or self.sim_state.time_is_up:
                break

            if self.sim_state.start_fly:
                ## keys for movement control
                pygame.event.pump()
                keys = pygame.key.get_pressed()

                if keys[pygame.K_UP]:
                    new_pose = self.interprete_key('forward')
                    self.do_action(new_pose)
                    drone_pose, drone_ori = self.convert_pose_for_save(new_pose)
                    self.sim_state.update_trajectory('forward', drone_pose, drone_ori)

                elif keys[pygame.K_LEFT]:
                    new_pose = self.interprete_key('move_left')
                    self.do_action(new_pose)
                    drone_pose, drone_ori = self.convert_pose_for_save(new_pose)
                    self.sim_state.update_trajectory('move_left', drone_pose, drone_ori)

                elif keys[pygame.K_RIGHT]:
                    new_pose = self.interprete_key('move_right')
                    self.do_action(new_pose)
                    drone_pose, drone_ori = self.convert_pose_for_save(new_pose)
                    self.sim_state.update_trajectory('move_right', drone_pose, drone_ori)

                elif keys[pygame.K_w]:
                    new_pose = self.interprete_key('ascend')
                    self.do_action(new_pose)
                    drone_pose, drone_ori = self.convert_pose_for_save(new_pose)
                    self.sim_state.update_trajectory('ascend', drone_pose, drone_ori)

                elif keys[pygame.K_s]:
                    new_pose = self.interprete_key('descend')
                    self.do_action(new_pose)
                    drone_pose, drone_ori = self.convert_pose_for_save(new_pose)
                    self.sim_state.update_trajectory('descend', drone_pose, drone_ori)

                elif keys[pygame.K_a]:
                    new_pose = self.interprete_key('turn_left')
                    self.do_action(new_pose)
                    drone_pose, drone_ori = self.convert_pose_for_save(new_pose)
                    self.sim_state.update_trajectory('turn_left', drone_pose, drone_ori)

                elif keys[pygame.K_d]:
                    new_pose = self.interprete_key('turn_right')
                    self.do_action(new_pose)
                    drone_pose, drone_ori = self.convert_pose_for_save(new_pose)
                    self.sim_state.update_trajectory('turn_right', drone_pose, drone_ori)

                else:
                    # new_pose = self.interprete_key('invalid_key')
                    # self.do_action(new_pose)
                    # print('Invalid Key')
                    pass

            self.update_drone_state(log_save_path)
            self.render_screen()
            time.sleep(self.movement_duration)


    def judge_refly(self):
        ## click "land" button too late
        search_samples = [item for item in self.sim_state.trajectory if item['flight_stage'] == 'land']
        sorted_samples = sorted(search_samples, key=lambda x: x['time_s'])
        if sorted_samples:
            first_time_step = sorted_samples[0]['time_step']
            last_time_step = sorted_samples[-1]['time_step']
            step_land = last_time_step - first_time_step
        else:
           step_land = 0

        if (step_land <= 5 and self.sim_state.is_end):
            not_need_refly = False
        else:
            not_need_refly = True

        print('pppppppppppp', not_need_refly, self.sim_state.is_end, step_land, self.sim_state.is_collide)

        return not_need_refly




    def main(self, data, save_path, drone, sim_state):
        ## initialization
        self.drone = drone
        self.data = data
        self.setup_pygame()
        self._setup_flight()
        self.sim_state = sim_state
        # todo: make sure the init time is the same
        self.sim_state.start_flying_logs()
        # self.start_time = pygame.time.get_ticks()  # 重置计时器
        self.start_time = None

        ## start flying
        self.control(save_path)
        time.sleep(2)
        pygame.quit()

        need_refly = self.judge_refly()
        return need_refly







