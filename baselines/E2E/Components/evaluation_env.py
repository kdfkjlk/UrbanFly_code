import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time
import sys
import random

try:
    from .grid_manage import GridMapManager
    from .obstacle_simulate import ObstacleManager, ObstacleManager3D, ObstacleManager_airsim2D
    from .Env_airsim.client import Drone_tool, drone_config
    from .Env_airsim.dataload_manager import DataLoad_Manager
    # from .Env_airsim.episode_manager import EpisodeTracker
    from .Env_airsim.simulator_manager import Simulator_Manager
except:
    from grid_manage import GridMapManager
    from obstacle_simulate import ObstacleManager, ObstacleManager3D, ObstacleManager_airsim2D
    from Env_airsim.client import Drone_tool, drone_config
    from Env_airsim.dataload_manager import DataLoad_Manager
    # from Env_airsim.episode_manager import EpisodeTracker
    from Env_airsim.simulator_manager import Simulator_Manager



class DummyEnv(gym.Env):
    def __init__(self, grid_size=61, max_steps=400, map_resolution=2.0, patch_size=9):

        assert grid_size % 2 == 1
        assert patch_size % 2 == 1

        # self.step_size = 2
        # assert self.step_size == map_resolution

        self.grid_size = int(grid_size / map_resolution) + 1
        self.patch_size = int(patch_size / map_resolution) + 1
        self.max_steps = max_steps
        self.map_resolution = map_resolution
        self.step_count = 0

        self.turn_angle = 90.0
        self.grid_mgr = GridMapManager(grid_size=self.grid_size, resolution=1, patch_size=self.patch_size)


        # x_min = y_min = -1.0
        # x_max = y_max = 1.0
        # yaw_min, yaw_max = -1.0, 1.0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            "map": spaces.Box(0, 1, shape=(1, self.grid_size, self.grid_size), dtype=np.float32),
            "pos": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
        })

        self.last_info = {}

        ## reward values
        # self.time_reward = -0.001
        # self.forward_reward = 0.01
        # self.visit_reward = 0.1
        # self.out_of_boundary_penalty = -0.5
        # self.budget_max = 100

        self.time_reward = -0.01
        # self.forward_reward = 0.005
        self.turn_reward = -0.005 # -0.01
        self.visit_reward = 0.05 # 1.0
        self.out_of_boundary_penalty = -3.0 # -5.0  # 0.1 / 1.0
        self.success_reward = 1.0  #3.0
        # self.budget_max = 100

        self.success_threshold = 0.5  # Example threshold for success
    
    
    def reset(self):
        """
        Reset the environment. Accepts the drone's world position.
        """
        self.step_count = 0
        # self.budget = self.budget_max
        
        # self.grid_center_pose_airsim = [0.0, 0.0]
        # self.agent_pos_airsim = [0.0, 0.0]
        # self.agent_yaw = 0
        # self.height = 1

        self.grid_mgr.reset()
        # self.grid_mgr.set_origin_world(self.grid_center_pose_airsim)
        # self.grid_mgr.update_grid_map()

        self.last_visited_grid = 0
        self.out_of_boundary = False
        self.last_action = None

        info = self.last_info 

        return self._get_obs(), info
    

    def step(self, action):
        """
        Move the agent to a new world position (simulating AirSim position input).
        """

        action = action['action']
        self.step_count += 1
        # self.budget -= 1  # 每一步消耗预算

        # self.interpret_action_to_world(action) ## update new pose as self.agent_pose and self.agent_yaw
        self.grid_mgr.update_grid_map(action)
        obs = self._get_obs()
        total_reward, explore_reward, explore_area = self._compute_reward()
        # done = self.step_count >= self.max_steps
        metrics = self._compute_metrics()
        done = self._check_done()

        info = {
            'reward': {
                'total': total_reward,
                'explore': explore_reward,
                'env': 0
            },
            'success': metrics['success'],
            'coverage': metrics['coverage'],
            'agent_success': 0,
            'distance_to_goal': explore_area
        }
        self.last_info = info
        self.last_action = action

        return obs, total_reward, done, info
    

    def _compute_reward(self):

        # region
        # time_reward = -0.001

        # visited = (self.grid_mgr.grid > 0).sum()
        # explore_area = visited / (self.grid_size * self.grid_size) # 1/3600=0.000278
        # explore_reward = 20 * (explore_area - self.last_visited_grid)
        # self.last_visited_grid = explore_area

        # total_reward = explore_reward + time_reward

        # # dis_agent2goal = np.linalg.norm(
        # #     np.array(self.grid_center_pose_airsim[0:2]) - np.array(self.agent_pos_airsim[:2])
        # # )
        # # print('dis_agent2goal = ', dis_agent2goal)
        # # if dis_agent2goal > self.grid_size/2:
        # #     total_reward += -5

        # grid_x, grid_y = self.grid_mgr.pose_world2grid(self.agent_pos_airsim[:2])
        # print('grid_x, grid_y: ', grid_x, grid_y)

        # grid_boundary = self.grid_size / self.map_resolution
        # if not (0 <= grid_x < grid_boundary and 0 <= grid_y < grid_boundary):
        #     total_reward += -1.0  # 出界严重惩罚

        # endregion

        total_reward = self.time_reward
        explore_reward = 0.0
        # grid_x, grid_y = self.grid_mgr.pose_world2grid(self.agent_pos_airsim[:2])
        grid_x, grid_y, grid_yaw = self.grid_mgr.grid_pos
        
        boundary_small = int(self.patch_size // 2)
        boundary_large = self.grid_size - int(self.patch_size // 2)  # grid size is 0-indexed

        if not (boundary_small <= grid_x <= boundary_large and boundary_small <= grid_y <= boundary_large):
            total_reward += self.out_of_boundary_penalty  # 出界严重惩罚
            self.out_of_boundary = True
        
        else:
            if self.last_action != 0:  # 前进奖励
                total_reward += self.turn_reward

            visited = (self.grid_mgr.grid > 0).sum()
            explore_area = max(0, visited - self.last_visited_grid)
            if explore_area > 0:
                # explore_reward = self.visit_reward * np.clip(explore_area, self.out_of_boundary_penalty, 1.0)
                # explore_reward = np.clip(explore_area, self.visit_reward, 1.0)
                explore_reward = self.visit_reward
                # total_reward += explore_reward * explore_area / (self.patch_size * self.patch_size)  # 探索奖励按比例分配
                total_reward += explore_reward * np.sqrt(explore_area) 
            # else:
                # total_reward += -self.visit_reward # no new explorated area
            
            if visited >= (self.grid_size * self.grid_size) * self.success_threshold:
                total_reward += self.success_reward
                self.done_success = True

            self.last_visited_grid = visited

        # print('reward: ', total_reward, ' explore reward: ', explore_reward)

        return total_reward, explore_reward, self.last_visited_grid / (self.grid_size * self.grid_size)
    

    def _check_done(self, metrics=None):
        done_step = self.step_count >= self.max_steps
        # done_budget = self.budget <= 0
        done_out_of_boundary = self.out_of_boundary
        # print('done_budget: ', done_budget, ' budget: ', self.budget)
        # print('done_step: ', done_step, ' step_count: ', self.step_count)
        # print('done_out_of_boundary: ', done_out_of_boundary)

        # done_success = metrics['success'] if metrics is not None else False
        done_success = self.done_success if metrics is not None else False

        # done = done_step or done_budget or done_out_of_boundary or done_success
        done = done_step or done_out_of_boundary or done_success
        return done
    
    def _compute_metrics(self):
        coverage = self.grid_mgr.grid.sum() / (self.grid_size * self.grid_size)
        success = coverage >= self.success_threshold  # Example threshold for success

        metrics = {
            'coverage': coverage,
            'success': success
        }
        return metrics


    def _get_obs(self):
        # obs = {
        #     'pos': np.array(self.agent_pos, dtype=np.float32)
        #     # 'grid_map': self.grid_mgr.grid.copy()
        # }
        # obs =  [i/self.grid_size for i in list(self.grid_mgr.grid_pos)] + [self.agent_yaw / 180]
        obs = [self.grid_mgr.grid_pos[0] / (self.grid_size - 1),
                self.grid_mgr.grid_pos[1] / (self.grid_size - 1),
                self.grid_mgr.grid_pos[2] / 180]
        obs = np.array(obs, dtype=np.float32)
        # print('obs: ', obs)

        obs = {
            'pos': obs,
            'map': self.grid_mgr.grid.copy()
        }

        return obs
    

        
    
    def render(self, mode='human'):
        """
        Dynamically render the 2D top-down grid with agent position and yaw.
        """

        # 初始化一次性图像窗口
        if not hasattr(self, "_fig"):
            plt.ion()  # 开启交互模式
            self._fig, self._ax = plt.subplots(figsize=(6, 6))

        self._ax.clear()

        # 1. 绘制探索区域（grid）
        grid = self.grid_mgr.grid
        print('grid_shape: ', grid.shape)

        if len(grid.shape)>2:
            grid = grid[0, :, :]  # 取第一个通道

        self._ax.imshow(grid, cmap='gray', origin='lower', extent=[0, self.grid_size, 0, self.grid_size])

        # 2. 坐标转换：世界坐标 -> Grid图像坐标
        # x = self.agent_pos_airsim[0] / self.grid_mgr.resolution + self.grid_size / 2
        # y = self.agent_pos_airsim[1] / self.grid_mgr.resolution + self.grid_size / 2
        x = self.grid_mgr.grid_pos[0]
        y = self.grid_mgr.grid_pos[1]

        # 3. 画出 agent 的朝向（红色箭头）
        # yaw_rad = math.radians(self.agent_yaw)
        yaw_rad = math.radians(self.grid_mgr.grid_pos[2])
        dx = math.cos(yaw_rad)
        dy = math.sin(yaw_rad)
        self._ax.arrow(x+0.5, y+0.5, dx, dy, head_width=1.0, head_length=1.0, fc='r', ec='r')

        # 4. 设置标题与显示区域
        # self._ax.set_title(f"Step {self.step_count} | Pos ({self.agent_pos_airsim[0]:.2f}, {self.agent_pos_airsim[1]:.2f}) | Yaw {self.agent_yaw}°")
        self._ax.set_title(f"Step {self.step_count} | Pos ({x:.2f}, {y:.2f}) | Yaw {round(yaw_rad, 2)}°")
        self._ax.set_xlim(0, self.grid_size)
        self._ax.set_ylim(0, self.grid_size)
        self._ax.set_aspect('equal')

        # 5. 实时更新
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.01)  # 控制刷新速度




class DummyEnv_obstacle_airsimmode_evaluate(DummyEnv):
    def __init__(self, args, grid_size=61, max_steps=400, map_resolution=2.0, patch_size=9):
        '''
        patch: a square area centered at the agent, (meters in real world).
        '''
        super().__init__(grid_size, max_steps, map_resolution, patch_size)

        obstacle_mode = args.obstacle_mode
        self.obstacle_mode = obstacle_mode

        assert args.obstacle_mode == 'airsim'
        if obstacle_mode == 'airsim':
            self.observation_space = spaces.Dict({
                "map": spaces.Box(0, 1, shape=(1, self.grid_size, self.grid_size), dtype=np.float32),
                "pos": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                "depth": spaces.Box(0, 1, shape=(1, 256, 256), dtype=np.float32),  # 假设深度图有 31 个值
            })
                

            self.data_manager = DataLoad_Manager(args.content_dir, args.train_mode)
            self.repeat_episode = False
            # self.data_manager.load_data_all()

            # map_names = self.data_manager.init_map_names(args.scene_names)
            map_names = self.data_manager.init_map_names(args.map_name)
            self.data_manager.load_data_part_maps(map_names)
            self.current_map_name = map_names[0]

            action_mapping_Simenv2Airsimenv = {0: 1,  # 前进
                                           1: 2,  # 左转
                                           2: 3  # 右转
                                           }
            
            self.simulator_manager = Simulator_Manager(
                max_num_step=max_steps, 
                action_mapping_Simenv2Airsimenv=action_mapping_Simenv2Airsimenv)
            

            # self.episode_trakder = EpisodeTracker(worker_id=args.rank, allocation_dir=args.worker_allocation_dir)
            
            self.drone_tool = Drone_tool(drone_config)
            # drone_name = f"Client_{args.rank + 1}"
            self.drone_tool.initialize_client(client_port=args.client_port)
            # self.drone_tool.change_drone_name(drone_name)
            self.obstacle_mgr = ObstacleManager_airsim2D(self.drone_tool)


        self.collide_reward = -1.0  # 碰撞惩罚
        
        # curriculum learning + exploration stagnation
        self.curriculum_stage = 1  
        self.base_success_threshold = 0.5  
        self.set_curriculum_stage(args.train_stage)

        if not args.turn_penalty:
            self.turn_reward = 0

        

    
    def update_stagnation_params(self):

        difficulty_factor = self.success_threshold / self.base_success_threshold
        
        # 基础参数 (较宽松，适合早期训练)
        base_threshold = 80
        base_penalty = -0.3
        base_done_threshold = 150
        
        # 根据难度调整参数 (难度越高，越不容忍停滞)
        self.stagnation_threshold = max(30, int(base_threshold / difficulty_factor))
        self.stagnation_penalty = base_penalty * difficulty_factor
        self.stagnation_done_threshold = max(80, int(base_done_threshold / difficulty_factor))
        self.final_stagnation_penalty = -1.0 * difficulty_factor
        
        

    def set_curriculum_stage(self, stage):

        if stage == 1:
            success_threshold = 0.5
        elif stage == 2:
            success_threshold = 0.6
        elif stage == 3:
            success_threshold = 0.7
        elif stage == 4:
            success_threshold = 0.8
        elif stage == 5:
            success_threshold = 0.9
        elif stage == 6:
            success_threshold = 1.0
        
        elif stage > 6:
            success_threshold = 1.0

        self.curriculum_stage = stage
        self.success_threshold = success_threshold
        self.update_stagnation_params()


    def reset(self):
        """
        Reset the environment. Accepts the drone's world position.
        """

        obs, info = super().reset()
        self.collide = False
        
        # 重置探索停滞相关变量
        self.steps_without_exploration = 0  # 连续没有新探索的步数
        self.done_stagnation = False  # 是否因为探索停滞而done

        return obs, info


    def step(self, action):
        '''add collision'''
        
        # print('in env.py, line522, action: ', action['action'])
        
        if self.obstacle_mode == 'airsim':
            # 在airsim模式下，使用模拟器进行一步操作
            self.simulator_manager.step(action, self.drone_tool)
            time.sleep(0.1)  # 等待模拟器更新

        obs, total_reward, done, info = super().step(action)
        info['collision'] = self.collide
        self.last_info = info

        return obs, total_reward, done, info
    
    
    def _compute_reward(self):
        total_reward = self.time_reward
        explore_reward = 0.0
        stagnation_penalty = 0.0
        # grid_x, grid_y = self.grid_mgr.pose_world2grid(self.agent_pos_airsim[:2])
        grid_x, grid_y, grid_yaw = self.grid_mgr.grid_pos
        
        boundary_small = int(self.patch_size // 2)
        boundary_large = self.grid_size - int(self.patch_size // 2)  # grid size is 0-indexed  

        
        # if self.obstacle_mgr.would_collide(self.grid_mgr.grid_pos, self.last_action):
        #     total_reward += self.collide_reward
        #     self.collide = True 
        
        # collision
        if self.obstacle_mode == '3D':
            if self.obstacle_mgr.would_collide(self.grid_mgr.grid_pos, self.agent_pos_z):
                total_reward += self.collide_reward
                self.collide = True 
                return total_reward, explore_reward, self.last_visited_grid / (self.grid_size * self.grid_size)
        
        elif self.obstacle_mode == '2D':
            if self.obstacle_mgr.would_collide(self.grid_mgr.grid_pos, self.last_action):
                total_reward += self.collide_reward
                self.collide = True 
                return total_reward, explore_reward, self.last_visited_grid / (self.grid_size * self.grid_size)
        
        elif self.obstacle_mode == 'airsim':
            depth_img = self.obstacle_mgr.get_local_depth()
            has_collide = self.obstacle_mgr.would_collide(depth_img)
            if has_collide:
                total_reward += self.collide_reward
                self.collide = True 
                return total_reward, explore_reward, self.last_visited_grid / (self.grid_size * self.grid_size)

        if not (boundary_small <= grid_x <= boundary_large and boundary_small <= grid_y <= boundary_large):
            total_reward += self.out_of_boundary_penalty  # 出界严重惩罚
            self.out_of_boundary = True
        
        else:
            if self.last_action != 0:  # 前进奖励
                total_reward += self.turn_reward

            visited = (self.grid_mgr.grid > 0).sum()
            explore_area = max(0, visited - self.last_visited_grid)
            
            # 探索停滞检测与penalty
            if explore_area > 0:
                # 有新探索，重置停滞计数器
                self.steps_without_exploration = 0
                explore_reward = self.visit_reward
                total_reward += explore_reward * np.sqrt(explore_area) 
            else:
                # 没有新探索，增加停滞计数器
                self.steps_without_exploration += 1
                
                # 如果超过阈值，开始给予停滞penalty
                if self.steps_without_exploration >= self.stagnation_threshold:
                    # 计算停滞penalty (逐渐递增)
                    excess_steps = self.steps_without_exploration - self.stagnation_threshold
                    stagnation_penalty = self.stagnation_penalty * (1 + excess_steps * 0.02)  # 递增2%
                    total_reward += stagnation_penalty
                
                # 如果停滞时间过长，标记为done
                if self.steps_without_exploration >= self.stagnation_done_threshold:
                    total_reward += self.final_stagnation_penalty
                    self.done_stagnation = True
            
            if visited >= (self.grid_size * self.grid_size) * self.success_threshold:
                total_reward += self.success_reward
                self.done_success = True

            self.last_visited_grid = visited

        # print('reward: ', total_reward, ' explore reward: ', explore_reward, 
        #       ' stagnation penalty: ', stagnation_penalty, 
        #       ' steps_no_explore: ', self.steps_without_exploration, 
        #       ' collide:', self.collide)

        return total_reward, explore_reward, self.last_visited_grid / (self.grid_size * self.grid_size)
    
    
    def _check_done(self, metrics=None):
        done = super()._check_done(metrics)
        done = done or self.collide  # 如果发生碰撞，也结束
        done = done or self.done_stagnation  # 如果探索停滞过久，也结束
        return done
    
    def _compute_metrics(self):
        metrics = super()._compute_metrics()
        metrics['collision'] = self.collide
        metrics['stagnation'] = self.done_stagnation
        metrics['steps_without_exploration'] = self.steps_without_exploration
        return metrics
    

    def _get_obs(self):
        obs = super()._get_obs()
        
        if self.obstacle_mode == 'airsim':
            depth = self.obstacle_mgr.get_local_depth()
            h, w, c = depth.shape
            depth = depth.reshape(c, h,w)  # 假设深度图是480x640的
            max_range = 255.0

        # 将depth归一化到[0,1]范围，保留距离信息的相对关系
        # 0表示最近(距离0)，1表示最远(max_range)
        depth_normalized = np.clip(depth / max_range, 0.0, 1.0)
        
        obs['depth'] = np.array(depth_normalized, dtype=np.float32)
        return obs


    def render(self, mode="human"):
        """
        2D俯视图可视化，显示障碍物投影、探索区域和agent位置
        - 背景：白色
        - 比agent低的障碍物：黄色
        - 其余障碍物：橙色
        - 探索过的地方：蓝色
        - agent：带箭头的红色
        """
        import matplotlib.pyplot as plt
        import math
        
        if self.obstacle_mode == 'airsim':
            raise NotImplementedError("Airsim obstacle rendering not implemented in DummyEnv_obstacle")
    
        # ---- 1. 初始化 Figure/Axes ----
        if not hasattr(self, "_fig"):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(8, 8))

        self._ax.clear()
        
        # 设置白色背景
        self._ax.set_facecolor('white')
        
        # ---- 2. 绘制探索区域（蓝色） ----
        grid = self.grid_mgr.grid[0] if self.grid_mgr.grid.ndim > 2 else self.grid_mgr.grid
        
        # 创建探索区域的蓝色遮罩
        explored_mask = grid > 0
        if explored_mask.any():
            # 创建蓝色RGBA图像
            blue_map = np.zeros((self.grid_size, self.grid_size, 4))
            blue_map[explored_mask, :3] = [0.3, 0.6, 1.0]  # 蓝色 RGB
            blue_map[explored_mask, 3] = 0.6  # 透明度
            
            self._ax.imshow(blue_map, origin='lower', 
                           extent=[0, self.grid_size, 0, self.grid_size], 
                           zorder=1)
        
        # ---- 3. 绘制障碍物投影 ----
        if self.obstacle_mode == '3D' and hasattr(self.obstacle_mgr, 'occ') and self.obstacle_mgr.occ is not None:
            agent_z = getattr(self, 'agent_pos_z', 15)  # agent高度，默认15
            self._render_3d_obstacle_projection(agent_z)
        elif self.obstacle_mode == '2D' and hasattr(self.obstacle_mgr, 'occ') and self.obstacle_mgr.occ is not None:
            self._render_2d_obstacles()

        # ---- 4. 绘制 Agent （红色箭头）----
        x, y, yaw_deg = self.grid_mgr.grid_pos
        yaw_rad = math.radians(yaw_deg)
        
        # 绘制agent箭头
        arrow_length = 1.5
        dx = arrow_length * math.cos(yaw_rad)
        dy = arrow_length * math.sin(yaw_rad)
        
        self._ax.arrow(x + 0.5, y + 0.5, dx, dy,
                      head_width=0.8, head_length=0.6,
                      fc='red', ec='red', linewidth=2, zorder=10)
        
        # agent位置用红色圆点标记
        self._ax.scatter([x + 0.5], [y + 0.5], c='red', s=80, zorder=9)

        # ---- 5. 设置标题、坐标轴 & 刷新 ----
        title = f"Step {self.step_count} | Pos({x:.0f},{y:.0f}) | Yaw {yaw_deg:.0f}°"
        if hasattr(self, 'agent_pos_z'):
            title += f" | Height {self.agent_pos_z:.1f}"
        self._ax.set_title(title, fontsize=12)
        
        self._ax.set_xlim(0, self.grid_size)
        self._ax.set_ylim(0, self.grid_size)
        self._ax.set_aspect('equal')
        self._ax.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', 
                   markersize=10, label='nether objects'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                   markersize=10, label='obstacles'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='explored area'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=8, label='Agent')
        ]
        self._ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.01)

        # 可选：若需要 'rgb_array' 模式：
        if mode == "rgb_array":
            w, h = self._fig.canvas.get_width_height()
            return np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    
    def _render_3d_obstacle_projection(self, agent_z):
        """渲染3D障碍物的xy平面投影"""
        # 获取3D障碍物数据
        occ = self.obstacle_mgr.occ  # shape: (Nz, N, N)
        
        # 创建两个投影：比agent低的障碍物 和 其余障碍物
        agent_z_grid = int(round(agent_z))
        
        # 比agent低的障碍物 (黄色)
        low_obstacles = np.any(occ[:agent_z_grid], axis=0)  # z < agent_z
        
        # 其余障碍物 (橙色) - agent高度及以上的障碍物  
        high_obstacles = np.any(occ[agent_z_grid:], axis=0)  # z >= agent_z
        
        # 绘制低障碍物 (黄色)
        if low_obstacles.any():
            yellow_map = np.zeros((self.grid_size, self.grid_size, 4))
            yellow_map[low_obstacles, :3] = [1.0, 1.0, 0.0]  # 黄色 RGB
            yellow_map[low_obstacles, 3] = 0.7  # 透明度
            
            self._ax.imshow(yellow_map, origin='lower',
                           extent=[0, self.grid_size, 0, self.grid_size], 
                           zorder=2)
        
        # 绘制高障碍物 (橙色)
        if high_obstacles.any():
            orange_map = np.zeros((self.grid_size, self.grid_size, 4))
            orange_map[high_obstacles, :3] = [1.0, 0.65, 0.0]  # 橙色 RGB
            orange_map[high_obstacles, 3] = 0.8  # 稍高透明度
            
            self._ax.imshow(orange_map, origin='lower',
                           extent=[0, self.grid_size, 0, self.grid_size], 
                           zorder=3)
    
    def _render_2d_obstacles(self):
        """渲染2D障碍物"""
        occ = self.obstacle_mgr.occ  # shape: (N, N)
        
        # 2D模式下，所有障碍物都显示为橙色
        if occ.any():
            orange_map = np.zeros((self.grid_size, self.grid_size, 4))
            orange_map[occ > 0, :3] = [1.0, 0.65, 0.0]  # 橙色 RGB
            orange_map[occ > 0, 3] = 0.8  # 透明度
            
            self._ax.imshow(orange_map, origin='lower',
                           extent=[0, self.grid_size, 0, self.grid_size], 
                           zorder=2)
            




class DummyEnv_obstacle_airsimmode_evaluate_fly3D(DummyEnv_obstacle_airsimmode_evaluate):
    def __init__(self, args, grid_size=61, max_steps=400, map_resolution=2.0, patch_size=9):
        assert args.env_mode == 'dummy_fly_vertical', "DummyEnv_obstacle_airsimmode_evaluate_fly3D only supports dummy_fly_vertical env mode"
        super().__init__(args, grid_size, max_steps, map_resolution, patch_size)
        print('Training stage: ', args.train_stage)

        ## add additional observation spaces
        self.observation_space.spaces['agent_height'] = spaces.Box(0, 1, (1,), np.float32)
        self.observation_space.spaces['distance_agent2objdown'] = spaces.Box(0, 1, (1,), np.float32)
        self.action_space = spaces.Discrete(5) ## add ascend and descend

        # define parameters related to vertical movement
        self.max_height = 30
        self.target_min_distance = 7.0
        self.target_max_distance = 13.0
        self.optimal_distance = 10.0  # 最优距离
        self.height_reward = 0.03

        self.down_obj_patch_ratio = 4  ## only consider 1/4 area of captured depth image for avg(obj_distance)


    def init_episode_info(self, episode_info):
        self.episode_info = episode_info


    def reset(self):
        self.agent_pos_z = self.episode_info['drone_pose'][2]
        obs, info = super().reset()
        return obs, info


    

    def _compute_height_reward(self, distance_to_obj):
        """
        基于下方物体高度的智能高度奖励计算
        Args:
            distance_to_obj: the distance between agent the downward objects
        """
       
        # 目标距离范围：8-12米
        target_min_distance = self.target_min_distance
        target_max_distance = self.target_max_distance
        optimal_distance = self.optimal_distance
       
        # 基础安全检查奖励/惩罚
        if distance_to_obj < target_min_distance:
            # 太近，给予较大惩罚，鼓励上升
            danger_penalty = -0.1 * (target_min_distance - distance_to_obj) / target_min_distance
            return danger_penalty
        
        elif distance_to_obj > target_max_distance:
            # 太远，给予惩罚，鼓励下降
            far_penalty = -0.05 * (distance_to_obj - target_max_distance) / target_max_distance
            return far_penalty
        
        # 在合理范围内的奖励计算
        elif target_min_distance <= distance_to_obj <= target_max_distance:
            # 在目标范围内，给予正奖励
            # 距离最优距离越近，奖励越高
            distance_from_optimal = abs(distance_to_obj - optimal_distance)
            max_deviation = (target_max_distance - target_min_distance) / 2
            reward_factor = 1.0 - (distance_from_optimal / max_deviation)
            height_reward = self.height_reward * reward_factor
            return height_reward
        

    
    def _extract_distance_agent2objdown(self, depth_downward):
        """
        从向下的深度图中提取下方物体的高度信息
        Args:
            depth_downward: shape (1, H, W) 向下的深度图
        Returns:
            obj_height: 下方物体的高度
        """
        if depth_downward is None:
            return self.agent_pos_z - 10.0  # 默认假设下方10米有物体
        
        # 提取深度图 (H, W)
        depth_img = depth_downward[0] if depth_downward.ndim == 3 else depth_downward
        H, W = depth_img.shape
        
        # 从中心区域提取一个patch (例如中心的1/4区域)
        center_h, center_w = H // 2, W // 2
        patch_size = min(H, W) // self.down_obj_patch_ratio  # patch大小为图像的1/4
        
        patch_start_h = max(0, center_h - patch_size // 2)
        patch_end_h = min(H, center_h + patch_size // 2)
        patch_start_w = max(0, center_w - patch_size // 2)
        patch_end_w = min(W, center_w + patch_size // 2)
        
        # 提取中心patch
        center_patch = depth_img[patch_start_h:patch_end_h, patch_start_w:patch_end_w]
        
        # 计算平均距离（agent到下方物体的距离）
        # 过滤掉过大的值（可能是无效测量）
        valid_depths = center_patch[center_patch < self.max_height]  # 只考虑30米以内的测量
        
        if len(valid_depths) > 0:
            avg_distance_to_obj = float(np.mean(valid_depths))
        else:
            avg_distance_to_obj = self.agent_pos_z  # 默认距离

        return avg_distance_to_obj
    


    def step(self, action):
        '''add vertical movement, where 4 is ascend, 5 is descend'''

        # print('in env.py, line522, action: ', action['action'])

        
        if action['action'] == 4:  # 上升
            self.agent_pos_z += 1.0
        elif action['action'] == 5:  # 下降
            self.agent_pos_z -= 1.0
            if self.agent_pos_z < 0:
                self.collide = True

        obs, total_reward, done, info = super().step(action)

        time.sleep(0.05)  # 等待高度变化生效

        info['collision'] = self.collide
        self.last_info = info

        return obs, total_reward, done, info
    


    def _get_obs(self):
        obs = super()._get_obs()  ## pos, map(grid_map), depth

        if self.obstacle_mode == '3D':
            # get downward depth image from self.obstacle_mgr, 原始深度图用于奖励计算（未归一化）
            depth_downward = self.obstacle_mgr.get_local_depth_downward(self.grid_mgr.grid_pos, self.agent_pos_z)
            distance_agent2objdown = self._extract_distance_agent2objdown(depth_downward)
            self._distance_agent2objdown = distance_agent2objdown
            
        elif self.obstacle_mode == 'airsim':
            #todo: get downward depth image from self.obstacle_mgr
            time.sleep(0.05)  # 等待airsim深度图更新
            depth_downward = self.obstacle_mgr.get_local_depth_downward()
            h, w, c = depth_downward.shape
            depth_downward = depth_downward.reshape(c, h,w)  # get深度图(1*480x640)

            distance_agent2objdown = self._extract_distance_agent2objdown(depth_downward)
            self._distance_agent2objdown = distance_agent2objdown

        obs['distance_agent2objdown'] = np.array([distance_agent2objdown / self.max_height])
        obs['agent_height'] = np.array([self.agent_pos_z / self.max_height])
        return obs





if __name__ == "__main__":
    env = DummyEnv()
    obs = env.reset()  ## yaw=0 towards x positive

   

    returns_list = []
    one_step_reward_list = []
    num_steps_list = []

    for ep in range(200):
        total = 0; obs = env.reset()
        done = False
        num_steps = 0

        while not done:
            action = env.action_space.sample()    # 纯随机
            action = {'action': action}  # 将动作包装成字典
            obs, r, done, _ = env.step(action)
            one_step_reward_list.append(r)
            total += r
            num_steps += 1

        returns_list.append(total)
        num_steps_list.append(num_steps)

    print("episode reward, mean", np.mean(returns_list), 
          "std", np.std(returns_list),
          "max", max(returns_list),
          "min", min(returns_list))

    print('one step reward mean: ', np.mean(one_step_reward_list), 
          'std', np.std(one_step_reward_list), 
          'max', max(one_step_reward_list), 
          'min', min(one_step_reward_list))

    returns_arr = np.array(returns_list, dtype=np.float32)
    norm_returns = (returns_arr - returns_arr.mean()) / (returns_arr.std() + 1e-6)
    print("归一化后方差:", norm_returns.std())

    print('episode steps, mean', np.mean(num_steps_list), 
          'std', np.std(num_steps_list), 
          'max', max(num_steps_list),
          'min', min(num_steps_list))


