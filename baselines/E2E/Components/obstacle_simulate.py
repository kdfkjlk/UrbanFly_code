import numpy as np
import math

from .Env_airsim.env_utils import get_forward_img_depth, get_downward_img_rgb




class ObstacleManager:
    def __init__(self, grid_size, resolution=1.0, seed=None):
        self.grid_size = grid_size
        self.resolution = resolution
        self.rng = np.random.RandomState(seed)
        self.occ = None   # 0/1 占据栅格

 

    def reset(self, density=0.05,
          min_block=2, max_block=10,
          max_trials=300):
        """
        生成近似 city-block 的障碍：随机矩形 + optional dilation
        """
        self.occ = np.zeros((self.grid_size, self.grid_size), np.uint8)
        target = int(self.grid_size * self.grid_size * density)
        placed = 0

        for _ in range(max_trials):
            w = self.rng.randint(min_block, max_block + 1)
            h = self.rng.randint(min_block, max_block + 1)
            x = self.rng.randint(0, self.grid_size - w)
            y = self.rng.randint(0, self.grid_size - h)

            # 避免 block 之间紧挨（留出街道）
            margin = 1
            xs, xe = max(0, x-margin), min(self.grid_size, x+w+margin)
            ys, ye = max(0, y-margin), min(self.grid_size, y+h+margin)

            if self.occ[ys:ye, xs:xe].any():
                continue

            self.occ[y:y+h, x:x+w] = 1
            placed += w * h
            if placed >= target:
                break

        # 可选：形态学膨胀让障碍更厚
        if self.rng.rand() < 0.2:
            from scipy.ndimage import binary_dilation
            self.occ = binary_dilation(self.occ, iterations=1).astype(np.uint8)



    def would_collide(self, grid_pos, action):
        """预测 '前进/转向' 后的前向 1 cell 是否撞上障碍"""
        x, y, yaw = grid_pos              # yaw ∈ {0,90,180,270}
        # 动作 0=forward, 1=turn-left, 2=turn-right
        if action != 0:                   # 仅前进会产生位移
            return False
        if self.occ is None:              # 添加空值检查
            return False
        dx, dy = ObstacleManager._dir_from_yaw(yaw)
        nx, ny = x + dx, y + dy
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return True  # 环境外也视为碰撞
        return bool(self.occ[ny, nx])


    def get_local_depth(self, grid_pos, yaw, fov_deg=90, max_range=8):
        """
        粗糙模拟 '深度相机'：返回 polar-ray 长度数组，可直接编码成 1-D 深度图
        """
        rays = 31                           # 奇数 → 中线对齐
        half_fov = math.radians(fov_deg / 2)
        depths = np.full(rays, max_range, dtype=np.float32)

        if self.occ is None:                # 添加空值检查
            return depths

        cx, cy, _ = grid_pos
        for i, theta in enumerate(np.linspace(-half_fov, half_fov, rays)):
            global_theta = math.radians(yaw) + theta
            dx, dy = math.cos(global_theta), math.sin(global_theta)
            # DDA / Bresenham 简化：步长 < 1 cell
            for r in np.linspace(0.5, max_range, int(max_range * 2)):
                gx, gy = int(round(cx + dx * r)), int(round(cy + dy * r))
                if not (0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
                    depths[i] = r
                    break
                if self.occ[gy, gx]:
                    depths[i] = r
                    break
        return depths  # shape=[31]

    def render(self, ax, **plt_kwargs):
        if self.occ is not None:
            ax.imshow(self.occ, cmap='Reds', alpha=0.4, origin='lower',
                      extent=[0, self.grid_size, 0, self.grid_size], **plt_kwargs)

    # ---------- util ----------
    @staticmethod
    def _dir_from_yaw(yaw_deg):
        # yaw==0 ➜ +x；90 ➜ +y；顺时针
        dirs = {0: (1, 0), 90: (0, 1), -180: (-1, 0), -90: (0, -1)}
        return dirs[int(yaw_deg)]



class ObstacleManager3D:
    def __init__(self, grid_xy, grid_z=30, res=1.0, seed=None):
        self.N = grid_xy
        self.Nz = grid_z
        self.res = res
        self.rng = np.random.RandomState(seed)
        self.occ = None  # shape [Nz, N, N]

    # ---------- reset ----------
    def reset(self, density=0.05, min_block=2, max_block=10,
              min_h=2, max_h=30):
        self.occ = np.zeros((self.Nz, self.N, self.N), np.uint8)
        target = int(self.N * self.N * density)

        placed = 0
        while placed < target:
            w, d = self.rng.randint(min_block, max_block+1, size=2)
            h = self.rng.randint(min_h, max_h+1)
            x = self.rng.randint(0, self.N - w)
            y = self.rng.randint(0, self.N - d)

            if self.occ[:h, y:y+d, x:x+w].any():       # 重叠检查
                continue
            self.occ[:h, y:y+d, x:x+w] = 1
            placed += w * d

    # ---------- 深度图 ----------
    def get_local_depth(self, pos_grid, pos_z, pitch_deg=0,
                        H=64, W=64, fov=90, max_range=15):
        """向量化优化的深度图生成"""
        if self.occ is None:
            return np.full((1, H, W), max_range, np.float32)
            
        cx, cy, yaw_deg = pos_grid
        cz = pos_z
        yaw, pitch = map(math.radians, (yaw_deg, pitch_deg))
        
        # 1. 批量生成所有像素的射线方向
        K = self._build_K(W, H, fov)
        inv_K = np.linalg.inv(K)
        
        # 生成所有像素坐标的网格
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
        pixel_coords = np.stack([u_coords.ravel(), v_coords.ravel(), 
                                np.ones(H*W)], axis=0)  # shape: (3, H*W)
        
        # 批量转换到相机坐标系
        dirs_cam = inv_K @ pixel_coords  # shape: (3, H*W)
        dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=0, keepdims=True)
        
        # 批量应用旋转变换
        dirs_world = self._cam2world_vectorized(dirs_cam, yaw, pitch)  # shape: (3, H*W)
        
        # 2. 向量化射线追踪
        depth = self._vectorized_raycast(cx, cy, cz, dirs_world, max_range, H, W)
        
        return depth.reshape(1, H, W)
        

    def get_local_depth_downward(self, pos_grid, pos_z, H=64, W=64, fov=90, max_range=15):
        """
        生成向下的深度图（downward-facing camera）
        Args:
            pos_grid: (x, y, yaw) agent在xy平面的位置和朝向
            pos_z: agent的z坐标（高度）
            H, W: 深度图的高度和宽度，与forward-facing camera保持一致
            fov: 视场角（度），与forward-facing camera保持一致
            max_range: 最大深度范围，与forward-facing camera保持一致
        Returns:
            depth: shape=(1, H, W) 向下的深度图
        """
        if self.occ is None:
            return np.full((1, H, W), max_range, np.float32)
        
        cx, cy, yaw_deg = pos_grid
        cz = pos_z
        yaw = math.radians(yaw_deg)
        # 向下相机的pitch角度为-90度（向下看）
        pitch = math.radians(-90)
        
        # 1. 批量生成所有像素的射线方向
        K = self._build_K(W, H, fov)
        inv_K = np.linalg.inv(K)
        
        # 生成所有像素坐标的网格
        u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
        pixel_coords = np.stack([u_coords.ravel(), v_coords.ravel(), 
                                np.ones(H*W)], axis=0)  # shape: (3, H*W)
        
        # 批量转换到相机坐标系
        dirs_cam = inv_K @ pixel_coords  # shape: (3, H*W)
        dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=0, keepdims=True)
        
        # 批量应用旋转变换（向下看）
        dirs_world = self._cam2world_vectorized(dirs_cam, yaw, pitch)  # shape: (3, H*W)
        
        # 2. 向量化射线追踪
        depth = self._vectorized_raycast(cx, cy, cz, dirs_world, max_range, H, W)
        
        return depth.reshape(1, H, W)


    def _cam2world_vectorized(self, dirs_cam, yaw, pitch):
        """向量化的坐标变换"""
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        # 构建旋转矩阵
        R_pitch = np.array([[1, 0, 0],
                           [0, cos_pitch, -sin_pitch],
                           [0, sin_pitch, cos_pitch]], dtype=np.float32)
        
        R_yaw = np.array([[cos_yaw, -sin_yaw, 0],
                         [sin_yaw, cos_yaw, 0],
                         [0, 0, 1]], dtype=np.float32)
        
        R = R_yaw @ R_pitch
        return R @ dirs_cam

    def _vectorized_raycast(self, cx, cy, cz, dirs_world, max_range, H, W):
        """向量化射线追踪 - 大幅提升性能"""
        if self.occ is None:
            return np.full((H, W), max_range, dtype=np.float32)
            
        depth = np.full(H*W, max_range, dtype=np.float32)
        active_rays = np.ones(H*W, dtype=bool)  # 跟踪仍在活跃的射线
        
        # 步长设置 - 平衡精度和性能
        step_size = 0.5
        max_steps = int(max_range / step_size)
        
        # 批量计算所有射线的起始点和方向
        start_points = np.array([cx, cy, cz], dtype=np.float32).reshape(3, 1)
        start_points = np.broadcast_to(start_points, (3, H*W))  # shape: (3, H*W)
        
        # 逐步推进射线
        for step in range(1, max_steps + 1):
            # 早期终止：如果所有射线都已找到深度，提前退出
            if not active_rays.any():
                break
                
            r = step * step_size
            
            # 只处理仍在活跃的射线
            active_indices = np.where(active_rays)[0]
            if len(active_indices) == 0:
                break
                
            # 计算当前步骤活跃射线的位置
            current_pos = start_points[:, active_indices] + r * dirs_world[:, active_indices]
            
            # 转换为网格坐标
            gx = np.round(current_pos[0]).astype(np.int32)
            gy = np.round(current_pos[1]).astype(np.int32)
            gz = np.round(current_pos[2]).astype(np.int32)
            
            # 检查边界
            valid_mask = ((gx >= 0) & (gx < self.N) & 
                         (gy >= 0) & (gy < self.N) & 
                         (gz >= 0) & (gz < self.Nz))
            
            # 对超出边界的射线设置深度并标记为非活跃
            out_of_bounds_local = ~valid_mask
            if out_of_bounds_local.any():
                out_of_bounds_global = active_indices[out_of_bounds_local]
                depth[out_of_bounds_global] = r
                active_rays[out_of_bounds_global] = False
            
            # 检查障碍物碰撞（只对仍在边界内的射线）
            if valid_mask.any():
                valid_local_indices = np.where(valid_mask)[0]
                valid_global_indices = active_indices[valid_local_indices]
                valid_gx = gx[valid_local_indices]
                valid_gy = gy[valid_local_indices] 
                valid_gz = gz[valid_local_indices]
                
                # 批量查询障碍物
                hit_obstacles = self.occ[valid_gz, valid_gy, valid_gx].astype(bool)
                
                # 对碰撞到障碍物的射线设置深度并标记为非活跃
                if hit_obstacles.any():
                    hit_global_indices = valid_global_indices[hit_obstacles]
                    depth[hit_global_indices] = r
                    active_rays[hit_global_indices] = False
        
        return depth.reshape(H, W)

    # ---------- util ----------
    def _build_K(self, W, H, fov_deg):
        """构建相机内参矩阵"""
        fov_rad = math.radians(fov_deg)
        fx = fy = W / (2.0 * math.tan(fov_rad / 2.0))
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        return K

    def _cam2world(self, dir_cam, yaw, pitch):
        """将相机坐标系方向向量转换到世界坐标系"""
        # 相机坐标系: x右，y下，z前
        # 世界坐标系: x右，y前，z上
        
        # 首先应用pitch旋转（绕x轴）
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
        R_pitch = np.array([[1, 0, 0],
                           [0, cos_pitch, -sin_pitch],
                           [0, sin_pitch, cos_pitch]])
        
        # 然后应用yaw旋转（绕z轴）
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        R_yaw = np.array([[cos_yaw, -sin_yaw, 0],
                         [sin_yaw, cos_yaw, 0],
                         [0, 0, 1]])
        
        # 组合旋转矩阵
        R = R_yaw @ R_pitch
        
        # 应用旋转
        dir_world = R @ dir_cam
        return dir_world[0], dir_world[1], dir_world[2]

    def _dir_from_yaw(self, yaw_deg):
        # yaw==0 ➜ +x；90 ➜ +y；顺时针
        dirs = {0: (1, 0), 90: (0, 1), -180: (-1, 0), -90: (0, -1)}
        return dirs[int(yaw_deg)]

    def would_collide(self, pos_xy, pos_z, collision_threshold=1.0, collision_ratio=0.1):
        """
        基于深度图像的碰撞检测
        Args:
            pos_xy: (x, y, yaw) agent在xy平面的位置和朝向
            pos_z: agent的z坐标（高度）
            collision_threshold: 碰撞距离阈值（grid单位），默认1.0
            collision_ratio: 触发碰撞的像素比例阈值，默认0.1 (10%)
        Returns:
            bool: True表示发生碰撞
        """
        if self.occ is None:
            return False

        x, y, yaw = pos_xy
        
        # 边界检查 - 如果agent在环境边界外，视为碰撞
        if not (0 <= x < self.N and 0 <= y < self.N and 0 <= pos_z < self.Nz):
            return True
            
        # 获取当前位置的深度图像
        depth_image = self.get_local_depth(pos_xy, pos_z, pitch_deg=0, 
                                          H=32, W=32, fov=60, max_range=5)
        
        # 深度图像shape为(1, H, W)，需要取出(H, W)
        if depth_image.ndim == 3:
            depth_image = depth_image[0]  # shape: (H, W)
        
        # 计算距离小于阈值的像素数量
        close_pixels = depth_image < collision_threshold
        close_pixel_count = np.sum(close_pixels)
        total_pixels = depth_image.size
        
        # 计算比例
        close_ratio = close_pixel_count / total_pixels
        
        # 如果超过阈值比例的像素距离过近，则判定为碰撞
        return close_ratio > collision_ratio
    

    def render(self, ax, agent_z=None, **plt_kwargs):
        """
        渲染3D障碍物地图的俯视图
        - 全部3D障碍物投影到xy平面：灰色
        - agent高度上下一个grid中的障碍物：黑色
        """
        if self.occ is None:
            return
            
        # 1. 将所有3D障碍物投影到2D平面（俯视图），标记为灰色
        top_view_all = np.any(self.occ, axis=0)  # 沿z轴投影，shape=(N, N)
        
        if top_view_all.any():
            # 创建灰色障碍物图层
            gray_obstacles = np.zeros((self.N, self.N, 4))  # RGBA
            gray_obstacles[top_view_all, :3] = 0.5  # 灰色 (RGB = 0.5, 0.5, 0.5)
            gray_obstacles[top_view_all, 3] = 0.4   # 透明度
            
            ax.imshow(gray_obstacles, origin='lower',
                      extent=[0, self.N, 0, self.N], **plt_kwargs)
        
        # 2. 如果提供了agent高度，则突出显示agent附近的障碍物为黑色
        if agent_z is not None:
            agent_z_grid = int(round(agent_z))
            z_min = max(0, agent_z_grid - 1)
            z_max = min(self.Nz, agent_z_grid + 2)  # +2因为是半开区间
            
            # agent高度附近的障碍物
            nearby_obstacles = np.any(self.occ[z_min:z_max], axis=0)
            
            if nearby_obstacles.any():
                # 创建黑色障碍物图层
                black_obstacles = np.zeros((self.N, self.N, 4))  # RGBA
                black_obstacles[nearby_obstacles, :3] = 0.0  # 黑色 (RGB = 0, 0, 0)
                black_obstacles[nearby_obstacles, 3] = 0.6   # 稍高透明度以突出显示
                
                ax.imshow(black_obstacles, origin='lower',
                          extent=[0, self.N, 0, self.N], **plt_kwargs)





class ObstacleManager_airsim2D:
    def __init__(self, drone_tool):
        self.drone_tool = drone_tool

    def get_local_depth(self):
        """
        Get forward depth image from the drone.
        """
        depth = get_forward_img_depth(self.drone_tool)
        # obs = depth.astype(np.float32) / 255.0
        obs = depth.astype(np.float32)
        return obs
    
    def get_local_depth_downward(self):
        """
        Get downward RGB image from the drone.
        """
        depth = get_downward_img_rgb(self.drone_tool)
        obs = depth.astype(np.float32)
        return obs
    
    def would_collide(self, depth_image, collision_threshold=1.0, collision_ratio=0.1):
        """
        基于深度图像的碰撞检测
        Args:
            depth_img
            collision_threshold: 碰撞距离阈值（grid单位），默认1.0
            collision_ratio: 触发碰撞的像素比例阈值，默认0.1 (10%)
        Returns:
            bool: True表示发生碰撞
        """
        
        # 深度图像shape为(1, H, W)，需要取出(H, W)
        if depth_image.ndim == 3:
            depth_image = depth_image[0]  # shape: (H, W)
        
        # 计算距离小于阈值的像素数量
        close_pixels = depth_image < collision_threshold
        close_pixel_count = np.sum(close_pixels)
        total_pixels = depth_image.size
        
        # 计算比例
        close_ratio = close_pixel_count / total_pixels
        
        # 如果超过阈值比例的像素距离过近，则判定为碰撞
        return close_ratio > collision_ratio
        



