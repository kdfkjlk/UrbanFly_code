import numpy as np



class GridMapManager:
    def __init__(self, grid_size=61, resolution=1.0, patch_size=5):
        """
        Initialize the grid map manager.

        Args:
        - grid_size: Number of grid cells (width and height)
        - resolution: Size of one grid cell in meters (AirSim world unit to grid)
        - patch_size: the area (small box with width=height=patch_size) centered at current_grid_pose
        """
        assert grid_size > 0 and isinstance(grid_size, int), "Grid size must be a positive integer"
        self.grid_size = grid_size  ## meters in real world
        self.resolution = resolution # meters/grid
        self.patch_size = patch_size

        # dx, dy, d_yaw
        # self.movement = {
        #     0: [1, 0, 0], # forward
        #     1: [0, 0, -90], # turn left
        #     2: [0, 0, 90] # turn right
        # }
        self.movement = {
            0: [1, 0],     # forward（走一步）: 向前走一格，方向由当前 yaw 决定
            1: -90,        # turn left
            2: 90          # turn right
        }

        self.reset()


    def reset(self):
        """
        Reset the grid to an empty state.
        """
        self.grid_pos = [
            int(self.grid_size / 2), 
            int(self.grid_size / 2),
            0
        ]  # start at grid center

        self.grid = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        self.grid = self.update_grid_map()
        return self.grid

   


    def update_current_pose(self, action):
        """
        Convert AirSim world coordinates to grid indices.
        return: self.grid_pos: relative pose to world_origin_pose, range from grid left to right, up to bottom
        """
        
        x, y, yaw = self.grid_pos

        if action == 0:
            # 执行前进动作，方向由当前 yaw 决定
            rad = np.deg2rad(yaw)
            dx = round(np.cos(rad))     # 四向动作，取整
            dy = round(np.sin(rad))
            x += dx
            y += dy

        elif action in [1, 2]:
            # 执行旋转动作
            delta_yaw = self.movement[action]
            yaw += delta_yaw

            if yaw < -180:
                yaw += 360
            elif yaw >= 180:
                yaw -= 360

        self.grid_pos = [x, y, yaw]
        return self.grid_pos


    def update_grid_map(self, action=None):
        """
        Update the grid with a fixed square observation patch centered at drone_world_pos.

        Parameters:
        - drone_world_pos: (x, y) in AirSim world coordinates; if None, use internal drone_pos
        - patch_size: int, must be odd
        """

        if action is not None:
            grid_pose = self.update_current_pose(action)
        else:
            grid_pose = self.grid_pos
        # print('grid_pose', grid_pose)

        x, y, yaw = grid_pose
        H, W = self.grid.shape[1], self.grid.shape[2]
        half = max(self.patch_size // 2, 1)

        if not (0 <= x < W and 0 <= y < H):
            # print("Grid index out of bounds, skipping update.")
            return self.grid

        xmin = max(x - half, 0)
        xmax = min(x + half + 1, W)
        ymin = max(y - half, 0)
        ymax = min(y + half + 1, H)

        self.grid[:, ymin:ymax, xmin:xmax] = 1.0

        return self.grid


if __name__ == "__main__":
    mgr = GridMapManager(grid_size=9, resolution=1.0, patch_size=3)
    # mgr.set_origin_world((5.0, 5.0))  # world origin at grid center

    grid = mgr.update_grid_map(action=None)
    print(grid)


    # pos1 = (9,8)  # center
    # grid = mgr.update_grid_map(pos1)
    # print(grid)
