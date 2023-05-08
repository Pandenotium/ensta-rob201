""" A simple robotics navigation code including SLAM, exploration, planning"""

import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self._conv_world_to_map(
            self.x_max_world, self.y_max_world)

        self.occupancy_map = np.zeros(
            (int(self.x_max_map), int(self.y_max_map)))

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

        # TP6
        self.path = None

    def _conv_world_to_map(self, x_world, y_world):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """
        x_map = (x_world - self.x_min_world) / self.resolution
        y_map = (y_world - self.y_min_world) / self.resolution

        if isinstance(x_map, float):
            x_map = int(x_map)
            y_map = int(y_map)
        elif isinstance(x_map, np.ndarray):
            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

        return x_map, y_map

    def _conv_map_to_world(self, x_map, y_map):
        """
        Convert from map coordinates to world coordinates
        x_map, y_map : list of x and y coordinates in cell numbers (~pixels)
        """
        x_world = self.x_min_world + x_map *  self.resolution
        y_world = self.y_min_world + y_map *  self.resolution

        if isinstance(x_world, np.ndarray):
            x_world = x_world.astype(float)
            y_world = y_world.astype(float)

        return x_world, y_world

    def add_map_line(self, x_0, y_0, x_1, y_1, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        """

        # convert to pixels
        x_start, y_start = self._conv_world_to_map(x_0, y_0)
        x_end, y_end = self._conv_world_to_map(x_1, y_1)

        if x_start < 0 or x_start >= self.x_max_map or y_start < 0 or y_start >= self.y_max_map:
            return

        if x_end < 0 or x_end >= self.x_max_map or y_end < 0 or y_end >= self.y_max_map:
            return

        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start  # recalculate differentials
        dy = y_end - y_start  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        points = np.array(points).T

        # add value to the points
        self.occupancy_map[points[0], points[1]] += val

    def add_map_points(self, points_x, points_y, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """
        x_px, y_px = self._conv_world_to_map(points_x, points_y)

        select = np.logical_and(np.logical_and(x_px >= 0, x_px < self.x_max_map),
                                np.logical_and(y_px >= 0, y_px < self.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        self.occupancy_map[x_px, y_px] += val


    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        score = 0
        dis = lidar.get_sensor_values()
        ang = lidar.get_ray_angles()

        # remove the points that exceed the range of lidar
        reserve = dis < lidar.max_range
        dis = dis[reserve]
        ang = ang[reserve]
        r = np.stack([dis, ang], 1)

        # calculate points at the odom rep
        points = np.empty_like(r)
        points[:, 0] = pose[0] + np.cos(r[:, 1] + pose[2]) * r[:, 0]
        points[:, 1] = pose[1] + np.sin(r[:, 1] + pose[2]) * r[:, 0]

        # transform the coord world to map
        points_map_x, points_map_y = self._conv_world_to_map(points[:, 0], points[:, 1])
        points_map = np.stack([points_map_x, points_map_y], 1)

        # reserve the coords in the map range
        x_reserve = points_map[:, 0] < self.occupancy_map.shape[0]
        y_reserve = points_map[:, 1] < self.occupancy_map.shape[1]
        reserve = x_reserve * y_reserve
        points_map = points_map[reserve]

        # sum the score
        score += np.sum(self.occupancy_map[points_map[:, 0], points_map[:, 1]])

        return score

    def get_corrected_pose(self, odom, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
        odom_pose = odom + odom_pose_ref
        corrected_pose = odom_pose

        return corrected_pose

    def localise(self, lidar, odom):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4
        max_score = self.score(lidar, self.get_corrected_pose(odom, self.odom_pose_ref))
        max_pos = self.odom_pose_ref

        N = 500
        sigma = 3
        sigma_a = 0.1

        # loop N times
        for i in range(N):
            # generate random pos
            rand_pos = np.random.normal(0, sigma, 2)
            rand_ang = np.random.normal(0, sigma_a, 1)
            t_pos = max_pos
            t_pos[0] += rand_pos[0]
            t_pos[1] += rand_pos[1]
            t_pos[2] += rand_ang
            # calculate the score corresponding to the random pos
            score = self.score(lidar, self.get_corrected_pose(odom, t_pos))
            # if better we keep the new pos
            if score >= max_score:
                max_score = score
                max_pos = t_pos

        if score > 200:
            self.odom_pose_ref = max_pos

        return max_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        dis = lidar.get_sensor_values()
        ang = lidar.get_ray_angles()
        lidar_corr = []

        for i in range(len(dis)):
            lidar_corr.append([pose[0] + np.cos(ang[i] + pose[2]) * dis[i], pose[1] + np.sin(ang[i] + pose[2]) * dis[i]])

        lidar_corr = np.array(lidar_corr)
        for item in lidar_corr:
            dis_l = np.sqrt((item[0] - pose[0])**2 + (item[1] - pose[1])**2)
            self.add_map_line(pose[0], pose[1], item[0] - 10*(item[0] - pose[0])/dis_l, item[1] - 10*(item[1]- pose[1])/dis_l, -1)
        self.add_map_points(lidar_corr[:,0], lidar_corr[:,1], 1)
        self.occupancy_map[self.occupancy_map <= -1.99] = -1.99
        self.occupancy_map[self.occupancy_map >= 1.99] = 1.99


    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        """
        # TODO for TP5

        # initialization
        openSet = [start]
        #closeSet = []
        cameFrom = {}
        gScore = np.full_like(self.occupancy_map, np.inf)
        gScore[start] = 0
        fScore = np.full_like(self.occupancy_map, np.inf)
        fScore[start] = np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)

        while openSet:
            t_openSet = np.array(openSet)
            t_fScore = fScore[t_openSet[:,0], t_openSet[:,1]]
            ind_cur = np.argmin(t_fScore)
            current = openSet[ind_cur]
            #print(current)
            #self.occupancy_map[current[0], current[1]] = 1.99
            #self.display2(self.odom_pose_ref)

            # if found the path return
            if current[0] == goal[0] and current[1] == goal[1]:
                total_path = [current]
                ind = current[0] * self.occupancy_map.shape[0] + current[1]
                while ind in cameFrom.keys():
                    current = cameFrom[ind]
                    total_path.append(current)
                    ind = current[0] * self.occupancy_map.shape[0] + current[1]
                total_path.reverse()
                self.path = np.array(total_path)
                return total_path
            del openSet[ind_cur]
            #closeSet.append(current)
            
            # neighbours
            neighbours = [[current[0] - 1, current[1] - 1], [current[0] - 1, current[1]], [current[0] - 1, current[1] + 1],
                          [current[0], current[1] - 1], [current[0], current[1] + 1],
                          [current[0] + 1, current[1] - 1], [current[0] + 1, current[1]], [current[0] + 1, current[1] + 1]]
            for n in neighbours:
                if self.occupancy_map[n[0], n[1]] > 1:
                    continue
                #if n in closeSet:
                #    continue
                t_gScore = gScore[current[0], current[1]] + np.sqrt((n[0]-current[0])**2 + (n[1]-current[1])**2)
                if t_gScore < gScore[n[0], n[1]]:
                    cameFrom[n[0]*self.occupancy_map.shape[0] + n[1]] = current
                    gScore[n[0], n[1]] = t_gScore
                    fScore[n[0], n[1]] = t_gScore + np.sqrt((goal[0]-n[0])**2 + (goal[1]-n[1])**2)
                    if n not in openSet:
                        openSet.append(n)
        return None

        path = [start, goal]  # list of poses
        return path

    def display(self, robot_pose):
        """
        Screen display of map and robot pose, using matplotlib
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        plt.cla()
        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world, self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")

        delta_x = np.cos(robot_pose[2]) * 10
        delta_y = np.sin(robot_pose[2]) * 10
        plt.arrow(robot_pose[0], robot_pose[1], delta_x, delta_y,
                  color='red', head_width=5, head_length=10, )

        # plt.show()
        plt.pause(0.001)

    def display2(self, robot_pose):
        """
        Screen display of map and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        disp_map = self.occupancy_map
        if self.path is not None:
            disp_map[self.path[:,0], self.path[:,1]] = 1.99

        img = cv2.flip(disp_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img2 = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)

        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        # print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        cv2.arrowedLine(img=img2, pt1=pt1, pt2=pt2,
                        color=(0, 0, 255), thickness=2)
        cv2.imshow("map slam", img2)
        cv2.waitKey(1)

    def save(self, filename):
        """
        Save map as image and pickle object
        filename : base name (without extension) of file on disk
        """

        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world,
                           self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")
        plt.savefig(filename + '.png')

        with open(filename + ".p", "wb") as fid:
            pickle.dump({'occupancy_map': self.occupancy_map,
                         'resolution': self.resolution,
                         'x_min_world': self.x_min_world,
                         'x_max_world': self.x_max_world,
                         'y_min_world': self.y_min_world,
                         'y_max_world': self.y_max_world}, fid)

    def load(self, filename):
        """
        Load map from pickle object
        filename : base name (without extension) of file on disk
        """
        # TODO

    '''
    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi,np.pi,np.pi/1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x,pt_y])
    '''
