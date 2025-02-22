"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import reactive_obst_avoid
from control import potential_field_control


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        self._size_area = (800, 800)
        self.tiny_slam = TinySlam(x_min=- self._size_area[0],
                                  x_max=self._size_area[0],
                                  y_min=- self._size_area[1],
                                  y_max=self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """
        self.counter += 1

        #self.tiny_slam.compute()

        '''
        TP3
        if self.counter % 10 == 0:
            self.tiny_slam.update_map(self.lidar(), self.odometer_values())
        '''

        if self.counter == 1:
            self.tiny_slam.update_map(self.lidar(), self.odometer_values())
        if self.counter % 20 == 0:
            score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
            if score >= 50:
                self.tiny_slam.update_map(self.lidar(), self.odometer_values())
                print("Map updated: score", score)

        if self.counter % 1000 == 0:
            start = self.tiny_slam.get_corrected_pose(self.odometer_values())
            start = self.tiny_slam._conv_world_to_map(start[0], start[1])
            goal = self.tiny_slam._conv_world_to_map(0, 0)
            print("Target map value: ", self.tiny_slam.occupancy_map[goal[0], goal[1]])
            print(start, goal)
            route = self.tiny_slam.plan(start, goal)
            print(route)
        
        self.tiny_slam.display2(self.odometer_values())

        # Compute new command speed to perform obstacle avoidance
        if self.tiny_slam.path is None:
            command = reactive_obst_avoid(self.lidar())
        else:
            aim = self.tiny_slam._conv_map_to_world(self.tiny_slam.path[0, 0], self.tiny_slam.path[0, 1])
            command = potential_field_control(self.lidar(), self.odometer_values(), [aim[0], aim[1], 1])
            if self.counter % 10 == 0:
                if self.tiny_slam.path.shape[0] == 1:
                    self.tiny_slam.path = None
                else:
                    self.tiny_slam.path = self.tiny_slam.path[1:]

        return command
