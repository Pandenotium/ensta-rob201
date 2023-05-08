""" A set of robotics control functions """
import random
import numpy as np

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1
    dis = lidar.get_sensor_values()[[135, 180, 225]]
    #pos = self.odometer_values() 
    if (min(dis) <= 30):
        command = {"forward": 0,
               "rotation": random.uniform(0,1)}
    else:
        command = {"forward": 0.1,
                "rotation": 0}

    return command

def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """
   # TODO for TP2

    K_a = 0.5
    dx = goal[0] - pose[0]
    dy = goal[1] - pose[1]
    dis = np.sqrt(dx**2 + dy**2)
    f_attire = [0, 0]
    f_attire[0] = K_a / dis * dx
    f_attire[1] = K_a / dis * dy

    #d_angle = np.arctan2(dy, dx) - pose[2]
    #print(dis)

    K_p = 10
    d_safe = 50
    imin = np.argmin(lidar.get_sensor_values())
    d_obs = lidar.get_sensor_values()[imin]
    #print(d_obs)
    a_obs = lidar.get_ray_angles()[imin]
    if (d_obs > d_safe):
        f_repousse = [0, 0]
    else:
        f_r = K_p / d_obs**3 * (1 / d_obs + 1 / d_safe) * d_obs
        f_repousse = [-f_r * np.cos(a_obs + pose[2]), -f_r * np.sin(a_obs + pose[2])]

    force = [f_attire[0] + f_repousse[0], f_attire[1] + f_repousse[1]]

    d_angle = np.arctan2(force[1], force[0]) - pose[2]
    #print(d_angle)
    #print(dis)
    
    if np.abs(d_angle) > 0.05:
        #command = {"forward": 0,
               #"rotation": 0.1*d_angle}
        command = {"forward": 0,
               "rotation": 0.3*np.sign(d_angle)}
    elif dis < 1:
        '''
        d_angle2 = pose[2] - goal[2]
        if np.abs(d_angle2) > 0.05:
            command = {"forward": 0,
               "rotation": 0.1*d_angle2}
        else:
        '''
        command = {"forward": 0,
               "rotation": 0}
    else:
        command = {"forward": np.sqrt(force[0]**2 + force[1]**2)/(5 + np.sqrt(force[0]**2 + force[1]**2)),
               "rotation": 0}
        #command = {"forward": 0.7,
        #       "rotation": 0}
               

    return command
