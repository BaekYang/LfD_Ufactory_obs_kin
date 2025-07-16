
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Interface for obtaining information
"""

import os
import csv
import sys
import time
import params
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI


#######################################################
"""
Just for test example
"""
# if len(sys.argv) >= 2:
#     ip = "192.168.1.194" #sys.argv[1] #
# else:
#     try:
#         from configparser import ConfigParser
#         parser = ConfigParser()
#         parser.read('../robot.conf')
#         ip = parser.get('xArm', 'ip')
#     except:
#         ip = input('Please input the xArm ip address:')
#         if not ip:
#             print('input error, exit')
#             sys.exit(1)
########################################################

ip = "192.168.1.194"
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(1)
arm.set_state(state=0)
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', action='store', type=str, help='file to run xArm6', required=True)
args = parser.parse_args()


print('=' * 50)
print('version:', arm.get_version())
print('state:', arm.get_state())
print('cmdnum:', arm.get_cmdnum())
print('err_warn_code:', arm.get_err_warn_code())
print('position(°):', arm.get_position(is_radian=False))
print('position(radian):', arm.get_position(is_radian=True))
print('angles(°):', arm.get_servo_angle(is_radian=False))
print('angles(radian):', arm.get_servo_angle(is_radian=True))
print('angles(°)(servo_id=1):', arm.get_servo_angle(servo_id=1, is_radian=False))
print('angles(radian)(servo_id=1):', arm.get_servo_angle(servo_id=1, is_radian=True))

# Function to get current coordinates and angles
def get_robot_state():
    coordinates = arm.get_position(is_radian=False)  # Replace with actual method to get coordinates
    angles = arm.get_servo_angle()      # Replace with actual method to get joint angles
    return coordinates[1], angles[1]
coordinates, angles = get_robot_state()

file_name = args.file_name
file_path = os.path.join('data', file_name if '.csv' in file_name[-4:] else file_name+'.csv')
data=[]

# Open a CSV file to load the data
with open(file_path, 'r') as csvfile:
    fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
    reader = csv.DictReader(csvfile)
    for row in reader:

        data.append({k:float(v) for k, v in row.items()})

for point in data:
    angles = [point['joint1'], point['joint2'], point['joint3'], point['joint4'], point['joint5'], point['joint6']]
    # arm.set_servo_angle(angle=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=True, radius=100.0)
    arm.set_servo_angle_j(angles=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=True, radius=100.0)
    arm.set_suction_cup(point['gripper'])
    time.sleep(0.0001)
    
coordinates, angles = get_robot_state()
print(coordinates, angles)
# arm.set_servo_angle(angle=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=False, radius=0.0)
# print(arm.get_position(), arm.get_position(is_radian=False))

arm.disconnect()