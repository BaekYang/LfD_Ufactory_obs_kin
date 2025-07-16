import os
import csv
import sys
import time
import params
import argparse
import keyboard  # 키 입력 감지를 위한 라이브러리

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

# 로봇 상태(좌표, 각도) 확인 함수
def get_robot_state():
    coordinates = arm.get_position(is_radian=False)  # 좌표
    angles = arm.get_servo_angle()                   # 관절 각도
    return coordinates[1], angles[1]

# (x, y, z, roll, pitch, yaw)를 관절 각도로 변환 (역기구학)
def calculate_joint_angles(x, y, z, roll, pitch, yaw):
    pose = [x, y, z, roll, pitch, yaw]
    code, angles = arm.get_inverse_kinematics(pose, input_is_radian=False, return_is_radian=False)
    if code == 0:
        return angles
    else:
        return None
ip = "192.168.1.194"
arm = XArmAPI(ip) 
arm.motion_enable(enable=True)
arm.set_mode(1)
arm.set_state(state=0)

# 명령줄 인자 설정
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', action='store', type=str, help='file to run xArm6', required=True)
args = parser.parse_args()

# CSV 파일 경로
file_name = args.file_name
file_path = os.path.join('data', file_name if '.csv' in file_name[-4:] else file_name + '.csv')

data = []
# CSV 파일 읽기
with open(file_path, 'r') as csvfile:
    fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            data.append({k: float(v) for k, v in row.items()})
        except:
            print(row)

# 첫 번째 타임스탬프를 기준으로 시작 시간 설정
start_time = None  # 스페이스바가 처음 눌린 시간을 저장
paused_time = 0    # 일시 정지 시간 보정
paused_start = None

# CSV 데이터를 순회하면서 로봇 제어
for point in data:
    if start_time is None:
        # 스페이스바가 처음 눌린 순간을 기준으로 시간 설정
        while not keyboard.is_pressed('space'):
            time.sleep(0.01)  # 대기
        start_time = time.perf_counter()  # 현재 시간을 0초로 설정

    # 스페이스바를 누르지 않으면 멈춤
    while not keyboard.is_pressed('space'):
        if paused_start is None:
            paused_start = time.perf_counter()  # 멈춘 시간 기록
        time.sleep(0.01)

    # 멈춘 시간만큼 보정 (다시 누르면 정상 시간 흐름 유지)
    if paused_start is not None:
        paused_time += time.perf_counter() - paused_start
        paused_start = None  # 보정 후 초기화

    # 로봇이 실행해야 할 정확한 시간 계산
    current_time = time.perf_counter() - start_time - paused_time
    target_time = (point['timestamp'] - data[0]['timestamp']) / 1000.0  # CSV 기준 초 단위 변환

    # 현재 시간이 목표 시간보다 앞설 경우 대기
    while current_time < target_time:
        time.sleep(0.001)
        current_time = time.perf_counter() - start_time - paused_time

    x, y, z = point['x'], point['y'], point['z']
    roll, pitch, yaw = point['roll'], point['pitch'], point['yaw']
    gripper = point['gripper']

    # 역기구학 계산
    #angles = calculate_joint_angles(x, y, z, roll, pitch, yaw)
    angles = point['joint1'],point['joint2'],point['joint3'],point['joint4'],point['joint5'],point['joint6']
    if angles == None : 
        continue

    # 로봇 이동
    arm.set_servo_angle_j(angles=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=True, radius=100.0)
    arm.set_suction_cup(gripper)

# 최종 상태 확인
coordinates, angles = get_robot_state()
print('Final coordinates:', coordinates)
print('Final angles:', angles)

# 연결 해제
arm.disconnect()
