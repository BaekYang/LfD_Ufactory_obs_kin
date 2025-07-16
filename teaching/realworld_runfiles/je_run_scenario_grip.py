import os
import csv
import sys
import time
import params
import argparse

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
        raise ValueError(f"Inverse kinematics calculation failed with code {code}")

ip = "192.168.1.194"
arm = XArmAPI(ip) #시뮬레이터 용이라 실제에서는 baud, check지우세요 / baud_checkset=False, check_joint_limit=False
arm.motion_enable(enable=True)
arm.set_mode(1)
arm.set_state(state=0)

# 명령줄 인자 설정
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', action='store', type=str, help='file to run xArm6', required=True)
args = parser.parse_args()

# 로봇 초기 정보 출력
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

# CSV 파일 경로
file_name = args.file_name
file_path = os.path.join('data', file_name if '.csv' in file_name[-4:] else file_name + '.csv')

data = []
# CSV 파일 읽기
with open(file_path, 'r') as csvfile:
    fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append({k: float(v) for k, v in row.items()})

# 첫 번째 타임스탬프를 기준으로 시작 시간 설정
start_time = time.perf_counter()
initial_timestamp = data[0]['timestamp']

# CSV 데이터를 순회하면서 로봇 제어
for point in data:
    target_time = start_time + (point['timestamp'] - initial_timestamp) / 1000.0  # 밀리세컨드 단위로 변환

    x, y, z = point['x'], point['y'], point['z']
    roll, pitch, yaw = point['roll'], point['pitch'], point['yaw']
    gripper = point['gripper']

    # 역기구학 계산
    angles = calculate_joint_angles(x, y, z, roll, pitch, yaw)

    # 로봇 이동
    arm.set_servo_angle_j(angles=angles, speed=params.angle_speed, mvacc=params.angle_acc, wait=True, radius=100.0)
    arm.set_suction_cup(gripper)

    # 타임스탬프 동기화
    while time.perf_counter() < target_time:
        time.sleep(0.001)

# 최종 상태 확인
coordinates, angles = get_robot_state()
print('Final coordinates:', coordinates)
print('Final angles:', angles)

# 연결 해제
arm.disconnect()