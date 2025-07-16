import os
import csv
import time
import keyboard  # 키보드 입력 감지 모듈
from xarm.wrapper import XArmAPI

# csv 저장 경로 설정: C:\Users\PDI\Downloads\0401_Csv
data_folder = r'..\..\data\kinesthetic demonstration\0401_Csv'
os.makedirs(data_folder, exist_ok=True)

# 기존 파일 중 마지막 번호 확인 (data001.csv, data002.csv 등)
existing_files = [f for f in os.listdir(data_folder) if f.startswith('data') and f.endswith('.csv')]
file_number = len(existing_files) + 1
datafile_path = os.path.join(data_folder, f'data{file_number:03d}.csv')

print(f"Recording data to: {datafile_path}")

ip = "192.168.1.180" # xarm IP 주소 :로봇암 바꾸면 바꾸세요
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# 현재 좌표 및 각도 가져오는 함수
def get_robot_state():
    coordinates = arm.get_position(is_radian=False)
    angles = arm.get_servo_angle()
    return coordinates[1], angles[1]

# 수동 모드로 변경
arm.set_mode(2) 
"""시작 전에 확인. 0이 default, 2가 수동"""
arm.set_state(0)

gripper_start_state = 0
"""시작 전에 확인"""
arm.set_suction_cup(gripper_start_state) 
is_gripper_open = gripper_start_state  # 그리퍼 제어 변수

# CSV 파일 작성
with open(datafile_path, 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    try:
        start_time = time.time()
        while True:
            if keyboard.is_pressed('up'):
                arm.set_suction_cup(True)
                is_gripper_open = 1
            elif keyboard.is_pressed('down'):
                arm.set_suction_cup(False)
                is_gripper_open = 0
            
            coordinates, angles = get_robot_state()
            timestamp = round((time.time() - start_time) * 1000, 0)
            
            writer.writerow({
                'timestamp': timestamp,
                'x': coordinates[0],
                'y': coordinates[1],
                'z': coordinates[2],
                'roll': coordinates[3],
                'pitch': coordinates[4],
                'yaw': coordinates[5],
                'joint1': angles[0],
                'joint2': angles[1],
                'joint3': angles[2],
                'joint4': angles[3],
                'joint5': angles[4],
                'joint6': angles[5],
                'gripper': is_gripper_open
            })
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Data recording stopped.")

arm.set_mode(0)
arm.disconnect()

# keyboard 모듈이 없을 경우 설치 방법
# pip install keyboard
