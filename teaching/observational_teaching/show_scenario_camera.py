import os
import csv
import time
import keyboard  # pip install keyboard
import threading
from xarm.wrapper import XArmAPI
import pyzed.sl as sl

# 이 코드는 zed를 실행시켜서 교시를 기록하는 코드입니다.
# 로봇암 (가동시켰을 경우) 기록됩니다.
# 로봇암이 없으면 이 코드를 사용하지 마세요.

# CSV와 SVO 파일 저장 경로 설정
csv_folder = r'..\..\data\kinesthetic demonstration'
svo_folder = r'.\demonstration_videos'
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(svo_folder, exist_ok=True)

# 기존 CSV 파일 개수를 확인하여 다음 파일 번호 결정 (예: data001.csv, data002.csv, …)
existing_files = [f for f in os.listdir(csv_folder) if f.startswith('data') and f.endswith('.csv')]
file_number = len(existing_files) + 1
csv_file_path = os.path.join(csv_folder, f'data{file_number:03d}.csv')
svo_file_path = os.path.join(svo_folder, f'data{file_number:03d}.svo')

print(f"CSV recording to: {csv_file_path}")
print(f"SVO recording file: {svo_file_path}")

# ============ ZED 카메라 녹화 설정 (pyzed.sl 사용) ============
zed = sl.Camera()

# 초기화 파라미터 (환경에 맞게 수정)
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # HD720, HD1080 등 선택
init_params.camera_fps = 30

# 재시도 루프를 통해 카메라 연결 확인 (최대 5회 재시도)
max_retries = 5
retry_count = 0
err = sl.ERROR_CODE.FAILURE
while retry_count < max_retries:
    err = zed.open(init_params)
    if err == sl.ERROR_CODE.SUCCESS:
        break
    print("Failed to open ZED camera, retrying in 2 seconds...")
    time.sleep(2)
    retry_count += 1

if err != sl.ERROR_CODE.SUCCESS:
    print("Unable to open ZED camera after several attempts. SVO recording will not proceed.")
    exit(1)

# 녹화 파라미터 설정 (H264 압축 모드 사용, 필요시 다른 압축 모드로 변경)
record_params = sl.RecordingParameters(svo_file_path, sl.SVO_COMPRESSION_MODE.H264)
err = zed.enable_recording(record_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to start SVO recording")
    exit(1)

# ZED 녹화 루프를 별도 스레드로 실행 (30 FPS 기준)
def zed_record_loop(stop_event):
    runtime = sl.RuntimeParameters()
    while not stop_event.is_set():
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # record() 호출 제거: grab() 호출 후 자동으로 프레임이 녹화됩니다.
            pass
        time.sleep(1/30.0)
    print("ZED recording loop terminated.")

stop_event = threading.Event()
zed_thread = threading.Thread(target=zed_record_loop, args=(stop_event,))
zed_thread.start()
# =============================================================

# ============ xArm 설정 및 CSV 기록 ============
ip = "192.168.1.180"
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

def get_robot_state():
    coordinates = arm.get_position(is_radian=False)
    angles = arm.get_servo_angle()
    return coordinates[1], angles[1]

# 수동 모드 설정 (0: default, 2: 수동)
arm.set_mode(2)
arm.set_state(0)

gripper_start_state = 0
arm.set_suction_cup(gripper_start_state)
is_gripper_open = gripper_start_state

with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
                  'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    start_time = time.time()  # 녹화 시작 시간 기록
    try:
        while True:
            # 키보드 입력으로 그리퍼 상태 변경
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
    finally:
        total_time = time.time() - start_time
        print(f"총 녹화 시간: {total_time:.2f}초")  # 총 녹화 시간 출력
        # 종료 시 ZED 녹화 루프 중지 및 리소스 정리
        stop_event.set()
        zed_thread.join()
        zed.disable_recording()
        zed.close()
        arm.set_mode(0)
        arm.disconnect()
