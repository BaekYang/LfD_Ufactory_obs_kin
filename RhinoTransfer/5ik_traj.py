import os
import csv
import sys
import time
from xarm.wrapper import XArmAPI

# xArm의 역기구학 계산 함수
def calculate_joint_angles(x, y, z, roll, pitch, yaw):
    pose = [x, y, z, roll, pitch, yaw]
    code, angles = arm.get_inverse_kinematics(pose, input_is_radian=False, return_is_radian=False)
    if code == 0:
        return angles
    else:
        raise ValueError(f"Inverse kinematics calculation failed with code {code}")

def main():
    # 입력 파일 경로 (윈도우 경로의 경우 raw string 사용)
    input_file = r"C:\je_kinesthetic\data\data11.csv"
    # 입력 파일이 위치한 폴더에 ik_traj.csv로 저장
    output_file = os.path.join(os.path.dirname(input_file), "kk.csv")

    # xArm API 초기화 (시뮬레이터/테스트용 IP: 192.168.1.180)
    ip = "192.168.1.180"
    global arm
    arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False)
    arm.motion_enable(enable=True)
    arm.set_mode(1)
    arm.set_state(state=0)

    # 입력 CSV 파일 읽기  
    # 입력 파일의 헤더는 아래 순서로 구성되어 있다고 가정합니다.  
    # [timestamp, x, y, z, r, p, y]  
    # 여기서 첫 번째 y는 좌표 y, 두 번째 y는 yaw (회전각)로 사용됩니다.
    data = []
    with open(input_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더 스킵
        for row in reader:
            # 순서: 0:timestamp, 1:x, 2:y(좌표), 3:z, 4:r(roll), 5:p(pitch), 6:y(yaw)
            try:
                timestamp = float(row[0])
                x_val = float(row[1])
                y_coord = float(row[2])
                z_val = float(row[3])
                r_val = float(row[4])
                p_val = float(row[5])
                yaw_val = float(row[6])
            except Exception as e:
                print(f"CSV 파싱 오류: {e} - row: {row}")
                continue

            data.append({
                "timestamp": timestamp,
                "x": x_val,
                "y_coord": y_coord,   # 좌표 y
                "z": z_val,
                "r": r_val,           # roll
                "p": p_val,           # pitch
                "yaw": yaw_val        # yaw
            })

    # 각 행에 대해 역기구학 계산 후 결과 저장
    output_rows = []
    for row in data:
        try:
            # 역기구학 계산: (x, y, z, roll, pitch, yaw)
            joints = calculate_joint_angles(row["x"], row["y_coord"], row["z"], row["r"], row["p"], row["yaw"])
        except Exception as e:
            print(f"역기구학 계산 실패 (timestamp {row['timestamp']}): {e}")
            # 실패 시 빈 문자열 또는 적당한 기본값으로 기록할 수 있습니다.
            joints = [""] * 6

        # 출력 행: 입력 값 + 관절 각도(joint1~joint6) + gripper(항상 0)
        # 주의: 출력 열 순서는 [timestamp, x, y, z, r, p, y, joint1, joint2, joint3, joint4, joint5, joint6, gripper]
        # 여기서 두 번째 'y'는 yaw를 의미합니다.
        output_rows.append([
            row["timestamp"],
            row["x"],
            row["y_coord"],
            row["z"],
            row["r"],
            row["p"],
            row["yaw"],
            joints[0],
            joints[1],
            joints[2],
            joints[3],
            joints[4],
            joints[5],
            0  # gripper 값은 무조건 0
        ])

    # 출력 CSV 파일 작성
    header_out = ["timestamp", "x", "y", "z", "r", "p", "y", 
                  "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    with open(output_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header_out)
        writer.writerows(output_rows)

    print("IK 계산 완료! 결과는 다음 위치에 저장되었습니다:", output_file)
    
    arm.disconnect()

if __name__ == "__main__":
    main()
