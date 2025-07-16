import pyzed.sl as sl
import math
import numpy as np
import sys
import time
import argparse
from ultralytics import YOLO
import csv
import cv2
import mediapipe as mp
from CoordinateTransform import transform_camera_to_robot, calculate_T, set_points, undistort_points

# 동영상부터 찍고 오세요
# YOLOv8-Pose 모델 로드
pose_model = YOLO("./learned_pt_file/yolov8n-pose.pt")

# Mediapipe Hands 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def perform_person_inference(image_np):
    """
    입력 이미지(numpy 배열, BGR 형식)에서 사람을 검출하고, 각 사람의 keypoints 정보를 반환.
    [
      {"keypoints": np.array([[x1, y1, conf1], [x2, y2, conf2], ..., [xN, yN, confN]])},
      ...
    ]
    """
    # 이미지가 BGRA 형식이면 BGR로 변환
    if image_np.shape[2] == 4:
        image_rgb = image_np[:, :, :3]
    else:
        image_rgb = image_np

    results = pose_model.predict(image_rgb, conf=0.5)
    detections = []
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy() if hasattr(result.keypoints, 'cpu') else result.keypoints
            for person_kps in keypoints:
                detections.append({"keypoints": person_kps})
    return detections

def detect_hand_with_mediapipe(detection, wrist_pos):
    """Mediapipe Hands를 이용해 손을 검출"""
    x, y = int(wrist_pos[0]), int(wrist_pos[1])
    # 손목 주변을 크롭
    hand_roi = detection[max(0, y-50):y+100, max(0, x-50):x+100]
    if hand_roi.size == 0:
        return None

    hand_roi_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(hand_roi_rgb)

    hand_landmarks_list = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for idx, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = hand_roi.shape
                hand_x, hand_y = int(lm.x * w), int(lm.y * h)
                # 원본 이미지 좌표로 변환
                hand_landmarks_list.append((hand_x + x - 50, hand_y + y - 50))
    return hand_landmarks_list

def visualize_inference_results(image_np, detections, frame_count, target_mode):
    """
    YOLO + Mediapipe 결과를 시각화하여 화면에 표시
    """
    image_copy = image_np.copy()
    if detections is not None:
        for detection in detections:
            keypoints = detection.get("keypoints", None)
            if keypoints is not None and len(keypoints) >= 10:
                left_wrist = keypoints[9][:2]  # 왼손목
                # right_wrist = keypoints[10][:2]  # 필요시 활성화
                if target_mode == 'hand':
                    for wrist in [left_wrist]:  # 필요시 오른손목도 추가
                        hand_landmarks = detect_hand_with_mediapipe(image_np, wrist)
                        if hand_landmarks:
                            for (hx, hy) in hand_landmarks:
                                cv2.circle(image_copy, (hx, hy), 5, (0, 255, 0), -1)
                    # 손목 위치 시각화 (빨간색 원)
                cv2.circle(image_copy, (int(left_wrist[0]), int(left_wrist[1])), 5, (0, 0, 255), -1)

    cv2.putText(image_copy, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("YOLOv8-Pose + Mediapipe Hands", image_copy)
    cv2.waitKey(1)
    return image_np

start_time = None  
def process_frame(zed, runtime_parameters, fps, csv_writer, frame_count, visualize, target_mode, pry_default):
    """
    한 프레임씩 캡처 -> 인퍼런스 -> (손목/손) 3D 좌표 계산 -> CSV 저장
    target_mode: "wrist" 또는 "hand"
    """
    global start_time

    # 좌표 변환 행렬 준비
    camera_points, robot_points = set_points()
    camera_points = undistort_points(camera_points) # 왜곡 보정
    T = calculate_T(camera_points, robot_points)

    grab_status = zed.grab(runtime_parameters)
    if grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
        print("End of SVO file reached.")
        return False, frame_count
    elif grab_status != sl.ERROR_CODE.SUCCESS:
        print("Error during grab:", grab_status)
        return False, frame_count

    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    zed.retrieve_image(image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

    # OpenCV용 numpy 배열
    image_np = image.get_data()

    # YOLO-Pose 추론
    detections = perform_person_inference(image_np)

    if detections:
        for detection in detections:
            keypoints = detection.get("keypoints", None)
            if keypoints is not None and len(keypoints) >= 10:
                # 왼손목, 오른손목 픽셀 좌표
                left_wrist = keypoints[9][:2]
                right_wrist = keypoints[10][:2]

                # 손목(들)을 순회
                for wrist in [left_wrist, right_wrist]:
                    x_pixel, y_pixel = int(wrist[0]), int(wrist[1])
                    err, wrist_3d = point_cloud.get_value(x_pixel, y_pixel)
                    # 3D 좌표 유효성 체크
                    if err == sl.ERROR_CODE.SUCCESS and math.isfinite(wrist_3d[2]):
                        # 로봇 좌표계로 변환
                        wrist_robot = transform_camera_to_robot(wrist_3d[:3], T)
                        print(f"[Frame {frame_count}] Wrist => {wrist_robot}")

                        # ---------------------
                        # (1) wrist 모드:
                        if target_mode == 'wrist':
                            if start_time is None:
                                start_time = time.time()
                            timestamp = start_time + frame_count / fps  # 일정한 간격 유지
                            csv_writer.writerow([
                                timestamp,
                                wrist_robot[0], wrist_robot[1], wrist_robot[2],
                                pry_default[0], pry_default[1], pry_default[2]
                            ])
                            print(f"저장(손목): {wrist_robot}")

                        # ---------------------
                        # (2) hand 모드:
                        elif target_mode == 'hand':
                            # Mediapipe로 손 키포인트 검출
                            hand_landmarks = detect_hand_with_mediapipe(image_np, wrist)
                            if hand_landmarks:
                                # 손가락 키포인트 평균 => 손 중심
                                hand_x = np.mean([hx for hx, hy in hand_landmarks])
                                hand_y = np.mean([hy for hx, hy in hand_landmarks])

                                err_h, hand_3d = point_cloud.get_value(int(hand_x), int(hand_y))
                                if (err_h == sl.ERROR_CODE.SUCCESS 
                                    and math.isfinite(hand_3d[2])):
                                    hand_robot = transform_camera_to_robot(hand_3d[:3], T)
                                    if start_time is None:
                                        start_time = time.time()
                                    timestamp = start_time + frame_count / fps  # 일정한 간격 유지
                                    csv_writer.writerow([
                                        timestamp,
                                        hand_robot[0], hand_robot[1], hand_robot[2],
                                        pry_default[0], pry_default[1], pry_default[2]
                                    ])
                                    print(f"저장(손): {hand_robot}")
                            else:
                                print(f"[Frame {frame_count}] 손 인식 실패")
                    else:
                        print(f"[Frame {frame_count}] 손목 유효 3D 좌표 없음")

    # 시각화
    if visualize:
        image_np = visualize_inference_results(image_np, detections, frame_count, target_mode)

    frame_count += 1
    time.sleep(1 / fps)
    return True, frame_count

def main():
    import pyzed.sl as sl

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, required=True,
                        help='Path to the input SVO file')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second to process')
    parser.add_argument('--target', type=str, choices=['wrist', 'hand'], default='wrist',
                        help='Choose which to record: "wrist" or "hand"')
    # CSV 기본 경로를 상대경로로 수정
    parser.add_argument('--output_csv', type=str, default=r'..\..\data\observation demonstration\output.csv',
                        help='Output CSV file path')
    args = parser.parse_args()
    
    print("Starting 3D Coordinate Extraction from SVO file")
    print(f"Target mode: {args.target}")

    #---------------설정----------------
    visualize = True
    pry_default = [-91.5, -7, -91.9]
    #----------------------------------

    # ZED 카메라 초기화
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.set_from_svo_file(args.input_svo_file)

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open:", repr(status), ". Exit program.")
        sys.exit(1)

    runtime_parameters = sl.RuntimeParameters()

    # CSV 열기
    csv_file = open(args.output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)

    # target 모드별로 CSV 헤더 설정
    csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])

    frame_count = 0
    continue_processing = True

    try:
        while continue_processing:
            continue_processing, frame_count = process_frame(
                zed, runtime_parameters, args.fps, csv_writer, frame_count,
                visualize=visualize, target_mode=args.target,
                pry_default=pry_default
            )
    except KeyboardInterrupt:
        print("Processing stopped by user.")
    finally:
        csv_file.close()
        zed.close()
        cv2.destroyAllWindows()
        print("Processing complete. Total frames processed:", frame_count)

if __name__ == "__main__":
    main()
