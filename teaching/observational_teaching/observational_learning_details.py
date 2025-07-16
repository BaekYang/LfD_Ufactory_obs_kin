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
import os

# YOLOv8-Pose 모델 로드 (반드시 필요)
pose_model = YOLO("./learned_pt_file/yolov8n-pose.pt")

# Mediapipe Hands 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def perform_person_inference(image_np):
    """
    YOLOv8-Pose로 사람 검출 + keypoints 추출
    """
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
    """
    Mediapipe Hands로 손 랜드마크(0~20) 검출
    """
    x, y = int(wrist_pos[0]), int(wrist_pos[1])
    # 손목 주변 영역을 잘라서 분석
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
                # ROI 오프셋을 더해 원본 좌표로 변환
                hand_landmarks_list.append((hand_x + x - 50, hand_y + y - 50))
    return hand_landmarks_list

def parse_target_mode(target_str):
    """
    --target 인자 파싱
    wrist      -> YOLO-Pose 왼/오른손목 좌표
    hand       -> Mediapipe로 검출한 전체 손 랜드마크의 평균 (손 중심)
    landmarkX  -> Mediapipe로 검출한 특정 인덱스 X(0~20) 좌표
    """
    if target_str in ['wrist', 'hand']:
        return (target_str, None)
    # 'landmark'라는 접두어가 있고, 뒤가 숫자라면
    if target_str.startswith('landmark'):
        idx_str = target_str[len('landmark'):]
        if idx_str.isdigit():
            idx_val = int(idx_str)
            if 0 <= idx_val <= 20:
                return ('landmark', idx_val)
    raise ValueError(f"Invalid target mode: {target_str}")

def visualize_inference_results(image_np, detections, frame_count, target_mode):
    """
    화면에 손목/손 랜드마크 시각화 + 인덱스 표시
    """
    mode_type, mode_val = target_mode
    image_copy = image_np.copy()

    if detections is not None:
        for detection in detections:
            keypoints = detection.get("keypoints", None)
            if keypoints is not None and len(keypoints) >= 10:
                left_wrist = keypoints[9][:2]   # 왼손목 (YOLO)
                right_wrist = keypoints[10][:2]  # 오른손목 (YOLO)

                # 왼손목만 예시로 시각화 (필요시 오른손목도 시각화)
                cv2.circle(image_copy, (int(left_wrist[0]), int(left_wrist[1])), 5, (0, 0, 255), -1)
                cv2.putText(image_copy, "yolo_wrist(L)", (int(left_wrist[0])+5, int(left_wrist[1])-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                if mode_type in ['hand', 'landmark']:
                    # Mediapipe로 검출한 손 랜드마크
                    hand_landmarks = detect_hand_with_mediapipe(image_np, left_wrist)
                    if hand_landmarks:
                        # 모든 점 시각화
                        for idx, (hx, hy) in enumerate(hand_landmarks):
                            cv2.circle(image_copy, (hx, hy), 3, (0, 255, 0), -1)
                            cv2.putText(image_copy, str(idx), (hx+2, hy-2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(image_copy, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("YOLOv8-Pose + Mediapipe Hands", image_copy)
    cv2.waitKey(1)
    return image_np

start_time = None

def process_frame(zed, runtime_parameters, fps, csv_writer, frame_count, visualize, target_mode, pry_default):
    """
    한 프레임씩 캡처 -> (YOLO + Mediapipe) -> 3D좌표 -> CSV
    """
    global start_time
    mode_type, mode_val = target_mode  # ('wrist', None), ('hand', None), ('landmark', X)

    # 좌표 변환행렬 준비
    camera_points, robot_points = set_points()
    camera_points = undistort_points(camera_points)  # 왜곡 보정
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

    image_np = image.get_data()
    detections = perform_person_inference(image_np)

    if detections:
        for detection in detections:
            keypoints = detection.get("keypoints", None)
            if keypoints is not None and len(keypoints) >= 10:
                # YOLO가 찾은 왼손목, 오른손목
                left_wrist = keypoints[9][:2]
                right_wrist = keypoints[10][:2]

                if mode_type == 'wrist':
                    valid_wrist_coords = []
                    for wrist in [left_wrist, right_wrist]:
                        x_pixel, y_pixel = int(wrist[0]), int(wrist[1])
                        err, wrist_3d = point_cloud.get_value(x_pixel, y_pixel)
                        if err == sl.ERROR_CODE.SUCCESS and math.isfinite(wrist_3d[2]):
                            wrist_robot = transform_camera_to_robot(wrist_3d[:3], T)
                            print(f"[Frame {frame_count}] YOLO Wrist => {wrist_robot}")
                            valid_wrist_coords.append(wrist_robot)
                        else:
                            print(f"[Frame {frame_count}] 손목 유효 3D 좌표 없음")
                    if valid_wrist_coords:
                        if len(valid_wrist_coords) == 2:
                            avg_robot = [(valid_wrist_coords[0][i] + valid_wrist_coords[1][i]) / 2 for i in range(3)]
                        else:
                            avg_robot = valid_wrist_coords[0]
                        if start_time is None:
                            start_time = 0
                        # 타임스탬프에 1초 오프셋 추가: 첫 행이 1.0부터 시작함
                        timestamp = start_time + 1 + frame_count / fps
                        csv_writer.writerow([
                            timestamp,
                            avg_robot[0], avg_robot[1], avg_robot[2],
                            pry_default[0], pry_default[1], pry_default[2]
                        ])
                        print(f"저장(손): {avg_robot}")

                elif mode_type in ['hand', 'landmark']:
                    valid_hand_coords = []
                    for wrist in [left_wrist, right_wrist]:
                        x_pixel, y_pixel = int(wrist[0]), int(wrist[1])
                        err, wrist_3d = point_cloud.get_value(x_pixel, y_pixel)
                        if err == sl.ERROR_CODE.SUCCESS and math.isfinite(wrist_3d[2]):
                            print(f"[Frame {frame_count}] YOLO Wrist => {wrist_3d}")
                            hand_landmarks = detect_hand_with_mediapipe(image_np, wrist)
                            if hand_landmarks:
                                if mode_type == 'hand':
                                    # 손 랜드마크 평균 => 손 중심 좌표 계산
                                    hand_x = np.mean([hx for hx, hy in hand_landmarks])
                                    hand_y = np.mean([hy for hx, hy in hand_landmarks])
                                    err_h, hand_3d = point_cloud.get_value(int(hand_x), int(hand_y))
                                    if err_h == sl.ERROR_CODE.SUCCESS and math.isfinite(hand_3d[2]):
                                        hand_robot = transform_camera_to_robot(hand_3d[:3], T)
                                        valid_hand_coords.append(hand_robot)
                                    else:
                                        print(f"[Frame {frame_count}] 유효한 손 3D 좌표 없음 (hand mode)")
                                else:  # mode_type == 'landmark'
                                    lm_idx = mode_val
                                    if lm_idx < len(hand_landmarks):
                                        hx, hy = hand_landmarks[lm_idx]
                                        err_lm, lm_3d = point_cloud.get_value(int(hx), int(hy))
                                        if err_lm == sl.ERROR_CODE.SUCCESS and math.isfinite(lm_3d[2]):
                                            lm_robot = transform_camera_to_robot(lm_3d[:3], T)
                                            valid_hand_coords.append(lm_robot)
                                        else:
                                            print(f"[Frame {frame_count}] 유효한 랜드마크 3D 좌표 없음 (landmark mode)")
                                    else:
                                        print(f"해당 랜드마크 인덱스 {lm_idx} 없음")
                            else:
                                print(f"[Frame {frame_count}] 손 인식 실패")
                        else:
                            print(f"[Frame {frame_count}] 손목 유효 3D 좌표 없음")
                    if valid_hand_coords:
                        if len(valid_hand_coords) == 2:
                            avg_robot = [(valid_hand_coords[0][i] + valid_hand_coords[1][i]) / 2 for i in range(3)]
                        else:
                            avg_robot = valid_hand_coords[0]
                        if start_time is None:
                            start_time = 0
                        # 타임스탬프에 1초 오프셋 추가: 첫 행이 1.0부터 시작함
                        timestamp = start_time + 1 + frame_count / fps
                        csv_writer.writerow([
                            timestamp,
                            avg_robot[0], avg_robot[1], avg_robot[2],
                            pry_default[0], pry_default[1], pry_default[2]
                        ])
                        if mode_type == 'hand':
                            print(f"저장(손): {avg_robot}")
                        else:
                            print(f"저장(landmark{mode_val}): {avg_robot}")

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
    parser.add_argument('--target', type=str, default='wrist',
                        help='Options: "wrist", "hand", "landmark0" ~ "landmark20"')
    parser.add_argument('--output_csv', type=str, 
                        default=r'..\..\data\observation demonstration\output.csv',
                        help='Output CSV file path')
    args = parser.parse_args()
    
    print("Starting 3D Coordinate Extraction from SVO file")
    print(f"Target mode: {args.target}")

    # target 모드 파싱
    target_mode = parse_target_mode(args.target)

    visualize = True
    pry_default = [-91.5, -7, -91.9]

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

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    csv_file = open(args.output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])

    frame_count = 0
    continue_processing = True

    try:
        while continue_processing:
            continue_processing, frame_count = process_frame(
                zed, runtime_parameters, args.fps, csv_writer, frame_count,
                visualize=visualize, target_mode=target_mode,
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
