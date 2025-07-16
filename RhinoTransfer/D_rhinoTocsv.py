# -*- coding: utf-8 -*-
import rhinoscriptsyntax as rs
import csv
import os

def export_curve_points_to_csv():
    # 1) 곡선 선택
    curve_id = rs.GetObject(u"CSV로 내보낼 곡선을 선택하세요", rs.filter.curve)
    if not curve_id:
        print("곡선이 선택되지 않았습니다.")
        return

    # 2) 포인트(편집점) 가져오기
    pts = rs.CurvePoints(curve_id)
    if not pts:
        print("곡선으로부터 포인트를 가져올 수 없습니다.")
        return

    # 3) 타임스탬프 설정
    start_time    = rs.GetReal("Enter start timestamp", 0.0)
    timestamp_gap = rs.GetReal("Enter timestamp gap (seconds)", 1.0)

    # 4) 파일명 입력
    filename = rs.GetString(u"저장할 CSV 파일명을 입력하세요 (예: result.csv)", "inference_result.csv")
    if not filename:
        print("파일명이 입력되지 않았습니다.")
        return
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
    filepath = filename

    # 5) Roll/Pitch/Yaw 고정값
    roll, pitch, yaw = 172.9, -83, 7

    # 6) CSV 쓰기 (Python 2.x에서는 binary 모드로)
    f_out = open(filepath, "wb")
    writer = csv.writer(f_out)
    writer.writerow(["timestamp", "x", "y", "z", "roll", "pitch", "yaw"])
    for idx, pt in enumerate(pts):
        ts = start_time + idx * timestamp_gap
        x = int(pt[0] * 100) / 100.0
        y = int(pt[1] * 100) / 100.0
        z = int(pt[2] * 100) / 100.0
        writer.writerow([int(ts), x, y, z, roll, pitch, yaw])
    f_out.close()

    print("CSV 파일이 저장되었습니다:\n" + os.path.abspath(filepath))

if __name__ == "__main__":
    export_curve_points_to_csv()
