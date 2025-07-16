
============================================================


교시 코드

택1. 비디오 교시 : teaching\observational_teaching\
1) show_scenario_camera.py = 실행하면 로봇암 동작 기록(csv형태)하고, zed 2i카메라로 .svo2형태의 영상 저장됨, csv와 svo2 동시저장, 파일명 동일 저장됨
2) observational_learning.py= 영상에서 손 관절좌표 디텍션하고 csv파일 출력
left_wrist = keypoints[9][:2]   # 왼손목 (x, y)
right_wrist = keypoints[10][:2] # 오른손목 (x, y)
3) observational_learning_details.py = 추가기능. 실행 안해두 됨(손 관절번호 입력하면 그 관절 좌표 출력)

택2. 직접 교시 : teaching\kinesthetic_teaching\
1) show_scenario_gripper = 실행하면 로봇암 동작 기록(csv, dataframe : )

교시가 끝나면 후처리를 해야합니다.


============================================================


후처리 코드 : teaching\realworld_runfiles
1) pumping_go=초당 300회 샘플링 데이터 보간(데이터를 생성하는) 코드(폴더에 있는 csv 전체)
2) CoordinateTansform.py = 좌표계 변환(완성) 카메라→로봇암(x,y,z)
3) ik_traj.py = x,y,z,r,p,y 값으로 joint1~joint6 계산 및 csv파일 출력

잘 움직이나 로봇암에서 실제로  확인하는 코드 :
1) run_scenario_result = 스페이스바 누르는 동안 csv파일 로봇암동작 재생(traj 팝업)
2) remove_csv.py = 트레젝터리 튀는 값을 없애기 위해서 y값이 200보다 작으면 해당 행부터 마지막 행까지 삭제시키고 csv 출력


============================================================


기타 코드 : 
- svo2_xyzdetection.py = .svo2 영상에서 클릭한 픽셀의 x,y,z좌표 출력
- moveing_avg.py = 좌표값 5개 평균으로만 이동, 행 간 1이상 이동x(궤적 매끄럽게 하는 작업)
-observational_learning_allsvo.py = 디렉토리 안에 모든 svo2파일에 대해 yolo hand detection 좌표를 각각의 csv로 출력(코드 내 디렉토리 지정필요)


============================================================

인과적 CNN 딥러닝 시켜서 특징을 잡아내 경로생성하는 코드
- cnn_50.py= csv파일 8초까지만 학습(cnn)하고 그래프 띄우기/ 
- params.py= 파라미터


