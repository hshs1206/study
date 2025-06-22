import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO
import cv2

attendance_file = 'attendance.xlsx'

# YOLO 모델 로드 (피아식별 1.2 모델로 학습된 모델)
model = YOLO("best.pt")  # 훈련된 모델 경로

# 엑셀 파일 읽기 (없으면 생성)
try:
    df = pd.read_excel(attendance_file)
    print("엑셀 파일의 컬럼 이름:", df.columns)  # 컬럼 이름 출력하여 확인
except FileNotFoundError:
    # 엑셀 파일이 없다면 새로 생성
    df = pd.DataFrame(columns=['Name', 'Time'])  # 컬럼 이름 'Name'과 'Time'으로 설정

# 출석 체크 함수
def mark_attendance(name):
    now = datetime.now()
    
    # 이미 출석한 사람인지 확인 (1시간 이내는 출석하지 않음)
    last_attendance_time = df[df['Name'] == name]['Time']
    if not last_attendance_time.empty:
        last_time = pd.to_datetime(last_attendance_time.iloc[-1])
        if (now - last_time) < timedelta(hours=1):
            return  # 이미 1시간 이내에 출석한 경우

    # 출석 기록 추가
    df.loc[len(df)] = [name, now]
    
    # 엑셀 파일에 저장 (엑셀 파일이 열려 있지 않다면, 파일 권한 확인)
    try:
        df.to_excel(attendance_file, index=False)  # 파일에 저장
    except PermissionError:
        # 엑셀 파일이 열려 있으면 다른 이름으로 저장 시도
        new_file = 'attendance_v2.xlsx'
        df.to_excel(new_file, index=False)  # 새로운 파일에 저장
        print(f"엑셀 파일에 쓰기 권한이 없어 '{new_file}'에 저장되었습니다.")

    # 출석 정보 출력
    print(f'{name} 출석 완료, 시간: {now.strftime("%Y-%m-%d %H:%M:%S")}')

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 기본 웹캠

# 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 비디오 코덱 설정
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # 저장할 영상의 이름과 프레임 크기 설정

while True:
    # 웹캠에서 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        break

    # 모델에 입력
    results = model(frame)  # 모델을 사용하여 프레임을 입력

    # 결과 처리
    for result in results:
        boxes = result.boxes  # 바운딩 박스 객체
        for box in boxes:
            label = int(box.cls)  # 클래스 레이블을 정수형으로 변환
            confidence = float(box.conf)  # 신뢰도를 실수형으로 변환
            name = model.names[label]  # YOLO 모델의 클래스 이름을 이름으로 변환

            # 사람이 인식된 경우에만 처리
            if confidence > 0.5:  # 신뢰도 50% 이상일 경우만
                x1, y1, x2, y2 = box.xyxy[0]  # 바운딩 박스 좌표
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{name} {confidence:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 출석 기록 (사람 이름이 나오면 출석 체크)
                mark_attendance(name)

    # 프레임을 비디오에 저장
    out.write(frame)  # 현재 프레임을 비디오 파일로 저장

    # 프레임을 화면에 표시
    cv2.imshow('Real-time Attendance Detection', frame)  # 실시간 객체 감지 영상 표시

    # ESC키를 눌러 종료 (키보드 입력 대기)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC(ASCII 27) 키가 눌리면 종료
        break

# 웹캠 종료
cap.release()  # 웹캠 해제
out.release()  # 비디오 파일 저장 종료
cv2.destroyAllWindows()  # 모든 OpenCV 윈도우 종료
