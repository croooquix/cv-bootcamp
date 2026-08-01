import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 가장 최근에 학습 완료된 가중치 모델 자동 감지 및 로드
    import glob
    runs_pattern = os.path.abspath(os.path.join(current_dir, "../../runs/detect/train*"))
    runs_dirs = glob.glob(runs_pattern)
    
    if runs_dirs:
        # 가장 최근에 수정된 폴더 찾기
        latest_run = max(runs_dirs, key=os.path.getmtime)
        model_path = os.path.join(latest_run, "weights/best.pt")
    else:
        model_path = "yolov8n.pt"
    
    if not os.path.exists(model_path):
        print(f"[!] 학습 완료된 best.pt 파일을 찾을 수 없습니다: {model_path}")
        print("[*] 대신 기본 사전학습 모델(yolov8n.pt)을 사용하여 추론을 테스트합니다.")
        model = YOLO("yolov8n.pt")
    else:
        print(f"[*] 최근 학습된 가중치 로드 완료: {model_path}")
        model = YOLO(model_path)
    
    # 2. 추론할 테스트 이미지 경로 지정
    # 본인의 데이터셋에 맞추어 테스트하고 싶은 이미지 경로로 교체해 주세요.
    image_path = os.path.abspath(os.path.join(current_dir, "../data/test_image.jpg"))
    if not os.path.exists(image_path):
        print(f"[!] 테스트용 이미지가 없습니다: {image_path}")
        print("[!] week3/data/test_image.jpg 경로에 테스트할 이미지를 배치해 주세요.")
        return

    # 3. 이미지 로드 및 객체 탐지 실행
    image = cv2.imread(image_path)
    results = model(image)
    
    # 4. 탐지된 객체 시각화 (OpenCV 활용)
    print("[*] 객체 탐지 결과 시각화를 진행합니다...")
    for result in results:
        for box in result.boxes:
            # 바운딩 박스 좌표 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 클래스 인덱스 및 라벨 이름
            class_idx = int(box.cls[0])
            label = result.names[class_idx]
            
            # 신뢰도 점수 (Confidence Score)
            confidence = float(box.conf[0])
            
            # 1단계 피드백 반영: 시각화하여 체크하기
            # 바운딩 박스 그리기 (초록색, 두께 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨 텍스트 쓰기
            text = f"{label} ({confidence:.2f})"
            cv2.putText(
                image, 
                text, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
            
            # Crop 및 정제 로직 예시 (추후 멀티모달 추천 시스템 연결용)
            crop_img = image[y1:y2, x1:x2]
            cv2.imwrite(f"crop_{label}_{class_idx}.jpg", crop_img)

    # 5. 결과물 윈도우 창 표시
    cv2.imshow("YOLOv8 Fashion Detection", image)
    print("[*] 아무 키나 누르면 결과 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
