import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

def main():
    # 1. 학습에 필요한 설정 파일 경로 지정
    # 프로젝트 루트 기준으로 상대경로를 절대경로로 변환합니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, "../config/data.yaml"))
    
    print(f"[*] 사용할 data.yaml 경로: {config_path}")
    
    # 2. YOLOv8 나노 모델 로드 (하이퍼파라미터 튜닝 실험용)
    model = YOLO("yolov8n.pt")
    
    # 3. 모델 학습 진행 (학습률 lr0=0.005, 배치 batch=8, 옵티마이저 AdamW)
    print("[*] YOLOv8 패션 아이템 검출 모델(Nano) 하이퍼파라미터 튜닝 학습을 시작합니다...")
    model.train(
        data=config_path, 
        epochs=10, 
        imgsz=640,
        lr0=0.005,
        batch=8,
        optimizer='AdamW',
        device=0 # GPU 학습을 위해 0으로 변경 (CUDA)
    )
    print("[*] 학습이 정상적으로 완료되었습니다.")
    print("[*] 결과 가중치 및 로그는 'week3/runs/detect/train' 폴더에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main()
