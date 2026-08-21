import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
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
        model_path = ""
        
    if not os.path.exists(model_path):
        print(f"[!] 학습 완료된 best.pt 파일을 찾을 수 없습니다: {model_path}")
        print("[*] 성능 평가를 위해서는 먼저 학습을 완료하여 best.pt 파일을 생성해야 합니다.")
        return
        
    model = YOLO(model_path)
    
    # 2. 모델 검증(Validation) 수행
    # data.yaml에 정의된 validation 데이터셋을 바탕으로 평가를 수행합니다.
    print("[*] validation 데이터셋을 사용하여 모델 평가를 시작합니다...")
    metrics = model.val()
    
    # 3. 주요 평가지표 출력
    print("\n" + "="*40)
    print("📊 모델 주요 성능 평가 결과")
    print("="*40)
    # mAP50, mAP50-95 등 출력
    print(f"- mAP50 (Mean Average Precision @ 0.5): {metrics.box.map50:.4f}")
    print(f"- mAP50-95 (Mean Average Precision @ 0.5:0.95): {metrics.box.map:.4f}")
    print(f"- Precision (정밀도): {metrics.box.mp:.4f}")
    print(f"- Recall (재현율): {metrics.box.mr:.4f}")
    print("="*40)
    
    print("\n[*] 검증 관련 그래프 및 이미지 파일들은 'runs/detect/val/' 폴더에 자동 저장되었습니다.")
    print("   - confusion_matrix.png: 클래스별 혼동 행렬")
    print("   - PR_curve.png: Precision-Recall 곡선")
    print("   - F1_curve.png: 신뢰도에 따른 F1 Score 곡선")

if __name__ == "__main__":
    main()
