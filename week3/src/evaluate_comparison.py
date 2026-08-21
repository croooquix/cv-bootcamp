import os
import pandas as pd
import matplotlib.pyplot as plt

# Windows OpenMP 런타임 충돌 방지 패치
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 비교 대상이 되는 세 가지 실험의 results.csv 경로 정의
    baseline_csv = os.path.abspath(os.path.join(current_dir, "../backup/baseline/results.csv"))
    exp1_csv = os.path.abspath(os.path.join(current_dir, "../../runs/detect/train4/results.csv")) # 증강+20 epoch
    exp2_csv = os.path.abspath(os.path.join(current_dir, "../../runs/detect/train5/results.csv")) # s-model 10 epoch
    exp3_csv = os.path.abspath(os.path.join(current_dir, "../../runs/detect/train6/results.csv")) # 하이퍼파라미터 튜닝
    
    plt.figure(figsize=(10, 6))
    has_data = False
    
    # 1. Baseline 데이터 로드 및 플롯
    if os.path.exists(baseline_csv):
        try:
            df_base = pd.read_csv(baseline_csv)
            # CSV 컬럼명 양끝 공백 제거
            df_base.columns = df_base.columns.str.strip()
            # YOLOv8의 mAP50 컬럼 매핑
            map50_col = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in df_base.columns else df_base.columns[6]
            
            plt.plot(df_base['epoch'], df_base[map50_col], label='Baseline (n-model, 10 epoch)', color='#FF5722', linewidth=2.5)
            has_data = True
            print("[+] Baseline 데이터 로드 완료")
        except Exception as e:
            print(f"[!] Baseline 데이터 로드 실패: {e}")
            
    # 2. 실험 1 (Data Augmentation) 데이터 로드 및 플롯
    if os.path.exists(exp1_csv):
        try:
            df_exp1 = pd.read_csv(exp1_csv)
            df_exp1.columns = df_exp1.columns.str.strip()
            map50_col = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in df_exp1.columns else df_exp1.columns[6]
            
            plt.plot(df_exp1['epoch'], df_exp1[map50_col], label='Exp 1 (n-model, Augment, 20 epoch)', color='#4CAF50', linewidth=2.5)
            has_data = True
            print("[+] 실험 1 (Augmentation) 데이터 로드 완료")
        except Exception as e:
            print(f"[!] 실험 1 데이터 로드 실패: {e}")
            
    # 3. 실험 2 (Small Model Scale-up) 데이터 로드 및 플롯
    if os.path.exists(exp2_csv):
        try:
            df_exp2 = pd.read_csv(exp2_csv)
            df_exp2.columns = df_exp2.columns.str.strip()
            map50_col = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in df_exp2.columns else df_exp2.columns[6]
            
            plt.plot(df_exp2['epoch'], df_exp2[map50_col], label='Exp 2 (s-model, 10 epoch)', color='#2196F3', linewidth=2.5)
            has_data = True
            print("[+] 실험 2 (s-model) 데이터 로드 완료")
        except Exception as e:
            print(f"[!] 실험 2 데이터 로드 실패: {e}")

    # 4. 실험 3 (Hyperparameter Tuning) 데이터 로드 및 플롯
    if os.path.exists(exp3_csv):
        try:
            df_exp3 = pd.read_csv(exp3_csv)
            df_exp3.columns = df_exp3.columns.str.strip()
            map50_col = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in df_exp3.columns else df_exp3.columns[6]
            
            plt.plot(df_exp3['epoch'], df_exp3[map50_col], label='Exp 3 (n-model, tuned lr0/batch, 10 epoch)', color='#9C27B0', linewidth=2.5)
            has_data = True
            print("[+] 실험 3 (Tuned) 데이터 로드 완료")
        except Exception as e:
            print(f"[!] 실험 3 데이터 로드 실패: {e}")

    if not has_data:
        print("[!] 비교할 데이터셋의 results.csv 파일이 존재하지 않습니다.")
        print("    먼저 실험 1과 실험 2를 진행해 주세요.")
        return

    # 그래프 스타일 세팅
    plt.title('YOLOv8 Model Performance Comparison (mAP50)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('mAP50 (Accuracy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right')
    
    # 이미지 저장 및 팝업 출력
    save_path = os.path.abspath(os.path.join(current_dir, "../backup/performance_comparison.png"))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] 비교 성능 그래프가 저장되었습니다: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
