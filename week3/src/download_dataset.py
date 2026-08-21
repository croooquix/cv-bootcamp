import os
import shutil
from roboflow import Roboflow

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.abspath(os.path.join(current_dir, "../data"))
    
    print("="*60)
    print("🛍️ Roboflow에서 Fashion/Clothing 데이터셋 다운로드 스크립트")
    print("="*60)
    
    # 1. API Key 입력 받기
    # Roboflow 로그인 후 Account Settings -> Private API Key에서 확인 가능합니다.
    api_key = input("[*] Roboflow Private API Key를 입력하세요: ").strip()
    
    if not api_key:
        print("[!] API Key가 입력되지 않았습니다. 다운로드를 중단합니다.")
        return
        
    try:
        # 2. Roboflow 클라이언트 초기화 및 다운로드
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("yanelys").project("clothing-segmentation-dataset")
        
        print("[*] 데이터셋 다운로드를 시작합니다 (YOLOv8 포맷)...")
        # target_data_dir 폴더로 바로 다운로드
        dataset = project.version(1).download("yolov8", location=target_data_dir)
        
        print(f"\n[+] 다운로드 성공! 데이터셋 위치: {dataset.location}")
        print("[*] config/data.yaml 파일을 확인하여 경로를 로컬에 맞게 수정해 주세요.")
        
    except Exception as e:
        print(f"\n[!] 다운로드 중 에러가 발생했습니다: {e}")
        print("[*] 방법 A(브라우저에서 zip 직접 다운로드)를 이용해 week3/data에 압축을 풀어주셔도 무방합니다.")

if __name__ == "__main__":
    main()
