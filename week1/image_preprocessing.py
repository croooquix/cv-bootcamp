import os
import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image

# 1. 처리된 이미지를 저장할 폴더 생성
output_dir = "preprocessed_samples"
os.makedirs(output_dir, exist_ok=True)
print(f"'{output_dir}' 폴더가 준비되었습니다.")

# 2. Hugging Face에서 실습용 데이터셋 불러오기 (Food-101 음식 데이터셋)
# streaming=True 옵션을 쓰면 5GB짜리 전체 데이터를 다 다운받지 않고,
# 인터넷에서 실시간으로 필요한 만큼만 쏙쏙 뽑아옵니다. (하드디스크 및 시간 절약)
print("Hugging Face에서 데이터셋 스트리밍 연결 중...")
dataset = load_dataset("ethz/food101", split="train", streaming=True)

# 3. 이상치 탐지 (Outlier Detection) 함수 정의
def is_outlier(image_bgr):
    # (1) 너무 어두운 이미지 검사 (그레이스케일 변환 후 평균 밝기 계산)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 40:  # 0(검정)~255(흰색) 중 평균 40 미만이면 어둡다고 판단
        return True, f"너무 어두움 (평균 밝기: {mean_brightness:.1f})"

    # (2) 객체 크기가 너무 작은 이미지 검사 (윤곽선 탐지 활용)
    # 윤곽선(물체의 테두리)을 찾아 가장 큰 물체의 넓이 계산
    edges = cv2.Canny(gray, 50, 150) # NOTE edge map: https://docs.opencv.org/4.13.0/d7/de1/tutorial_js_canny.html?utm_source=copilot.com
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # edge map에서 외곽선만 추출, 직선구간에서는 양끝점만 저장하여 메모리 절약 
    
    if not contours:
        return True, "감지된 객체 없음"
        
    max_area = max(cv2.contourArea(c) for c in contours)
    total_area = image_bgr.shape[0] * image_bgr.shape[1]
    
    # 가장 큰 물체가 전체 화면의 5% 미만이면 너무 작은 객체로 판단
    if (max_area / total_area) < 0.05:
        return True, f"객체가 너무 작음 (화면의 {(max_area/total_area)*100:.1f}%)"

    return False, "정상 이미지"

# 4. 데이터 증강 (Data Augmentation) 함수 정의
def augment_image(image_bgr):
    # (1) 좌우 반전 (Horizontal Flip)
    flipped = cv2.flip(image_bgr, 1)
    
    # (2) 15도 회전 (Rotation)
    h, w = image_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2), 15, 1.0)
    rotated = cv2.warpAffine(image_bgr, matrix, (w, h))
    
    # (3) 색상 변화 (밝기 30 증가)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)
    color_changed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flipped, rotated, color_changed

# 5. 본격적인 데이터 파이프라인 실행 (정상 이미지 5장 수집 및 전처리)
saved_count = 0
total_checked = 0

print("\n데이터 검사 및 전처리 파이프라인 시작")

for item in dataset:
    total_checked += 1
    # PIL 이미지를 OpenCV용 BGR 배열로 변환
    pil_img = item["image"]
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # [심화 과제] 이상치 탐지 필터링 수행
    is_bad, reason = is_outlier(img_bgr)
    if is_bad:
        print(f"[제외됨 #{total_checked}] 사유: {reason}")
        continue  

    # --- 여기까지 합격한 정상 이미지에 대해 전처리 수행 ---
    print(f"[합격 #{total_checked}] 정상 이미지 전처리 진행 중... ({saved_count+1}/5)")
    
    # (1) 크기 조정 (224 x 224)
    resized = cv2.resize(img_bgr, (224, 224))
    
    # (2) 노이즈 제거 (Gaussian Blur 적용)
    # NOTE Gaussian Blur: https://filmora.wondershare.com/video-effects/gaussian-blur.html?utm_source=copilot.com
    # NOTE sliding window: https://homo-deus.com/lab/signal-processing/convolution/?signal_freq=2
    blurred = cv2.GaussianBlur(resized, (5, 5), 0) 

    # (3) 색상 변환 (Grayscale & Normalize 정규화)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32) / 255.0  # 0.0 ~ 1.0 사이 소수로 정규화
    
    # (4) 데이터 증강 (좌우반전, 회전, 색상변화)
    flipped, rotated, color_changed = augment_image(resized)

    # 6. 결과물 이미지 파일로 저장
    # (참고: 정규화된 float 데이터는 다시 255를 곱해 uint8로 바꿔야 눈으로 보는 이미지 파일로 저장됩니다)
    norm_vis = (normalized * 255).astype(np.uint8)
    
    # 5장 저장 중 1장 처리될 때마다 원본, 전처리본, 증강본을 폴더에 저장
    cv2.imwrite(f"{output_dir}/sample_{saved_count}_1_resized.jpg", resized)
    cv2.imwrite(f"{output_dir}/sample_{saved_count}_2_grayscale_norm.jpg", norm_vis)
    cv2.imwrite(f"{output_dir}/sample_{saved_count}_3_blurred.jpg", blurred)
    cv2.imwrite(f"{output_dir}/sample_{saved_count}_4_flipped.jpg", flipped)
    cv2.imwrite(f"{output_dir}/sample_{saved_count}_5_rotated.jpg", rotated)
    
    saved_count += 1
    if saved_count >= 5:
        break 

print(f"\n총 {total_checked}장의 이미지 중 합격한 5장의 전처리 및 증강 완료")
print(f"'{output_dir}' 폴더에 들어가서 생성된 25장의 이미지들을 확인")