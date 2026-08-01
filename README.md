# Computer Vision AI 데이터 전처리 파이프라인 (1차 업무)

이 프로젝트는 Computer Vision AI 모델 학습을 위한 픽셀 단위 이미지 처리 및 Hugging Face 데이터셋 전처리 파이프라인 구축 실습 프로젝트입니다.

---

## 기술 스택 (Tech Stack)
- **Language**: Python 3.x
- **Libraries**: OpenCV (`opencv-python`), NumPy, Hugging Face `datasets`, Pillow

---

## 업무 수행 내용 및 파이프라인

### 1. 픽셀 단위 특정 색상 감지 (`color_filter.py`)
- **BGR -> HSV 색상 공간 변환:** 조명 변화에 강건한 색상 감지를 위해 HSV 공간 활용
- **빨간색 마스킹 (Red Masking):** 0~10도 및 170~180도 두 구간의 Hue 범위를 합산(`mask1 + mask2`)하여 빨간색 영역 감지 및 필터링 구현

### 2. AI 학습용 데이터 전처리 파이프라인 (`image_preprocessing.py`)
Hugging Face의 `ethz/food101` 데이터셋을 실시간 스트리밍(`streaming=True`)으로 로드하여 다음 과정을 수행합니다:

1. **이상치 탐지 (Outlier Detection):**
   - **어두운 이미지 제거:** 그레이스케일 변환 후 픽셀 평균 밝기가 40 미만인 데이터 필터링
   - **작은 객체 이미지 제거:** Canny Edge 및 Contour 탐지를 활용하여 가장 큰 객체의 면적이 전체 화면의 5% 미만인 데이터 필터링
2. **기본 전처리 (Basic Preprocessing):**
   - **크기 조정 (Resize):** ResNet, ViT 등 표준 AI 모델 입력 규격인 `224×224` 크기로 변환
   - **노이즈 제거 (Blur):** Gaussian Blur(5×5) 필터를 적용하여 센서 노이즈 완화
   - **색상 변환 & 정규화 (Normalize):** Grayscale 변환 후 픽셀값을 `0.0 ~ 1.0` 사이의 float 소수로 정규화
3. **데이터 증강 (Data Augmentation):**
   - AI 모델의 과적합(Overfitting) 방지를 위해 좌우 반전(Horizontal Flip), 15도 회전(Rotation), 밝기/색상 변화 적용
4. **결과 저장:**
   - 파이프라인을 최종 통과한 대표 샘플 5장을 `preprocessed_samples/` 폴더에 저장

---

## 실행 방법

```bash
# 1. 필요 패키지 설치
pip install opencv-python numpy datasets pillow

# 2. 색상 필터링 실습 실행
python color_filter.py

# 3. AI 학습 데이터 전처리 파이프라인 실행
python image_preprocessing.py