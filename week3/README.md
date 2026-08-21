# 🛍️ Week 3: 이커머스 추천 시스템을 위한 YOLOv8 패션 아이템 검출

본 프로젝트는 **Shopify, SSENSE** 등의 이커머스 플랫폼을 타겟팅하는 **"시각 기반 멀티모달 추천 시스템 (Visual RecSys)"**의 첫 번째 단계인 **YOLOv8 기반 패션 아이템 검출(Object Detection) 모델**을 구현합니다.

---

## 📂 폴더 구조 (2주차 피드백 반영)
2주차 피드백을 적극 반영하여 실행 코드, 설정 파일, 데이터셋 경로를 엄격히 분리하여 설계했습니다.

```text
week3/
├── config/
│   └── data.yaml          # YOLOv8 학습용 데이터셋 메타데이터 (경로, 클래스 정보)
├── data/
│   ├── train/             # 학습 이미지 및 라벨
│   ├── valid/             # 검증 이미지 및 라벨
│   └── test/              # 테스트 이미지 및 라벨 (선택)
├── src/
│   ├── train.py           # YOLOv8 모델 학습 스크립트
│   ├── inference.py       # OpenCV 연동 객체 검출 및 Bounding Box 시각화 스크립트
│   └── evaluate.py        # 모델 성능 평가 지표 및 Precision/Recall 곡선 시각화 스크립트
├── runs/                  # YOLOv8 학습 과정 및 메트릭 자동 저장 폴더 (.gitignore 대상)
└── README.md              # 본 프로젝트 진행 및 학습 결과 요약 문서
```

---

## 🛠️ 개발 환경 구성
학습 및 추론을 위해 필요한 라이브러리 목록입니다.

```bash
pip install torch torchvision opencv-python matplotlib ultralytics
```

---

## 🎯 3주차 진행 및 학습 과제
1. **데이터셋 준비**: Roboflow Universe 등에서 패션/의류 관련 YOLO 포맷 데이터셋을 다운로드받아 `data/` 폴더에 배치합니다.
2. **data.yaml 작성**: `config/data.yaml`에 데이터셋의 실제 경로와 탐지할 의류 클래스 정보를 입력합니다.
3. **학습 진행 (`src/train.py`)**: `yolov8n.pt` 모델을 사용하여 에포크 10 이상 학습을 진행하고 `best.pt` 가중치를 추출합니다.
4. **결과 시각화 (`src/inference.py`)**: OpenCV를 사용하여 이미지 상에 검출된 패션 아이템 영역에 바운딩 박스를 치고, 신뢰도(Confidence)와 클래스를 시각화합니다.
5. **성능 분석 (`src/evaluate.py`)**: `runs/` 폴더 밑의 `confusion_matrix.png`, Loss 곡선 등을 직접 확인하며 학습의 정상 여부를 진단합니다.
