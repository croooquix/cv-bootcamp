import os
import cv2

# 현재 파일(config.py)의 상위 폴더의 상위 폴더(= week2)를 기준으로 data/sample.jpg 경로 지정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
IMAGE_PATH = os.path.join(DATA_DIR, "sample.jpg")

# 다른 모듈에서 'from config.config import IMAGE' 로 가져다 쓸 공통 이미지 변수
IMAGE = cv2.imread(IMAGE_PATH)