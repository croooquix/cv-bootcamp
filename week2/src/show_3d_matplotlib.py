import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.advanced_3d import generate_3d_points
from config.config import IMAGE


def visualize_3d_matplotlib(image, step=5):
    '''Matplotlib을 활용한 3D 포인트 클라우드 시각화 함수'''
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    
    # 3D 포인트 클라우드 생성
    points_3d = generate_3d_points(image)
    
    # 1. 렌더링 속도를 위해 5픽셀 간격으로 샘플링 (Downsampling)
    X_sub = points_3d[::step, ::step, 0].flatten()
    Y_sub = points_3d[::step, ::step, 1].flatten()
    Z_sub = points_3d[::step, ::step, 2].flatten()

    # 원래 이미지의 색상(RGB)도 함께 가져와서 3D 점들의 색으로 지정
    color_sub = cv2.cvtColor(image[::step, ::step], cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0 # 0~1 범위로 정규화

    # 2. Matplotlib 3D 공간에 점 찍기
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    # 3D 산점도(Scatter Plot) 그리기
    ax.scatter(X_sub, Y_sub, Z_sub, c=color_sub, s=2)

    # 시각화 편의를 위해 Y축(위아래)을 뒤집어 사진 방향과 맞춤
    ax.invert_yaxis()

    ax.set_title("3D Point Cloud Visualization")
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Height)")
    ax.set_zlabel("Z (Depth / Brightness)")

    plt.show()

