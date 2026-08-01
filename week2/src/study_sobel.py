import cv2
import numpy as np
import matplotlib.pyplot as plt
from config.config import IMAGE_PATH

def run_sobel_experiment(image_path):
    """
    Sobel 필터의 X축 미분(Gx), Y축 미분(Gy), 최종 합성 강도(G)를 눈으로 확인하는 실습 함수
    """
    # 1. 이미지 로드 및 그레이스케일 변환
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 찾을 수 없습니다: {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 노이즈를 살짝 줄여주면 미분 윤곽선이 훨씬 깔끔하게 나옵니다
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Sobel X (가로 방향 미분 -> 세로로 서 있는 수직 경계선 탐지)
    # cv2.CV_64F를 쓰는 이유: 미분값(차분)은 음수가 나올 수 있으므로 int8이 아닌 float64로 받아야 짤리지 않습니다!
    sobel_x_64 = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x_64)  # 절대값을 취해 0~255 이미지로 변환

    # 3. Sobel Y (세로 방향 미분 -> 가로로 누워 있는 수평 경계선 탐지)
    sobel_y_64 = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y_64)

    # 4. Combined Sobel (피타고라스 정리를 통한 최종 윤곽선 강도 G = np.sqrt(Gx^2 + Gy^2))
    sobel_combined_64 = np.sqrt(sobel_x_64**2 + sobel_y_64**2)
    sobel_combined = cv2.convertScaleAbs(sobel_combined_64)

    # 5. Matplotlib 4-Panel 한눈에 비교 시각화
    plt.figure(figsize=(12, 10))
    plt.suptitle("Sobel Filter Matrix Exploration (Gx vs Gy vs Combined)", fontsize=16, fontweight='bold')

    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("1. Original Grayscale Image", fontsize=13)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title("2. Sobel X (Detects Vertical Edges / |[-1,0,+1]|)", fontsize=13)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title("3. Sobel Y (Detects Horizontal Edges / |[-1,-2,-1]^T|)", fontsize=13)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title("4. Combined Sobel (Total Edge Magnitude G)", fontsize=13)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # cv2.imshow("Sobel X (Vertical Edges)", sobel_x)
    # cv2.imshow("Sobel Y (Horizontal Edges)", sobel_y)
    # cv2.imshow("Combined Sobel (Total Edges)", sobel_combined)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    run_sobel_experiment(IMAGE_PATH)
