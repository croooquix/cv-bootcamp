#### 기본적인 Depth Map 생성 코드 (OpenCV 활용)
import cv2
import numpy as np


def generate_depth_map(image):
    '''입력 이미지로부터 Grayscale 및 가상 Depth Map을 반환하는 함수'''
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 깊이 맵 생성 (가상의 깊이 적용)
    depth_map = cv2.applyColorMap(gray,  cv2.COLORMAP_JET)
    return depth_map, gray

# import 될 때는 안 돌고, 이 파일을 직접 python basic_depth.py 로 실행할 때만 돌게 만든다. 
if __name__ == '__main__':
    from config.config import IMAGE
    if IMAGE is not None:
        depth_map, _ = generate_depth_map(IMAGE)
        cv2.imshow('Depth Map', depth_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()