#### 심화 코드: Depth Map을 기반으로 3D 포인트 클라우드 생성
import cv2
import numpy as np
from src.basic_depth import generate_depth_map

def generate_3d_points(image):
    """입력 BGR 이미지로부터 3D Point Cloud 좌표 배열(H, W, 3)을 생성합니다."""
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    depth_map, gray = generate_depth_map(image)
    h, w = depth_map.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32) # Depth 값을 Z 축으로 사용

    # 3D 좌표 생성
    return np.dstack((X, Y, Z))

if __name__ == '__main__':
    from config.config import IMAGE

    if IMAGE is not None:
        points_3d = generate_3d_points(IMAGE)
        print(points_3d.shape, points_3d.dtype, points_3d.min(), points_3d.max())
        # 결과 출력
        cv2.imshow('3D Point Cloud', points_3d)
        cv2.waitKey(0)
        cv2.destroyAllWindows()