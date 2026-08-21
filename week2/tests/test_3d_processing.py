import numpy as np
import pytest
import cv2
from src.basic_depth import generate_depth_map
from src.advanced_3d import generate_3d_points
from config.config import IMAGE

# pytest를 활용한 기본 Unit Test

def test_generate_depth_map():
    """정상 이미지 로드 시 2D Depth Map의 shape 일치 여부 검증"""
    depth_map, gray = generate_depth_map(IMAGE)
    assert depth_map.shape == IMAGE.shape, "Depth map 크기가 입력 이미지와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."

def test_generate_depth_map_with_none():
    """None 입력 시 예외(ValueError) 방어 검증"""
    with pytest.raises(ValueError, match="입력된 이미지가 없습니다."):
        generate_depth_map(None)

def test_generate_3d_points():
    """3D Point Cloud 생성 시 (H, W, 3) 입체 배열 shape 검증"""
    points_3d = generate_3d_points(IMAGE)
    assert points_3d.shape == IMAGE.shape, "3D Point Cloud 크기가 입력 이미지와 다릅니다."
    assert points_3d.dtype == np.float64 or points_3d.dtype == np.float32
    
def test_random_image_robustness():
    """랜덤 해상도(50x50) 이미지 입력 시 파이프라인 강건성 검증"""
    random_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    points_3d = generate_3d_points(random_img)
    assert points_3d.shape == (50, 50, 3)

# pytest 실행
if __name__ == "__main__":
    pytest.main()