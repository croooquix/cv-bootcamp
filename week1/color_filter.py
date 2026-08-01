import cv2
import numpy as np

# 빨간색 영역을 감지하고 필터링하는 예제 코드

# 1. 이미지 로드
image = cv2.imread("sample.jpg")  # 분석할 이미지 불러오기

# 만약 이미지가 없으면 에러 메시지 출력 후 종료
if image is None:
    print("에러: sample.jpg 파일을 찾을 수 없습니다! 폴더에 파일이 있는지 확인해주세요.")
    exit()

# 2. BGR 색상 공간에서 HSV 색상 공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 빨간색 범위 지정 (0~10 구간 & 170~180 구간)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 4. 마스크(흰색/검은색 필터) 생성 및 합치기
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2  # 두 빨간색 영역 합치기

# 5. 원본 이미지에서 마스크가 흰색(빨간색)인 부분만 추출
result = cv2.bitwise_and(image, image, mask=mask)

# 6. 결과 이미지 띄우기
cv2.imshow("Original", image)
cv2.imshow("Red Filtered", result)

print("키보드에서 아무 키나 누르면 창이 닫힙니다.")
cv2.waitKey(0)  # 아무 키나 누를 때까지 창 유지
cv2.destroyAllWindows()  # 모든 창 닫기