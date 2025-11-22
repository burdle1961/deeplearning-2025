import requests
import cv2

# 전송할 JPEG 파일 경로 지정
img_path = './yolo.jpg'  # 원하는 이미지 파일명/경로로 수정

# 이미지 파일을 읽고, OpenCV로 디코드
with open(img_path, 'rb') as f:
    img_bytes = f.read()

# POST 요청: 그대로 바이너리 전송 (이미 JPEG 포맷)
res = requests.post('http://127.0.0.1:5000/detect', data=img_bytes)
print(res.text)