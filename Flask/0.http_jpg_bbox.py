import requests
import cv2
import json


def print_bbox(img, bbox) :
    
    for box in bbox :

        pt1 = (int(box['box']['x1']), int(box['box']['y1']))
        pt2 = (int(box['box']['x2']), int(box['box']['y2']))
        cv2.rectangle(img, pt1, pt2, (0,255,255), 2)
        
        pt3 = (pt1[0], pt1[1] - 10)
        cv2.putText(img, f"{box['name']},conf = {box['confidence']}", org=pt3,
                    color=(0,255,0), fontScale=1, thickness=2, lineType = cv2.LINE_AA, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

    return

# 1. 이미지 파일 읽기 및 윈도우 출력
img_path = './yolo.jpg'
img = cv2.imread(img_path)
cv2.imshow('Original', img)
# cv2.waitKey(0) 

# 2. 서버에 전송 (JPEG 그대로 바이너리 전송)
with open(img_path, 'rb') as f:
    img_bytes = f.read()
res = requests.post('http://127.0.0.1:5000/detect', data=img_bytes)

# 3. 바운딩 박스 정보 파싱 (predict().tojson() response 구조에 따라 수정)
results = json.loads(res.text)
draw_img = img.copy()
print_bbox(draw_img, json.loads(res.text))

# 4. 바운딩 박스 그려진 결과를 다른 윈도우에 출력
cv2.imshow('Detected', draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
