from ultralytics import YOLO
from datetime import datetime

model = YOLO ('d:/fish/runs/detect/train/weights/best.pt')

# Stream mode로 예측

print ("start prediction : ", datetime.now())
result = model.predict(source='d:/fish/tropicalfishes.mp4', show=False, verbose=False, stream=True)
print (type(result))        # <class 'generator'>
print ("finish prediction : ", datetime.now())

for res in result :

    if len(res.boxes.cpu().numpy()) > 0 :
        print (res.boxes.data.cpu().numpy())

# Stream mode off로 예측
# mp4 내의 모든 frame을 처리한 후에 result 값 반환 (영상이 긴 경우, memory 부족 발생 가능)
print ("start prediction : ", datetime.now())
result = model.predict(source='d:/fish/tropicalfishes.mp4', show=False, verbose=False, stream=False)
print (type(result))    # <class 'list'>
print ("finish prediction : ", datetime.now())

for res in result :

    if len(res.boxes.cpu().numpy()) > 0 :
        print (res.boxes.data.cpu().numpy())