from ultralytics import YOLO
import cv2
model = YOLO ('E:/fish/runs/detect/train/weights/best.pt')
cap = cv2.VideoCapture('./fish2.mp4')
frameCount = 0
while (True) :
    ret, frame = cap.read()
    if (not(ret)) : break
    
    frame = cv2.resize(frame, dsize=(640, 360))
    result = model.predict(source=frame, show=True, verbose=False, stream=False, conf=0.7, imgsz=640)
    res = result[0]  # cap.read()에서 한장의 이미지 프레임만 읽어 예측했기 때문에, result[0]

    for box in res.boxes :
        print(f"FrameCount = {frameCount}, {box.data.cpu().numpy()}")
        npp = box.xyxy.cpu().numpy()
        npcls = box.cls.cpu().numpy()
        cx = int((npp[0][0]+npp[0][2])/2)
        cy = int((npp[0][1]+npp[0][3])/2)
        frame = cv2.circle(frame, (cx, cy),30, (0,255,255), 3)
        cv2.putText(frame, res.names[npcls[0]] , org=(cx,cy), color=(0,255,0), fontScale=1, thickness=2, lineType = cv2.LINE_AA, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        cv2.imshow('Detected Object', frame) 
        if (cv2.waitKey(1) == 27) :
            break
        frameCount += 1
cap.release()
cv2.destroyAllWindows()
