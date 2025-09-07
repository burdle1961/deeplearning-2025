from ultralytics import YOLO
#import cv2

model = YOLO ('d:/fish/runs/detect/train/weights/best.pt')
#cap = cv2.VideoCapture('d:/fish/tropicalfishes.mp4')

frameCount = 0
result = model.predict(source='d:/fish/fish2.mp4', show=True, verbose=False, stream=True, conf=0.7, imgsz=640)

for res in result :
    # ret, frame = cap.read()
    # if (not(ret)) : break
    
    # frame = cv2.resize(frame, dsize=(640, 360))
    if (res.boxes) :
        print(f"FrameCount = {frameCount}, {res.boxes.data.cpu().numpy()}")    
    frameCount += 1

    # for r in result[0] :
    #     npp = r.boxes.xyxy.numpy()
    #     npcls = r.boxes.cls.numpy()
    #     cx = int((npp[0][0]+npp[0][2])/2)
    #     cy = int((npp[0][1]+npp[0][3])/2)

#     cx = 320
#     cy = 160
#     frame = cv2.circle(frame, (cx, cy),30, (0,255,255), 3)
#     cv2.putText(frame,str(frameCount) , org=(cx,cy), color=(0,255,0), fontScale=1, thickness=2, lineType = cv2.LINE_AA, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
#     frameCount += 1

#     cv2.imshow('Input Stream', frame) 
#     if (cv2.waitKey(1) == 27) :
#         break
    
# cap.release()
# cv2.destroyAllWindows()













