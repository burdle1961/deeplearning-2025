from ultralytics import YOLO

model = YOLO ('d:/fish/runs/detect/train/weights/best.pt')

frameCount = 0
result = model.predict(source='d:/fish/fish2.mp4', show=True, verbose=False, stream=True, conf=0.7, imgsz=640)

for res in result :

    if (res.boxes) :
        print(f"FrameCount = {frameCount}, {res.boxes.data.cpu().numpy()}")
            
    frameCount += 1













