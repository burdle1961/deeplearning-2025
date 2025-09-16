import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

st.title("Webcam Object Detection with YOLO")
st.write("Predict 버튼을 클릭하면 현재 웹캠 이미지를 캡처해 객체를 검출합니다.")

# 웹캠 객체를 세션 상태에 저장(처음 한 번만 cap 생성)
if "cap" not in st.session_state:
    st.session_state["cap"] = cv2.VideoCapture(0)

cap = st.session_state["cap"]

# predict 버튼 추가
if st.button("predict()"):
    ret, frame = cap.read()
    if not ret:
        st.error("웹캠을 찾을 수 없습니다.")
    else:
        results = model(frame)[0]
        boxes = results.boxes
        img_bbox = results.plot()

        st.image(img_bbox, channels="BGR", caption="Detected Frame")

        detections = []
        for box in boxes:
            for cls in box.cls :
                cls_id = int(cls.cpu().numpy())
                cls_name = model.names[cls_id]
                coords = box.xyxy.cpu().numpy()
                detections.append({"Class": cls_name, "BBox": coords})

        st.subheader("Detected Objects")
        for det in detections:
            with st.expander(det["Class"]):
                st.write(f"BBox(x1,y1,x2,y2): {det['BBox']}")

# 앱 종료 시 cap.release() 
# (Streamlit에서는 자동화가 어려우니, 필요시 별도 버튼 또는 작업에서 지원)
