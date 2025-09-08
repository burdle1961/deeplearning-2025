from flask import Flask, render_template, request

from ultralytics import YOLO

import numpy as np
import cv2
import pickle

model = YOLO('yolov8n.pt')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def show():
    return 'Hello YOLO'
    
@app.route('/detect', methods=['POST'])
def detect_object():

    # image = request.data
    # image = pickle.loads(image)
    np_arr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    res = model.predict(img, show=False, verbose=False)

    ret = res[0].to_json()
    
    return ret
   
app.run()
