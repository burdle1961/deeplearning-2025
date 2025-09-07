from flask import Flask, render_template, request

from ultralytics import YOLO

import numpy as np
import cv2
import pickle

def stringToRGB(base64_string):
    imgdata = base64.b64decode(base64_string)
    dataBytesIO = io.BytesIO(imgdata)
    image = Image.open(dataBytesIO)
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

model = YOLO('yolov8n.pt')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def show():
    return 'Hello YOLO'
    
@app.route('/detect', methods=['POST'])
def detect_object():

    image = request.data
    image = pickle.loads(image)
    res = model.predict(image, show=False, verbose=False)

    ret = res[0].to_json()
    
    return ret
   
app.run()
