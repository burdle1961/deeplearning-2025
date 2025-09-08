import requests
import cv2
import pickle
import json


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

def print_bbox(img, bbox) :
    
    for box in bbox :

        pt1 = (int(box['box']['x1']), int(box['box']['y1']))
        pt2 = (int(box['box']['x2']), int(box['box']['y2']))
        cv2.rectangle(img, pt1, pt2, (0,255,255), 2)
        
        pt3 = (pt1[0], pt1[1] - 10)
        cv2.putText(img, f"{box['name']},conf = {box['confidence']}", org=pt3,
                    color=(0,255,0), fontScale=1, thickness=2, lineType = cv2.LINE_AA, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

    return

while (True) :

    ret, frame = cap.read()
    if not(ret) : break  

    cv2.imshow('Camera Input', frame)
    key = cv2.waitKey(10)
    if (key == 27) :
        break
    elif (key == ord('s')) :
        ret, buf = cv2.imencode('.jpg', frame)
        res = requests.post('http://127.0.0.1:5000/detect', data=buf.tobytes())
        print (res.text)
        print_bbox(frame, json.loads(res.text))
        
        cv2.imshow('Detected Object', frame) 
        if (cv2.waitKey(1) == 27) :
            break
       
cap.release()
cv2.destroyAllWindows()


