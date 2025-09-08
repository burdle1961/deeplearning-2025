import requests
import cv2
import pickle


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while (True) :

    ret, frame = cap.read()
    if not(ret) : break  

    cv2.imshow('Camera Input', frame)
    key = cv2.waitKey(10)
    if (key == 27) :
        break
    elif (key == ord('s')) :
        # frame = pickle.dumps(frame)
        ret, buf = cv2.imencode('.jpg', frame)
        res = requests.post('http://127.0.0.1:5000/detect', data=buf.tobytes())
        print (res.text)
       
cap.release()
cv2.destroyAllWindows()


