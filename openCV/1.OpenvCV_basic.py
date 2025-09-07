import cv2

cap = cv2.VideoCapture('d:/fish/fish2.mp4')

frameCount = 0
while (True) :
  
    ret, frame = cap.read()
    if (not(ret)) : break
    
    frame = cv2.resize(frame, dsize=(640, 360))
    
    cx = 320
    cy = 160
    frame = cv2.circle(frame, (cx, cy),30, (0,255,255), 3)
    cv2.putText(frame,str(frameCount) , org=(cx,cy), color=(0,255,0), fontScale=1, thickness=2, lineType = cv2.LINE_AA, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    frameCount += 1

    cv2.imshow('Input Stream', frame) 
    if (cv2.waitKey(1) == 27) :
        break
    
cap.release()
cv2.destroyAllWindows()













