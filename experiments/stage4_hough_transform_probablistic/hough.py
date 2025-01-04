import numpy as np
import cv2

cap = cv2.VideoCapture('../../data_input/20.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)


        cv2.imshow("ouptut", frame)

        cv2.waitKey(500)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
