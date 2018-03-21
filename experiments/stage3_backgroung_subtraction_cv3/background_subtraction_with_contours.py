import numpy as np
import cv2

cap = cv2.VideoCapture('../../data_input/17.avi')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    #
    #gradient = cv2.morphologyEx(fgmask, cv2.MORPH_GRADIENT, kernel)
    if(fgmask is None):
        pass
    else:
        #cv2.imshow('frame',fgmask)
        blur = cv2.medianBlur(fgmask,7)
        closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=4)
        #cv2.imshow('closing',closing)
        cont_img = closing.copy()
        _, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 1000:
                continue
            if len(cnt) < 5: 
                continue 
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow('frame',frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
