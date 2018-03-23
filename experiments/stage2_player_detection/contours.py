import cv2
import numpy as np

img = cv2.imread('../../data_input/tennis1.png',0)
cv2.imshow('img',img)
cv2.waitKey(0)
ret,thresh = cv2.threshold(img,127,255,0)
print(ret,thresh)

_,contours,hierarchy = cv2.findContours(thresh, 1, 2)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 100 or area > 1000:
        continue
    if len(cnt) < 5: 
        continue 
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('frame',img)
    cv2.waitKey(40)



