import cv2
import imutils
import numpy as np
import os

for i in range(1,2):
    cap = cv2.VideoCapture('../../data_input/'+str(i)+'.avi')

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    noOfFrames = 0

    newpath = "../../data_output/people/"+str(i)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    while(noOfFrames<10):
        ret, frame = cap.read()

        #background subtractor mask applied on frame
        fgmask = fgbg.apply(frame)

        #it is used for morphological operations as a filter
        kernel = np.ones((3, 3), np.uint8)
        #
        #gradient = cv2.morphologyEx(fgmask, cv2.MORPH_GRADIENT, kernel)
        if(fgmask is not None):
        #cv2.imshow('frame',fgmask)

        # removing noise (salt and pepper) by 70%
            blur = cv2.medianBlur(fgmask,7)

            #filling the boundaries detected to find objects
            closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=4)

            #cv2.imshow('closing',closing)

            #finding contours: closed shapes
            _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                # discard areas that are too small
                if area < 100 or area > 1000:
                    continue

                #paramters for drawing the bounding rectangle
                x,y,w,h = cv2.boundingRect(cnt)


                #draw the bounding rectangle
                # cv2.rectangle(frame,(x-w,y-h),(x+w*2,y+h*2),(0,255,0),2)
                # cv2.imshow('frame',frame)

                # get the image inside bounded rectangle
                people = frame[y-h:y + h*2, x-w:x + w*2]
                if(people.size>0):
                    cv2.imshow('img',people)
                    noOfFrames += 1
                    cv2.waitKey(500)
                    #cv2.imwrite(newpath+"/"+str(noOfFrames)+".jpg",people)






            k = cv2.waitKey(50) & 0xff
            if k == 27:
                break


    cap.release()
    cv2.destroyAllWindows()
