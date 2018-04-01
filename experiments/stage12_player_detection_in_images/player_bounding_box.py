import cv2
from skimage import feature
import pickle
import numpy as np

cap = cv2.VideoCapture('../../data_input/20.avi') #opening video file
pickle_in = open('svm40.pickle','rb') #opening svm model
clf = pickle.load(pickle_in)
# 360X640 -> size of each frame

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        x1 = 10
        y1 = 220
        while(y1<232):
            while(x1<576):
                height, width, channels = frame.shape
                print(height, width)
                ball = frame[220:340, 400:490]
                #ball = frame[220:340, 400:490]
                #ball = frame[220:348, 400:464] #aspect ratio maintained

                img = frame
                resize = cv2.resize(img,(64,128))
                gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
                features = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), block_norm ='L2-Hys' ,transform_sqrt=True)

                if(clf.predict([features])[0]==0):
                    print("No player found")
                else:
                    cv2.rectangle(frame,(x1,y1),(x1+64,y1+128),(0,255,0),2)
                    print("Player found")


                cv2.imshow("ouptut", frame)
                #cv2.imshow("ball", ball)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                x1 += 10
            y1 += 5

    cv2.waitKey(100)


cap.release()
cv2.destroyAllWindows()

'''
img = cv2.imread('../../data_output/people/tagged/42/positive/8.jpg')
resize = cv2.resize(img,(64,128))
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
features = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), block_norm ='L2-Hys' ,transform_sqrt=True)

pickle_in = open('svm40.pickle','rb')
clf = pickle.load(pickle_in)
if(clf.predict([features])[0]==0):
    print("No player found")
else:
    print("Player found")
'''
