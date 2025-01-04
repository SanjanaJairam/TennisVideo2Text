import numpy as np		      # importing Numpy for use w/ OpenCV
import cv2                            # importing Python OpenCV
from datetime import datetime         # importing datetime for naming files w/ timestamp

def diffImg(t0, t1, t2):              # Function to calculate difference between images.
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

threshold = 78000                     # Threshold for triggering "motion detection"
cam = cv2.VideoCapture('../../data_input/1.avi')             # Lets initialize capture on webcam

# Read three images first:
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
# Lets use a time check so we only take 1 pic per sec
timeCheck = datetime.now().strftime('%Ss')

while True:
	ret, frame = cam.read()	      # read from camera
	totalDiff = cv2.countNonZero(diffImg(t_minus, t, t_plus))	# this is total difference number
	if totalDiff > threshold and timeCheck != datetime.now().strftime('%Ss'):	
		print('yolo')
		dimg= cam.read()[1]
		cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)
		timeCheck = datetime.now().strftime('%Ss')
		blur = cv2.medianBlur(dimg,7)

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
			cv2.rectangle(dimg,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.imshow('frame',dimg)

			#get the image inside bounded rectangle
			people = dimg[y:y + h, x:x + w]
			cv2.imshow('img',people)
			cv2.imwrite("../../data_output/cut_boxes/"+datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f')+".jpg",people)


	# Read next image
	t_minus = t
	t = t_plus
	next_frame = cam.read()[1]
	if(next_frame is not None):
		t_plus = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
		cv2.imshow("Frames", frame)
	else:
		break

	key = cv2.waitKey(40)
	if key == 27:			 # comment this 'if' to hide window
		cv2.destroyWindow(winName)
		break