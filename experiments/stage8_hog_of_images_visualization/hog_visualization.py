from skimage import exposure
from skimage import feature
import cv2

for i in range(1,11):
	image = cv2.imread('../../data_output/people/generated/1/'+str(i)+'.jpg')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	(features, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
		cells_per_block=(2, 2), block_norm ='L2-Hys' ,visualise=True,transform_sqrt=True)
	print(features)
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")

	cv2.imshow("HOG Image", hogImage)
	#cv2.imwrite("../../data_output/hog_visualizations/"+str(i)+'.jpg',hogImage)
	cv2.waitKey(0)
