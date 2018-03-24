import cv2
from skimage import feature
import pickle

'''
model = cv2.ml.SVM_create()
model.load('svm_data.dat')
model.predict(features)
'''
#img = cv2.imread('../../data_output/people/tagged/42/positive/11.jpg')
img = cv2.imread('../../data_output/people/tagged/45/negative/2.jpg')
#img = cv2.imread('../../data_output/tennis1.png')
#cv2.imshow("before_resize", img)
resize = cv2.resize(img,(64,128))
#cv2.imshow("after_resize", resize)
#cv2.waitKey(5000);
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
features = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), block_norm ='L2-Hys' ,transform_sqrt=True)

pickle_in = open('svm40.pickle','rb')
clf = pickle.load(pickle_in)
if(clf.predict([features])[0]==0):
    print("No player found")
else:
    print("Player found")
