import cv2
import glob
import numpy as np
import os
from skimage import feature
import pickle

'''
model = cv2.ml.SVM_create()
model.load('svm_data.dat')
model.predict(features)
'''

pickle_in = open('svm40.pickle','rb')
clf = pickle.load(pickle_in)
samples = []
labels = []
predictions = []

for i in range(41,51):
    positivePath = "../../data_output/people/tagged/"+str(i)+"/positive/"
    negativePath = "../../data_output/people/tagged/"+str(i)+"/negative/"


    # Get positive samples
    for filename in glob.glob(os.path.join(positivePath, '*.jpg')):
        img = cv2.imread(filename)
        resize = cv2.resize(img,(64,128))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        features = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
		    cells_per_block=(2, 2), block_norm ='L2-Hys' ,transform_sqrt=True)
        samples.append(features)
        labels.append(1)
        predictions.append(clf.predict([features])[0])


# '''
# Get negative samples

    for filename in glob.glob(os.path.join(negativePath, '*.jpg')):
        img = cv2.imread(filename)
        resize = cv2.resize(img,(64,128))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        features = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm ='L2-Hys' ,transform_sqrt=True)
        samples.append(features)
        labels.append(0)
        predictions.append(clf.predict([features])[0])

correct = 0
for i in range(len(labels)):
    if(labels[i]==predictions[i]):
        correct += 1
print(correct*100.0/len(labels))
