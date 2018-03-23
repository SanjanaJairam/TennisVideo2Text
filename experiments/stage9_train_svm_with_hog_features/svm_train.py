import cv2
import glob
import numpy as np
import os
import pickle
from skimage import feature
from sklearn import svm

samples = []
labels = []    
count= 0
for i in range(1,51):
    positivePath = "../../data_output/people/tagged/"+str(i)+"/positive/"
    negativePath = "../../data_output/people/tagged/"+str(i)+"/negative/"


    # Get positive samples
    for filename in glob.glob(os.path.join(positivePath, '*.jpg')):
        count+=1
        img = cv2.imread(filename)
        resize = cv2.resize(img,(64,128))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        features = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
		    cells_per_block=(2, 2), block_norm ='L2-Hys' ,transform_sqrt=True)
        samples.append(features)
        labels.append(1)
    

# '''
# Get negative samples

    for filename in glob.glob(os.path.join(negativePath, '*.jpg')):
        count+=1
        img = cv2.imread(filename)
        resize = cv2.resize(img,(64,128))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        features = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm ='L2-Hys' ,transform_sqrt=True)
        samples.append(features)
        labels.append(0)

print(len(labels))
print(count)
#Convert objects to Numpy Objects
samples = np.float32(samples)
labels = np.array(labels)


# Shuffle Samples
rand = np.random.RandomState(321)
shuffle = rand.permutation(len(samples))
samples = samples[shuffle]
labels = labels[shuffle]    

clf = svm.SVC()
clf.fit(samples, labels)  
with open('svm.pickle','wb') as f:
    pickle.dump(clf, f)

