import glob
import math
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel

samples = []
labels = []


for i in range(1,3):

    histogram_root = "../../histogram_files/train/"+str(i)+"/"

    files = glob.glob(os.path.join(histogram_root, '*.histogram.csv'))

    for filename in files:

        histogram = open(filename,'r')

        data = (np.loadtxt(fname=histogram, delimiter=',', dtype='int')).flatten().tolist()

        samples.append(data)

        labels.append(i)

        histogram.close()

samples = np.array(samples)
labels = np.array(labels)

print(labels)

clf = svm.SVC(kernel = chi2_kernel)
clf.fit(samples,labels)
with open('svm_part3.pickle','wb') as f:
	pickle.dump(clf, f)
#histogram_files/test/1/Diving_Side_012_flipped.vob.features.histogram.csv
#histogram_files/test/2/Walk_Front_014_flipped.vob.features.histogram.csv
#histogram_files/train/2/Walk_Front_005_flipped.vob.features.histogram.csv
#histogram_files/train/1/Diving_Side_009.vob.features.histogram.csv
histogram_test = open('../../histogram_files/train/1/Diving_Side_009.vob.features.histogram.csv','r')

test_data = (np.loadtxt(fname=histogram_test, delimiter=',', dtype='int')).flatten().tolist()

histogram.close()

print(clf.predict([test_data]))
