import csv
import glob
import numpy as np
import os
import pickle
from random import shuffle
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel


samples = []
labels = []
for i in range(1,3):
    for j in range(10):
        if(i==1):
            x = [[i] for i in range(5)]
            shuffle(x)
            samples.append((np.array(x)).flatten())
            labels.append(i)
        else:
            x = [[i] for i in range(5,10)]
            shuffle(x)
            samples.append((np.array(x)).flatten())
            labels.append(i)


labels = np.array(labels)
clf = svm.SVC(kernel = chi2_kernel)
clf.fit(samples,labels)
with open('svm_basic1.pickle','wb') as f:
	pickle.dump(clf, f)

x = [[i] for i in range(5,10)]
shuffle(x)
histogram = np.array(x)
histogram = histogram.flatten()
print(clf.predict([histogram]))

#print(clf.predict(kernel_test))
