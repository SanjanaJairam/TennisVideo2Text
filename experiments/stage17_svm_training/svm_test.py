import numpy as np
import pickle

histogram = open('../../histogram_files/test/1/Diving_Side_012.vob.features.histogram.csv','r')

data = (np.loadtxt(fname=histogram, delimiter=',', dtype='int')).flatten().tolist()

histogram.close()

pickle_in = open('svm.pickle','rb')
clf = pickle.load(pickle_in)
print(np.dot(clf.dual_coef_,data))
'''
if(clf.predict([data])[0]==1):
    print("Diving")
else:
    print("Walking")
'''
