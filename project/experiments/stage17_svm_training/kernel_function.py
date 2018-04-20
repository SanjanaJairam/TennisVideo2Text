import glob
import math
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel

samples = []
labels = []


def chi_squared(x,y):

	distance = 0

	for i in range(len(x)):
		if((x[i]+y[i]) == 0):
			distance += 0
		else:
			distance += round(((x[i]-y[i])**2)/((x[i]+y[i])*2),4)

	return distance


def custom_kernel(h1,h2):
	kernel_transform = []
	h1_temp = []
	h2_temp = []
	n = 4000
	for i in range(0, len(h1), n):
		h1_temp.append(h1[i:i + n])
		h2_temp.append(h2[i:i + n])

	print(h1 == h2)

	for x in zip(h1_temp,h2_temp,gamma_data):
		kernel_transform.append(x[2]*chi_squared(x[0].flatten(),x[1].flatten()))
		print("*\n")

	kernel_sum = sum(kernel_transform)
	kernel_value = math.exp(-kernel_sum)
	return kernel_value


for i in range(1,3):

	histogram_root = "../../histogram_files/"+str(i)+"/"


	gamma_root = histogram_root + "gamma.csv"

	gamma_file = open(gamma_root,'r')
	global gamma_data
	gamma_data = (np.loadtxt(fname=gamma_file, delimiter=',', dtype='float')).tolist()

	gamma_file.close()

	files = glob.glob(os.path.join(histogram_root, '*.histogram.csv'))

	for filename in files:

		histogram = open(filename,'r')

		data = (np.loadtxt(fname=histogram, delimiter=',', dtype='int')).flatten().tolist()

		histogram.close()

		samples.append(data)

		labels.append(i)

samples = np.array(samples)
labels = np.array(labels)


clf = svm.SVC(kernel = custom_kernel)
clf.fit(samples, labels)
'''
with open('svm.pickle','wb') as f:
	pickle.dump(clf, f)
'''
