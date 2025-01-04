import glob
import math
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel

samples = []
labels = []
n = 4000
def custom_kernel(h1):

	K = []
	K_data = []
	kernel_data = [[],[],[],[],[]]
'''
	for x in h1:
		for i in range(0,len(x),n):
			kernel_data[i//4000].append(x[i:i+n])

	for j in range(5):
		K_data.append(chi2_kernel(kernel_data[j],gamma = gamma_data[j]))

	l = list(zip(K_data[0],K_data[1],K_data[2],K_data[3],K_data[4]))

	for x in l:
		K.append(list(map(lambda x:x[0]*x[1]*x[2]*x[3]*x[4] ,list(zip(x[0],x[1],x[2],x[3],x[4])))))

	return K

	K_trajectory = chi2_kernel(kernel_trajectory,gamma=gamma_data[0])
	K_hog = chi2_kernel(kernel_hog,gamma=gamma_data[0])
	K_trajectory = chi2_kernel(kernel_trajectory,gamma=gamma_data[0])
	K_trajectory = chi2_kernel(kernel_trajectory,gamma=gamma_data[0])
	K_trajectory = chi2_kernel(kernel_trajectory,gamma=gamma_data[0])
			print(K)
			break

		break

	print(h1 == h2)

	for x in zip(h1_temp,h2_temp,gamma_data):
		kernel_transform.append(x[2]*chi_squared(x[0].flatten(),x[1].flatten()))
		print("*\n")

	kernel_sum = sum(kernel_transform)
	kernel_value = math.exp(-kernel_sum)
	return kernel_value
'''

for i in range(1,3):

	histogram_root = "../../histogram_files/train/"+str(i)+"/"


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

K = custom_kernel(samples)

clf = svm.SVC(kernel = 'precomputed',degree=2)
clf.fit(K,labels)
with open('svm1.pickle','wb') as f:
	pickle.dump(clf, f)


histogram = open('../../histogram_files/test/1/Diving_Side_012.vob.features.histogram.csv','r')

test_data = (np.loadtxt(fname=histogram, delimiter=',', dtype='int')).flatten().tolist()

histogram.close()

kernel_test = np.dot(test_data,np.array(K))

print(kernel_test)

#print(clf.predict(kernel_test))
