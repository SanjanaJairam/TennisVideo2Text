import glob
import math
import numpy as np
import os
import pickle
from sklearn import svm

samples = []
labels = []
count= 0

def chi_squared(x,y):
	
	distance = 0
	
	for i in range(len(x)):
		if((x[i]+y[i]) == 0):
			distance += 0
		else:
			distance += round(((x[i]-y[i])**2)/((x[i]+y[i])*2),4)
	
	return distance
	
	
def custom_kernel(h1,h2,gamma_data):
	'''
	kernel_transform = map(lambda (x,y,gamma): gamma*chi_squared(h1,h2), zip(h1,h2,gamma_data) )
	kernel_sum = sum(kernel_transform)
	kernel_value = math.exp(-kernel_sum)
	return kernel_value
	'''
	for x in zip(h1,h2,gamma_data):
		print(x._2)
		break
	
histogram_root = "../../feature_files/features/histogram_files/1/"

gamma_root = histogram_root + "gamma.csv"
	
gamma_file = open(gamma_root,'r')

gamma_data = (np.loadtxt(fname=gamma_file, delimiter=',', dtype='float')).tolist()
	
gamma_file.close()

histogram1_root = histogram_root+"Diving_Side_001.vob.features.histogram.csv"
histogram2_root = histogram_root+"Diving_Side_002.vob.features.histogram.csv"

histogram1_file = open(histogram1_root,'r')
histogram2_file = open(histogram2_root,'r')

histogram1_data = (np.loadtxt(fname=histogram1_file, delimiter=',', dtype='float')).tolist()
histogram2_data = (np.loadtxt(fname=histogram2_file, delimiter=',', dtype='float')).tolist()
	
histogram1_file.close()	
histogram2_file.close()

print(custom_kernel(histogram1_data,histogram2_data,gamma_data))
	
	