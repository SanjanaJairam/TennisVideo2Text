import csv
import glob
import numpy as np
import os 

def chi_squared(x,y):
	
	distance = 0
	
	for i in range(len(x)):
		if((x[i]+y[i]) == 0):
			distance += 0
		else:
			distance += round(((x[i]-y[i])**2)/((x[i]+y[i])*2),4)
	
	return distance
	
	
for i in range(1,3):
	
	histogram_root = "../../feature_files/features/histogram_files/"+str(i)+"/"
	histogram_mean_file = open(histogram_root+"mean.csv",'r')
	
	histogram_mean_data = (np.loadtxt(fname=histogram_mean_file, delimiter=',', dtype='float')).tolist()
	
	histogram_mean_file.close()
	
	chi_squared_sum = [0]*5
	
	gamma = [1]*5
	
	files = glob.glob(os.path.join(histogram_root, '*.histogram.csv'))
	
	count = len(files)
	
	for filename in files:
		
		histogram = open(filename,'r')
		
		data = (np.loadtxt(fname=histogram, delimiter=',', dtype='int')).tolist()
		
		histogram.close()
		
		chi_squared_distance = map(lambda (x,y): chi_squared(x,y), zip(data,histogram_mean_data))
		
		chi_squared_sum = map(lambda (x,y): x+y , zip(chi_squared_sum,chi_squared_distance))
		
		
	chi_squared_mean = map(lambda x: round((x/count),4),chi_squared_sum)
		
	gamma = map(lambda (x,y): round(x/y,6), zip(gamma,chi_squared_mean))
		
	gamma_file = open(histogram_root+'gamma.csv','w')
	
	gamma_writer = csv.writer(gamma_file)
		
	gamma_writer.writerow(gamma)
	
	gamma_file.close()