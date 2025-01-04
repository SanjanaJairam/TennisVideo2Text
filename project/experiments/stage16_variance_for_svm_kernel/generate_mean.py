from __future__ import division
import csv
import glob
import numpy as np
import os
for i in range(1,3):
	
	histogram_root = "../../feature_files/features/histogram_files/"+str(i)+"/"
	
	count = 0
	mean_trajectory = 0
	mean_hog = 0
	mean_hof = 0
	mean_mbhx = 0
	mean_mbhy = 0
	
	trajectory_sum = [0]*4000 
	hog_sum = [0]*4000  
	hof_sum = [0]*4000 
	mbhx_sum = [0]*4000 
	mbhy_sum = [0]*4000 
	
	for filename in glob.glob(os.path.join(histogram_root, '*.csv')):
	
		count += 1
		
		histogram = open(filename,'r')
		
		data = np.loadtxt(fname=histogram, delimiter=',', dtype='int')
		
		histogram.close()
		
		trajectory_sum = [(x+y) for x,y in zip(trajectory_sum,list(data[0]))]
		hog_sum = [(x+y) for x,y in zip(trajectory_sum,list(data[1]))]
		hof_sum = [(x+y) for x,y in zip(trajectory_sum,list(data[2]))]
		mbhx_sum = [(x+y) for x,y in zip(trajectory_sum,list(data[3]))]
		mbhy_sum = [(x+y) for x,y in zip(trajectory_sum,list(data[4]))]
		
	
	if(count!=0):
		mean_trajectory = map(lambda x: round((x/count),2),trajectory_sum)
		mean_hog = map(lambda x: round((x/count),2),hog_sum)
		mean_hof = map(lambda x: round((x/count),2),hof_sum)
		mean_mbhx = map(lambda x: round((x/count),2),mbhx_sum)
		mean_mbhy = map(lambda x: round((x/count),2),mbhy_sum)
		
		mean_histogram = open(histogram_root+"mean.csv",'w')
		mean_writer = csv.writer(mean_histogram)
		mean_writer.writerow(mean_trajectory)
		mean_writer.writerow(mean_hog)
		mean_writer.writerow(mean_hof)
		mean_writer.writerow(mean_mbhx)
		mean_writer.writerow(mean_mbhy)
		
		mean_histogram.close()
		
			
			
			
			
			
		