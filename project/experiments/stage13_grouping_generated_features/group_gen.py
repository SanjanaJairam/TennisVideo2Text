import csv
import glob
import os 

for i in range(1,3):
	root = "../../feature_files/generated_features/train/"+str(i)+"/"
	
	feature_root = "../../feature_files/features/"+str(i)

	if not os.path.exists(feature_root):
		os.makedirs(feature_root)
		
	info_trajectory_file = open(feature_root+"/info.csv","w")
	trajectory_file = open(feature_root+"/trajectory.csv","w")
	hog_file = open(feature_root+"/hog.csv","w")
	hof_file = open(feature_root+"/hof.csv","w")
	mbhx_file = open(feature_root+"/mbhx.csv","w")
	mbhy_file = open(feature_root+"/mbhy.csv","w")	

	info_writer = csv.writer(info_trajectory_file)
	trajectory_writer = csv.writer(trajectory_file)
	hog_writer = csv.writer(hog_file)
	hof_writer = csv.writer(hof_file)
	mbhx_writer = csv.writer(mbhx_file)
	mbhy_writer = csv.writer(mbhy_file)

	for filename in glob.glob(os.path.join(root, '*.features')):
		f = open(filename,'r')
		lines = f.readlines()
		file_sizes = open(feature_root+"/file_sizes.txt","w")
		file_sizes.write(str(len(lines))+'\n')
		
		for line in lines:
			feature = line.split('\t')
			
			info_trajectory = feature[0:10]
			info_writer.writerow(info_trajectory)
			
			trajectory = feature[10:40]
			trajectory_writer.writerow(trajectory)
			
			hog = feature[40:136]
			hog_writer.writerow(hog)
			
			hof = feature[136:244]
			hof_writer.writerow(hof)
			
			mbhx = feature[244:340]
			mbhx_writer.writerow(mbhx)
			
			mbhy = feature[340:436]
			mbhy_writer.writerow(mbhy)

	file_sizes.close()
	info_trajectory_file.close()
	trajectory_file.close()
	hog_file.close()
	hof_file.close()
	mbhx_file.close()
	mbhy_file.close()	
	