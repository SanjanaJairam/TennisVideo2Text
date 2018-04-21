import csv
import glob
import os

for i in range(1,3):
	root = "../../generated_features/train/"+str(i)+"/"

	feature_root = "../../features/train/"+str(i)

	if not os.path.exists(feature_root):
		os.makedirs(feature_root)



	for filename in glob.glob(os.path.join(root, '*.features')):

		if not os.path.exists(feature_root+"/info"):
			os.makedirs(feature_root+"/info")
			os.makedirs(feature_root+"/trajectory")
			os.makedirs(feature_root+"/hog")
			os.makedirs(feature_root+"/hof")
			os.makedirs(feature_root+"/mbhx")
			os.makedirs(feature_root+"/mbhy")

		csv_filename = os.path.basename(filename)

		info_trajectory_file = open(feature_root+"/info/"+csv_filename+".csv","w")
		trajectory_file = open(feature_root+"/trajectory/"+csv_filename+".csv","w")
		hog_file = open(feature_root+"/hog/"+csv_filename+".csv","w")
		hof_file = open(feature_root+"/hof/"+csv_filename+".csv","w")
		mbhx_file = open(feature_root+"/mbhx/"+csv_filename+".csv","w")
		mbhy_file = open(feature_root+"/mbhy/"+csv_filename+".csv","w")

		info_writer = csv.writer(info_trajectory_file)
		trajectory_writer = csv.writer(trajectory_file)
		hog_writer = csv.writer(hog_file)
		hof_writer = csv.writer(hof_file)
		mbhx_writer = csv.writer(mbhx_file)
		mbhy_writer = csv.writer(mbhy_file)

		f = open(filename,'r')
		lines = f.readlines()

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


		info_trajectory_file.close()
		trajectory_file.close()
		hog_file.close()
		hof_file.close()
		mbhx_file.close()
		mbhy_file.close()
