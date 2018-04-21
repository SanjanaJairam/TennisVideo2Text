import numpy as np
import csv
import glob
import os
import pickle

pickle_root = "../../pickle_files/"

kmeans_trajectory = pickle.load(open(pickle_root+"trajectory.pkl",'rb'))
kmeans_hog = pickle.load(open(pickle_root+"hog.pkl",'rb'))
kmeans_hof = pickle.load(open(pickle_root+"hof.pkl",'rb'))
kmeans_mbhx = pickle.load(open(pickle_root+"mbhx.pkl",'rb'))
kmeans_mbhy = pickle.load(open(pickle_root+"mbhy.pkl",'rb'))

for i in range(1,3):

	root = "../../generated_features/train/"+str(i)+"/"

	for filename in glob.glob(os.path.join(root, '*.features')):

		predictions_trajectory = []
		predictions_hog = []
		predictions_hof = []
		predictions_mbhx = []
		predictions_mbhy = []


		f = open(filename,'r')
		lines = f.readlines()

		for line in lines:
			feature = line.split('\t')

			trajectory = feature[10:40]
			predict = kmeans_trajectory.predict([trajectory])
			predictions_trajectory.append(predict[0])


			hog = feature[40:136]
			predict = kmeans_hog.predict([hog])
			predictions_hog.append(predict[0])

			hof = feature[136:244]
			predict = kmeans_hof.predict([hof])
			predictions_hof.append(predict[0])

			mbhx = feature[244:340]
			predict = kmeans_mbhx.predict([mbhx])
			predictions_mbhx.append(predict[0])

			mbhy = feature[340:436]
			predict = kmeans_mbhy.predict([mbhy])
			predictions_mbhy.append(predict[0])

		f.close()

		cluster_size = np.arange(1,4002)

		histogram_trajectory = np.histogram(predictions_trajectory,bins=cluster_size)
		histogram_hog = np.histogram(predictions_hog,bins=cluster_size)
		histogram_hof = np.histogram(predictions_hof,bins=cluster_size)
		histogram_mbhx = np.histogram(predictions_mbhx,bins=cluster_size)
		histogram_mbhy = np.histogram(predictions_mbhy,bins=cluster_size)

		histogram_root = "../../histogram_files/train/"+str(i)

		if not os.path.exists(histogram_root):
			os.makedirs(histogram_root)

		histogram_filename = os.path.basename(filename)

		histogram_file = open(histogram_root+"/"+histogram_filename+".histogram.csv","w")

		histogram_writer = csv.writer(histogram_file)

		histogram_writer.writerow(histogram_trajectory[0])
		histogram_writer.writerow(histogram_hog[0])
		histogram_writer.writerow(histogram_hof[0])
		histogram_writer.writerow(histogram_mbhx[0])
		histogram_writer.writerow(histogram_mbhy[0])

		histogram_file.close()
