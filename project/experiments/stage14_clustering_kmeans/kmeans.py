from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
import csv
import pickle

pickle_root = "../../feature_files/features/pickle_files/"

for i in range(1,3):
	#root = "../../feature_files/features/"+str(i)+"/hog/"
	#root = "../../feature_files/features/"+str(i)+"/hof/"
	#root = "../../feature_files/features/"+str(i)+"/trajectory/"
	#root = "../../feature_files/features/"+str(i)+"/mbhx/"
	root = "../../feature_files/features/"+str(i)+"/mbhy/"
	

	kmeans = MiniBatchKMeans(n_clusters=4000, random_state=0)	

	for files in os.walk(root):
		for file in files[2]:
			print(file)
			features = open(root+file,"rb")
			data = np.loadtxt(fname=features, delimiter=',')
			kmeans.partial_fit(data)

#pickle.dump(kmeans,open(pickle_root+"hog.pkl",'wb'))
#pickle.dump(kmeans,open(pickle_root+"hof.pkl",'wb'))
#pickle.dump(kmeans,open(pickle_root+"trajectory.pkl",'wb'))
#pickle.dump(kmeans,open(pickle_root+"mbhx.pkl",'wb'))
pickle.dump(kmeans,open(pickle_root+"mbhy.pkl",'wb'))
	
features.close()
	
#print(len(data))

'''
kmeans = KMeans(n_clusters=2000, random_state=0).fit(data)
	
pickle.dump(kmeans, open("hog.pkl", 'wb'))

print(kmeans.labels_)
	
features.close()



line = f.readlines()
diving_hog_features = line[0].split(" ")
print(diving_hog_features)
print(len(diving_hog_features))


   

for files in os.walk(diving_root):
	for file in files[2]:
		print(file)

for files in os.walk(walking_root):
	for file in files[2]:
		print(file)

#array of arrays of features
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
			  
#clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

#predict which cluster it belongs to 
print(kmeans.predict([[0, 0], [4, 4]]))

#attributes
print(kmeans.labels_)
print(kmeans.cluster_centers_)
'''