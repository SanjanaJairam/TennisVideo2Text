from sklearn.cluster import KMeans
import numpy as np

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
