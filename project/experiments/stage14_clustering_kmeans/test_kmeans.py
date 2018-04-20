import pickle

pickle_root = "../../feature_files/pickle_files/"
loaded_kmeans = pickle.load(open(pickle_root+"hog.pkl",'r'))

print(loaded_kmeans.labels_)