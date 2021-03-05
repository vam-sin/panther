# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import biovec
import pickle
from sklearn.utils import shuffle

# # dataset import 
# ds = pd.read_csv('../data/dataset_non.csv')

# # 5,515 unique classes -> 5481 unique classes with more than one datapoint
# # 3898 unique classes with >= 100 training examples
# # removing those classes that have only one datapoint
# values = ds["class"].value_counts()
# to_remove = list(values[values < 100].index)
# ds = ds[ds["class"].isin(to_remove) == False]

# ds = ds.reset_index() 
# ds.columns = ["one", "two", "three", "sequence", "class"]
# ds = ds.drop(columns = ["one", "two", "three"])

# # biovec model
# vectorizer = biovec.models.load_protvec('SSG5.biovec')

# # X and y
# X = list(ds["sequence"])
# y = ds["class"]

# mini = -112.18917
# maxi = 115.491425
# sub = maxi - mini

# # X process
# # X_vec = []

# # for i in range(len(X)):
# # 	print(i, len(X))
# # 	vec = np.asarray(vectorizer.to_vecs(X[i]))

# # 	# min-max scaling
# # 	vec = vec - mini 
# # 	vec /= sub

# # 	X_vec.append(vec)

# # print("Converted X sequences to vectors")

# filename = 'X_BV_100_SSG5.pickle'
# # outfile = open(filename, 'wb')
# # pickle.dump(X_vec, outfile)
# # outfile.close()

# print("Loading X")
# infile = open(filename,'rb')
# X_vec = pickle.load(infile)
# infile.close()

# # split
# X_vec, y = shuffle(X_vec, y, random_state=42)
# print("Shuffled")

# X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size = 0.2, random_state = 42)
# X_train = np.asarray(X_train)
# X_train = np.reshape(X_train, (X_train.shape[0], 300))
# X_test = np.asarray(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], 300))
# print(X_train.shape, X_test.shape)

filename = 'X_train.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(X_train, outfile)
# outfile.close()

infile = open(filename,'rb')
X_train = pickle.load(infile)
infile.close()

filename = 'X_test.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(X_test, outfile)
# outfile.close()

infile = open(filename,'rb')
X_test = pickle.load(infile)
infile.close()

filename = 'y_train.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(y_train, outfile)
# outfile.close()

infile = open(filename,'rb')
y_train = pickle.load(infile)
infile.close()

filename = 'y_test.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(y_test, outfile)
# outfile.close()

infile = open(filename,'rb')
y_test = pickle.load(infile)
infile.close()

# KNN
f = open("results.txt", 'w')
f.write("Training\n")
print("Training")
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
f.write("Testing\n")
print("Testing")
score = str(knn.score(X_test, y_test))
f.write(score)
print(score)

f.close()


'''
Test Accuracy:
0.15765466909935313: Naive Bayes
0.9284790180792835: KNN (5)
'''