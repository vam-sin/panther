# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import biovec
import pickle

ds = pd.read_csv('../data/dataset_ssg5.csv')

X = list(ds["SEQUENCE"])
y = ds["SSG5_CLUSTER"]

# print(X, len(y.unique()))
# y process
# 140 unique classes
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
# print(y, le.classes_)

# X process
# bm = biovec.models.load_protvec('SSG5.biovec')
# X_vec = []

# for i in range(len(X)):
# 	print(i)
# 	X_vec.append(bm.to_vecs(X[i]))

# print("Converted X sequences to vectors")

filename = 'X_vec_ssg5.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(X_vec, outfile)
# outfile.close()

infile = open(filename,'rb')
X_vec = pickle.load(infile)
infile.close()

# split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size = 0.2, random_state = 42)
X_train = np.asarray(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], 300))
X_test = np.asarray(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], 300))
print(X_train.shape, X_test.shape)

# KNN
print("Training")
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
print("Testing")
score = knn.score(X_test, y_test)
print(score)


'''
Test Accuracy:
0.15765466909935313: Naive Bayes
0.9284790180792835: KNN (5)
'''