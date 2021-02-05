# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import biovec
import pickle

ds = pd.read_csv('../data/dataset_non.csv')

# 5,515 unique classes -> 5481 unique classes with more than one datapoint
# removing those classes that have only one datapoint
values = ds["class"].value_counts()
to_remove = list(values[values < 2].index)
ds = ds[ds["class"].isin(to_remove) == False]

ds = ds.reset_index() 
ds.columns = ["one", "two", "three", "sequence", "class"]
ds = ds.drop(columns = ["one", "two", "three"])

X = list(ds["sequence"])
y = ds["class"]

# print(X, len(y.unique()))
# y process
# 140 unique classes
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
# print(y, le.classes_)

# X process
bm = biovec.models.load_protvec('SSG5.biovec')

for i in range(len(X)):
	print(i, len(X))
	# try:
	vec = bm.to_vecs(X[i])
	vec = np.asarray(vec)
	filename = 'biovectors/' + str(i)
	np.save(filename, vec)
		# print(vec.shape)
	# except:
	# 	print("Except")
	# 	X_vec.append(np.random.uniform(0, 0.2, size=(3,100)))

print("Converted X sequences to vectors")

# filename = 'X_vec_ssg5.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(X_vec, outfile)
# outfile.close()

# infile = open(filename,'rb')
# X_vec = pickle.load(infile)
# infile.close()

# # split
# X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size = 0.2, random_state = 42)
# X_train = np.asarray(X_train)
# X_train = np.reshape(X_train, (X_train.shape[0], 300))
# X_test = np.asarray(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], 300))
# print(X_train.shape, X_test.shape)

# # logistic regression
# print("Training")
# clf = LogisticRegression(random_state=0, verbose=1, max_iter = 500).fit(X_train, y_train)
# print("Testing")
# score = clf.score(X_test, y_test)
# print(score)


'''
Test Accuracy:
0.5578371205838447: 100 iterations (did not converge)
0.5982418311494444: 200 iterations (did not converge)
0.6146458782551003: 500 iterations
'''