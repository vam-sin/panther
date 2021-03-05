# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# load files
filename = 'processed_data/X_BV_100_SSG5_Full.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(X_vec, outfile)
# outfile.close()
infile = open(filename,'rb')
X = pickle.load(infile)
infile.close()

filename = 'processed_data/y_BV_100_SSG5_Full.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(y_vec, outfile)
# outfile.close()
infile = open(filename,'rb')
y = pickle.load(infile)
infile.close()

# y process
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))
print(num_classes)
print("Loaded X and y")

X, y = shuffle(X, y, random_state=42)
print("Shuffled")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Conducted Train-Test Split")

num_classes_train = len(np.unique(y_train))
num_classes_test = len(np.unique(y_test))
print(num_classes_train, num_classes_test)

#assert num_classes_test == num_classes_train, "Split not conducted correctly"

# # logistic regression
# f = open("log_reg_results.txt", "w")
print("Training\n")
clf = LogisticRegression(random_state=0, verbose=1, max_iter = 100).fit(X_train, y_train)
print("Testing\n")
score = str(clf.score(X_test, y_test))
print(score)


'''
Test Accuracy:

'''