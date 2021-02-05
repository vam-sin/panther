import biovec
import pandas as pd 
import numpy as np 
from sklearn import preprocessing

ds = pd.read_csv('../data/dataset_non.csv')

X = list(ds["sequence"])

# X_non = []

# for i in range(len(X)):
# 	print(i)
# 	X_non.append(X[i].replace('\n',''))

# ds["sequence"] = X_non

# ds.to_csv("../data/dataset_non.csv")

# f = open('all_seq.fasta', 'w')

# for i in range(len(X)):
# 	print(i)
# 	f.write('>seq ' + str(i) + '\n')
# 	f.write(X[i])
# 	f.write('\n')

# f.close()

pv = biovec.models.ProtVec("all_seq.fasta", corpus_fname="output_corpusfile_path.txt", n=3)
print("done")
pv.save('SSG5.biovec')