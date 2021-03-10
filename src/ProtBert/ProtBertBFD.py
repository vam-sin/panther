# libraries
import numpy as np
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder
import pandas as pd 

embedder = ProtTransBertBFDEmbedder()

ds = pd.read_csv('dataset_non.csv')

# 5,515 unique classes -> 5481 unique classes with more than one datapoint
# 3898 unique classes with >= 100 training examples
# removing those classes that have only one datapoint
values = ds["class"].value_counts()
to_remove = list(values[values < 100].index)
ds = ds[ds["class"].isin(to_remove) == False]

ds = ds.reset_index() 
ds.columns = ["one", "two", "three", "sequence", "class"]
ds = ds.drop(columns = ["one", "two", "three"])

sequences_Example = list(ds["sequence"])
num_seq = len(sequences_Example)
y = list(ds["class"])

i = 0
length = 100
while i < num_seq:
	print(i, num_seq)
	start = i 
	end = i + length

	sequences = sequences_Example[start:end]

	embeddings = []
	for seq in sequences:
		embeddings.append(np.mean(np.asarray(embedder.embed(seq)), axis=0))

	s_no = start / length
	filename = 'ProtBert_Embeddings/' + 'PB_' + str(s_no) + '.npz'
	embeddings = np.asarray(embeddings)
	# print(embeddings.shape)
	np.savez_compressed(filename, embeddings)
	i += length