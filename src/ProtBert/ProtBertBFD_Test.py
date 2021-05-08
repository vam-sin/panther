# libraries
import numpy as np
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder
import pandas as pd 

embedder = ProtTransBertBFDEmbedder()

ds = pd.read_csv('SSG5_Test.csv')

sequences_Example = list(ds["Sequence"])
num_seq = len(sequences_Example)

i = 0
length = 1000
while i < num_seq:
	print("Doing", i, num_seq)
	start = i 
	end = i + length

	sequences = sequences_Example[start:end]

	embeddings = []
	for seq in sequences:
		embeddings.append(np.mean(np.asarray(embedder.embed(seq)), axis=0))

	s_no = start / length
	filename = 'PB_Test/' + 'PB_' + str(s_no) + '.npz'
	embeddings = np.asarray(embeddings)
	# print(embeddings.shape)
	np.savez_compressed(filename, embeddings)
	i += length