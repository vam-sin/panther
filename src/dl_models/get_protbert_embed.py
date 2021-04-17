# libraries
# libraries
import numpy as np
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder
import pandas as pd 

embedder = ProtTransBertBFDEmbedder()

def get_pb(sequences):
	embeddings = []
	for seq in sequences:
		embeddings.append(np.mean(np.asarray(embedder.embed(seq)), axis=0))

	return embeddings

if __name__ == '__main__':
	# get sequence list
	seq = ["A", "M", "AM"]
	embed = np.asarray(get_pb(seq))
	print(embed.shape)