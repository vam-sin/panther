# libraries
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import pandas as pd 
import os
import pickle
import requests
from tqdm.auto import tqdm

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

model = AutoModel.from_pretrained("Rostlab/prot_bert")
# model = model.half()

fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)

# # dataset import 
# ds = pd.read_csv('dataset_non.csv')

# # 5,515 unique classes -> 5481 unique classes with more than one datapoint
# # 3898 unique classes with >= 100 training examples
# # removing those classes that have only one datapoint
# values = ds["class"].value_counts()
# to_remove = list(values[values < 100].index)
# ds = ds[ds["class"].isin(to_remove) == False]

# ds = ds.reset_index() 
# ds.columns = ["one", "two", "three", "sequence", "class"]
# ds = ds.drop(columns = ["one", "two", "three"])

# sequences_Example = list(ds["sequence"])
# num_seq = len(sequences_Example)
# y = list(ds["class"])

# sequences_Example = [s.replace("", " ")[1: -1] for s in sequences_Example]

# print("Added Gaps")

# sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

# print("Substituted Rare Amino Acids")

filename = 'X_subst.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(y_vec, outfile)
# outfile.close()
infile = open(filename,'rb')
sequences_Example = pickle.load(infile)
infile.close()

num_seq = len(sequences_Example)

length = 100

i = 0
while i < num_seq:
	print(i, num_seq)
	start = i 
	end = i + length

	sequences = sequences_Example[start:end]

	embedding = fe(sequences)

	features = [] 
	for seq_num in range(len(embedding)):
	    seq_len = len(sequences[seq_num].replace(" ", ""))
	    start_Idx = 1
	    end_Idx = seq_len+1
	    seq_emd = embedding[seq_num][start_Idx:end_Idx]
	    features.append(seq_emd)

	embeddings = []
	for j in features:
		embeddings.append(np.mean(np.asarray(j), axis=0))

	s_no = start / length
	filename = 'ProtBert_Embeddings/' + 'PB_' + str(s_no) + '.npz'
	embeddings = np.asarray(embeddings)
	# print(embeddings.shape)
	np.savez_compressed(filename, embeddings)
	i += length