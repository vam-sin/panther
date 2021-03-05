# libraries
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

model = AutoModel.from_pretrained("Rostlab/prot_bert")
# model = model.half()

fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)

def get_pb(sequences):
	embedding = fe(sequences)

	features = [] 

	for seq_num in range(len(embedding)):
	    seq_len = len(sequences[seq_num].replace(" ", ""))
	    start_Idx = 1
	    end_Idx = seq_len+1
	    seq_emd = embedding[seq_num][start_Idx:end_Idx]
	    features.append(seq_emd)

	res_embeddings = []
	for j in features:
		res_embeddings.append(np.mean(np.asarray(j), axis=0))

	return np.asarray(res_embeddings)

if __name__ == '__main__':
	# get sequence list
	seq = ["A", "M", "AM"]
	embed = get_pb(seq)
	print(embed)