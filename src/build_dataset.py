# libraries
import pandas as pd
import numpy as np
import requests
import ast 
import json
import pickle
import multiprocessing
import time
import glob, os
import csv
import os
import wget
from Bio import AlignIO

# read data
ds = pd.read_csv('../data/cath_domain_summary.v4_2_0.dsv', sep="\t")

# make dataset
s_val = 5 # SSG5 or SSG9

if s_val == 5:
	ds = ds.drop(["SSG9_NUMBER"], axis=1)
elif s_val == 9:
	ds = ds.drop(["SSG5_NUMBER"], axis=1)
else:
	print("Invalid s_val")

# print(ds.columns)
# making the classes
ds = ds[['DOMAIN_ID', 'FUNFAM_NUMBER', 'SUPERFAMILY_ID' ,'SSG5_NUMBER']]
ds = ds[~ds['SSG5_NUMBER'].isnull()]
ds["CLASS"] = ds["SUPERFAMILY_ID"].astype(str) + '_' + ds["SSG5_NUMBER"].astype(str)

ds = ds.reset_index()
ds = ds.drop(['index'], axis=1)
# print(ds)

# downloading the Stockholm alignments
df = ds[['SUPERFAMILY_ID', 'FUNFAM_NUMBER']]
# print(df)
df = df.drop_duplicates()
# print(df)

sup_fam = list(df['SUPERFAMILY_ID'])
ff = list(df['FUNFAM_NUMBER'])
# print(len(sup_fam), len(ff))

# for i in range(6466, 6468):
# 	print(i, len(sup_fam))
# 	filename = '/home/vamsi/Academics/UCL/src/stockholm/' + str(sup_fam[i]) + '_' + str(ff[i]).replace('.0','') + '.stockholm_aln'
# 	url = 'http://www.cathdb.info/version/v4_2_0/superfamily/' + str(sup_fam[i]) + '/funfam/' + str(ff[i]).replace('.0','') + '/files/stockholm'
# 	print(url)
# 	cmd = 'wget -O ' + filename + ' ' + url
# 	os.system(cmd)

# for i in range(len(sup_fam)):
# 	filename = '/home/vamsi/Academics/UCL/src/stockholm/' + str(sup_fam[i]) + '_' + str(ff[i]).replace('.0','') + '.stockholm_aln'
# 	if os.stat(filename).st_size == 0:
# 		print(i, filename)

# dataset = []

# for i in range(len(sup_fam)):
# 	filename = 'stockholm/' + str(sup_fam[i]) + '_' + str(ff[i]).replace('.0','') + '.stockholm_aln'
# 	f = filename.split('_')
# 	sup_fam_id = f[0].split('/')[1]
# 	ff_id = f[1].split('.')[0]
# 	# print(sup_fam_id, ff_id)
# 	rslt_df = ds[ds.SUPERFAMILY_ID == sup_fam_id]
# 	rslt_df = rslt_df[rslt_df.FUNFAM_NUMBER == int(ff_id)]
# 	# print(rslt_df)
# 	ssg5_id = list(set(list(rslt_df['SSG5_NUMBER'])))[0]
# 	# print(ssg5_id)
# 	class_id = str(sup_fam_id) + '_' + str(int(ssg5_id))

# 	try:
# 		# only one alignment
# 		align = AlignIO.read(filename, "stockholm")
# 		seq_list = []
# 		for record in align:
# 			seq_list.append(str(record.seq).replace('-',''))
# 	except:
# 		# multiple alignments
# 		file = open(filename)
# 		for line in file:
# 			if line[0] != '#' and line != '\n':
# 				split_line = line.split(' ')
# 				new_split = []
# 				for j in split_line:
# 					if j != '':
# 						new_split.append(j)
# 				try:
# 					seq = new_split[1].replace('-','')
# 					seq_list.append(str(seq))
# 				except:
# 					pass

# 	seq_list = list(set(seq_list))
# 	for seq in seq_list:
# 		row = [seq, class_id]
# 		dataset.append(row)

# 	print(i, len(sup_fam), len(seq_list), class_id)

filename = 'dataset.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(dataset, outfile)
# outfile.close()

infile = open(filename,'rb')
dataset = pickle.load(infile)
infile.close()

print(len(dataset))

dataset = pd.DataFrame(dataset)
dataset.columns = ['sequence', 'class'] # class = SUPFAM_ID-SSG5_ID
print(dataset)
dataset.to_csv('dataset.csv')