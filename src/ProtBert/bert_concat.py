import numpy as np 
import h5py

# find out max and min for normalization
# a1 = np.load('ProtBert_Embeddings/PB_0.0.npz')
# a1 = a1['arr_0']

# for i in range(1, 40796):
# 	print(i, 40796)
# 	filename = 'ProtBert_Embeddings/PB_' + str(i) + '.0.npz'
# 	arr = np.load(filename)
# 	arr = arr['arr_0']
# 	a1 = np.concatenate((a1, arr), axis = 0)

# 	print(a1.shape)

# np.savez_compressed('ProtBert_100.npz', a1)

# hf = h5py.File('PB_100.h5', 'w')

# for i in range(40796):
# 	print(i, 40796)
# 	filename = 'ProtBert_Embeddings/PB_' + str(i) + '.0.npz'
# 	arr = np.load(filename)
# 	arr = arr['arr_0']
# 	hf.create_dataset(str(i) + '_PB', data=arr)

# hf.close()

lis = []

def add_average(vec):
	inter = vec[0]
	inter += vec[1]
	inter += vec[2]

	inter /= 3

	return inter

for i in range(40796):
	print(i, 40796)
	filename = 'ProtBert_Embeddings/PB_' + str(i) + '.0.npz'
	arr = np.load(filename)
	arr = arr['arr_0']
	for j in arr:
		jprime = add_average(j)
		# print(jprime.shape)
		lis.append(jprime)

# print(min_num, max_num)

np.savez_compressed('ProtBert_100.npz', lis)