import numpy as np

filename = 'ProtBert_Embeddings/' + 'PB_' + '0.0' + '.npz'
pb_arr = np.load(filename)['arr_0']

for i in range(1, 800):
	print(i, pb_arr.shape)
	try:
		filename = 'ProtBert_Embeddings/' + 'PB_' + str(i) + '.0' + '.npz'
		arr = np.load(filename)['arr_0']
		pb_arr = np.append(pb_arr, arr, axis = 0)
	except:
		pass

np.savez_compressed('SSG5_BLAST_L100_PB.npz', pb_arr)