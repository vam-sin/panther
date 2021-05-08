import numpy as np

filename = 'PB_Train/' + 'PB_' + '0.0' + '.npz'
pb_arr = np.load(filename)['arr_0']

for i in range(1, 1000):
	print(i, pb_arr.shape)
	try:
		filename = 'PB_Train/' + 'PB_' + str(i) + '.0' + '.npz'
		arr = np.load(filename)['arr_0']
		pb_arr = np.append(pb_arr, arr, axis = 0)
	except:
		pass

np.savez_compressed('SSG5_Train_ProtBert.npz', pb_arr)