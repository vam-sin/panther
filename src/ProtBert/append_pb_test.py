import numpy as np

filename = 'PB_Test/' + 'PB_' + '0.0' + '.npz'
pb_arr = np.load(filename)['arr_0']

for i in range(1, 800):
	print(i, pb_arr.shape)
	try:
		filename = 'PB_Test/' + 'PB_' + str(i) + '.0' + '.npz'
		arr = np.load(filename)['arr_0']
		pb_arr = np.append(pb_arr, arr, axis = 0)
	except:
		pass

np.savez_compressed('SSG5_Test_ProtBert.npz', pb_arr)