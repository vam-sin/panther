# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import biovec
import math
import pickle
import tensorflow as tf
from sklearn import preprocessing
from keras.models import Model
from tensorflow.keras.models import load_model
import keras.backend as K
from keras import optimizers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, GaussianNoise, LeakyReLU, Add
from keras.utils import to_categorical, np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import pickle
from keras import regularizers
from keras import backend as K
import keras

# GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.keras.backend.clear_session()



config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# dataset import 
ds = pd.read_csv('../data/dataset_non.csv')

# 5,515 unique classes -> 5481 unique classes with more than one datapoint
# 3898 unique classes with >= 100 training examples
# removing those classes that have only one datapoint
values = ds["class"].value_counts()
to_remove = list(values[values < 100].index)
ds = ds[ds["class"].isin(to_remove) == False]

ds = ds.reset_index() 
ds.columns = ["one", "two", "three", "sequence", "class"]
ds = ds.drop(columns = ["one", "two", "three"])

# train_id = []
# test_id = []

# values = pd.DataFrame(ds["class"].value_counts())
# values = values.to_dict()
# # print(values["class"])
# split_percentage = 0.8
# k = 1
# for i in values["class"].keys():
# 	print(k, len(values["class"].keys()))
# 	k += 1
# 	if values["class"][i] >= 5:
# 		row_nums = ds[ds['class'] == str(i)].index.to_numpy()
# 		np.random.shuffle(row_nums)
# 		train_vals = int(split_percentage*len(row_nums))
# 		row_train_nums = row_nums[:train_vals]
# 		row_test_nums = row_nums[train_vals:]
# 		# print(i, len(row_nums), len(row_train_nums), len(row_test_nums))
# 	else:
# 		row_nums = ds[ds['class'] == str(i)].index.to_numpy()
# 		np.random.shuffle(row_nums)
# 		train_vals = values["class"][i] - 1
# 		row_train_nums = row_nums[:train_vals]
# 		row_test_nums = row_nums[train_vals:]
# 		# print(i, len(row_nums), len(row_train_nums), len(row_test_nums))

# 	for j in row_train_nums:
# 		train_id.append(j)
# 	for j in row_test_nums:
# 		test_id.append(j)

# filename = 'train_id.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(train_id, outfile)
# outfile.close()

# filename = 'test_id.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(test_id, outfile)
# outfile.close()

# X and y
X = list(ds["sequence"])
y = ds["class"]

# y process
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
# print(np.unique(y))
num_classes = len(np.unique(y))
# y = to_categorical(y, num_classes)
print(num_classes)

# bm = biovec.models.load_protvec('SSG5.biovec')

# X_vec = []

# for i in range(len(X)):
# 	print(i, len(X))
# 	X_vec.append(bm.to_vecs(X[i]))

# filename = 'X_BioVec_100.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(X_vec, outfile)
# outfile.close()

print("Loading BioVec X")
filename = 'X_BioVec_100.pickle'
infile = open(filename,'rb')
X_vec = pickle.load(infile)
infile.close()
print("Loaded X and y")

print("Normalizing X")
scaler = preprocessing.StandardScaler().fit(X_vec)
X_vec = scaler.transform(X_vec)
print("Normalization Complete")

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify = y)
print("Conducted Train-Test Split")

# filename = 'test_id.pickle'
# infile = open(filename,'rb')
# test_id = pickle.load(infile)
# infile.close()

# # split: 3304987 828979 
# # X_train = [X[i] for i in train_id]
# # y_train = [y[i] for i in train_id]

# # X_test = [X[i] for i in test_id]
# # y_test = [y[i] for i in test_id]

# # print(len(X_train), len(X_test), len(y_train), len(y_test))

# # print(len(list(set(y_test))), len(list(set(y_train))))

# # print(len(X_train) + len(X_test))

# # print(max(train_id), max(test_id), min(train_id), min(test_id))

# # print(min(y_test), max(y_test), min(y_train), max(y_train))

# # print(num_classes)

# # for i in train_id:
# # 	if i in test_id:
# # 		print("YES")

# bm = biovec.models.load_protvec('SSG5.biovec')

# # y_vec = np.zeros((num_classes))
# # y_vec[y[train_id[0]]] = 1
# # x_vec = bm.to_vecs(X[train_id[0]])
# # print(X[train_id[0]], x_vec, y[train_id[0]], y_vec)

def bm_generator(X_t, y_t, batch_size):
	val = 0

	while True:
		X_batch = []
		y_batch = []

		for j in range(batch_size):

			if val == len(X_t):
				val = 0

			X_batch.append(X_t[val])
			y_enc = np.zeros((num_classes))
			y_enc[y_t[val]] = 1
			y_batch.append(y_enc)
			val += 1

		X_batch = np.asarray(X_batch)
		y_batch = np.asarray(y_batch)

		yield X_batch, y_batch

# def bm_test_generator(X_t, y_t, batch_size):
# 	val = 0

# 	while True:
# 		X_batch = []
# 		y_batch = []

# 		for j in range(batch_size):

# 			if val == len(test_id):
# 				val = 0

# 			# try:
# 			vec = bm.to_vecs(X_t[test_id[val]])
# 			# except:
# 			# 	vec = np.random.uniform(0, 0.2, size=(3,100))
# 			# print(vec)
# 			X_batch.append(vec)

# 			y_enc = np.zeros((num_classes))
# 			y_enc[y_t[test_id[val]]] = 1
# 			y_batch.append(y_enc)
# 			val += 1

# 		X_batch = np.asarray(X_batch)
# 		y_batch = np.asarray(y_batch)
# 		# X_batch = np.expand_dims(X_batch, axis=0)
# 		# print(X_batch.shape)

# 		yield X_batch, y_batch

bs = 32

train_gen = bm_generator(X_train, y_train, bs)
test_gen = bm_generator(X_test, y_test, bs)

# keras nn model
input_ = Input(shape = (3, 100))
# x = Conv1D(32, (3), padding = 'same', kernel_initializer = 'glorot_uniform', activation = 'tanh')(input_)
x = Flatten()(input_)
x = Dense(1028)(x)
x = LeakyReLU()(x)
x = Dense(1028)(x)
x = LeakyReLU()(x)
x = Dense(1028)(x)
x = LeakyReLU()(x)
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)

print(model.summary())

opt = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

# callbacks
mcp_save = keras.callbacks.callbacks.ModelCheckpoint('ann.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# training
num_epochs = 10
history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs*num_epochs)), verbose=1, shuffle = False, validation_data = test_gen, validation_steps = len(X_test), workers = 0, callbacks = callbacks_list)
