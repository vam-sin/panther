# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import biovec
import math
import pickle
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input
from keras.utils import to_categorical, np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import regularizers
from keras import backend as K
import keras

# GPU config for Vamsi's Laptop
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
# ds = pd.read_csv('../data/dataset_non.csv')

# # 5,515 unique classes -> 5481 unique classes with more than one datapoint
# # 3898 unique classes with >= 100 training examples
# # removing those classes that have only one datapoint
# values = ds["class"].value_counts()
# to_remove = list(values[values < 100].index)
# ds = ds[ds["class"].isin(to_remove) == False]

# ds = ds.reset_index() 
# ds.columns = ["one", "two", "three", "sequence", "class"]
# ds = ds.drop(columns = ["one", "two", "three"])

# X = list(ds["sequence"])
# y = list(ds["class"])

# # # biovec model
# vectorizer = biovec.models.load_protvec('ds_preprocess/SSG5.biovec')

# # min-max normalization
# mini = -112.18917
# maxi = 115.491425
# sub = maxi - mini

# X_vec = []
# y_vec = []

# for i in range(len(X)):
#     print(i, len(X))
#     vec = vectorizer.to_vecs(X[i])
#     vec = np.asarray(vec)
#     vec = vec - mini
#     vec /= sub
#     X_vec.append(vec)
#     y_vec.append(y[i])

filename = 'X_BV_100_SSG5_Full.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(X_vec, outfile)
# outfile.close()
infile = open(filename,'rb')
X = pickle.load(infile)
infile.close()

filename = 'y_BV_100_SSG5_Full.pickle'
# outfile = open(filename, 'wb')
# pickle.dump(y_vec, outfile)
# outfile.close()
infile = open(filename,'rb')
y = pickle.load(infile)
infile.close()

# y process
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))
print(num_classes)
print("Loaded X and y")

X, y = shuffle(X, y, random_state=42)
print("Shuffled")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Conducted Train-Test Split")

num_classes_train = len(np.unique(y_train))
num_classes_test = len(np.unique(y_test))
print(num_classes_train, num_classes_test)

assert num_classes_test == num_classes_train, "Split not conducted correctly"

# generator
def bm_generator(X_t, y_t, batch_size):
	val = 0

	while True:
		X_batch = []
		y_batch = []

		for j in range(batch_size):

			if val == len(X_t):
				val = 0

			X_batch.append(np.reshape(X_t[val], (300)))
			y_enc = np.zeros((num_classes))
			y_enc[y_t[val]] = 1
			y_batch.append(y_enc)
			val += 1

		X_batch = np.asarray(X_batch)
		y_batch = np.asarray(y_batch)

		yield X_batch, y_batch

# batch size
bs = 256

# test and train generators
train_gen = bm_generator(X_train, y_train, bs)
test_gen = bm_generator(X_test, y_test, bs)

# num_classes = 3898

# sampled softmax loss
class SampledSoftmaxLoss(object):
  """ The loss function implements the Dense layer matmul and activation
  when in training mode.
  """
  def __init__(self, model):
    self.model = model
    output_layer = model.layers[-1]
    self.input = output_layer.input
    self.weights = output_layer.weights

  def loss(self, y_true, y_pred, **kwargs):
    labels = tf.argmax(y_true, axis=1)
    labels = tf.expand_dims(labels, -1)
    loss = tf.nn.nce_loss(
        weights=self.weights[0],
        biases=self.weights[1],
        labels=labels,
        inputs=self.input,
        num_sampled = 99,
        num_classes = 100,
        num_true = 1,
    )
    return loss

# sensitivity metric
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# specificity metric
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# keras nn model
input_ = Input(shape = (300,))
x = Dense(1024, activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
# x = Dropout(0.3)(x) 
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)

# loss function
loss_calculator = SampledSoftmaxLoss(model)

print(model.summary())

# adam optimizer
opt = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy', sensitivity, specificity])

# callbacks
mcp_save = keras.callbacks.callbacks.ModelCheckpoint('ann.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# training
num_epochs = 100
with tf.device('/gpu:0'): # use gpu
    history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = test_gen, validation_steps = len(X_test)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)

# Best: 0.11162