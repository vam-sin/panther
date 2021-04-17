# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import math
import pickle
import re
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import regularizers
from keras import backend as K
import keras
from get_protbert_embed import get_pb

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
ds = pd.read_csv('SSG5_BLAST_L100.csv')

X = np.load('SSG5_BLAST_L100_PB.npz')['arr_0']
X = np.expand_dims(X, axis = 1)
y = list(ds["SSG5_Class"])

# y process
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
num_classes = len(np.unique(y))
print(num_classes)
print("Loaded X and y")

X, y = shuffle(X, y, random_state=42)
print("Shuffled")

# indices = np.arange(len(y))

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

            X_batch.append(X_t[val])
            y_enc = np.zeros((num_classes))
            y_enc[y_t[val]] = 1
            y_batch.append(y_enc)
            val += 1

        # X_batch = get_pb(X_batch)
        X_batch = np.asarray(X_batch)
        y_batch = np.asarray(y_batch)

        yield X_batch, y_batch

# batch size
bs = 256

# test and train generators
train_gen = bm_generator(X_train, y_train, bs)
test_gen = bm_generator(X_test, y_test, bs)

# num_classes = 3898

# sensitivity metric
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# keras nn model
input_ = Input(shape = (1,1024,))
x = Conv1D(512, (3), padding="same", activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Conv1D(512, (3), padding="same", activation = "relu")(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x) 
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)

print(model.summary())

# adam optimizer
opt = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy', sensitivity])

# callbacks
mcp_save = keras.callbacks.ModelCheckpoint('cnn_protbert.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# training
num_epochs = 500
with tf.device('/gpu:0'): # use gpu
    history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = test_gen, validation_steps = len(X_test)/bs, workers = 0, shuffle = False, callbacks = callbacks_list)
    model = load_model('cnn_protbert.h5', custom_objects={'sensitivity':sensitivity})

with tf.device('/cpu:0'):
    y_pred = model.predict(X_test)
    print("Classification Report Validation")
    cr = classification_report(y_test, y_pred.argmax(axis=1), output_dict = True)
    df = pd.DataFrame(cr).transpose()
    df.to_csv('CR_CNN_ProtBert.csv')
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    print(matrix)
    print("F1 Score")
    print(f1_score(y_test, y_pred.argmax(axis=1), average = 'weighted'))

'''
cnn_protbert.h5 (On Beaker)
loss: 0.0111 - accuracy: 0.9966 - sensitivity: 0.9964 - 
val_loss: 0.0768 - val_accuracy: 0.9932 - val_sensitivity: 0.9929
'''