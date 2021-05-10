#libraries
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import pickle
from sklearn.preprocessing import OneHotEncoder 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import math
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, MaxPooling1D
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow import keras

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
ds_train = pd.read_csv('SSG5_Train.csv')

X = list(ds_train["Sequence"])
y = list(ds_train["SSG5_Class"])

ds_test = pd.read_csv('SSG5_Test.csv')

X_test = np.load('SSG5_Test_OneHot.npz')['arr_0']
y_test = list(ds_test["SSG5_Class"])

# maximum sequence length is 694 residues in the ds

max_length = 1203

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def create_dict(codes):
    char_dict = {}
    for index, val in enumerate(codes):
        char_dict[val] = index+1

    return char_dict

char_dict = create_dict(codes)

print(char_dict)
print("Dict Length:", len(char_dict))

def integer_encoding(data):
    """
    - Encodes code sequence to integer values.
    - 20 common amino acids are taken into consideration
    and rest 4 are categorized as 0.
    """
    encode_list = []
    for row in data:
        row_encode = [] 
        for code in row: 
            row_encode.append(char_dict.get(code, 0))
        encode_list.append(np.array(row_encode))

    return encode_list
  
# y process
# y process
y_tot = []

for i in range(len(y)):
    y_tot.append(y[i])

for i in range(len(y_test)):
    y_tot.append(y_test[i])

le = preprocessing.LabelEncoder()
le.fit(y_tot)
y = np.asarray(le.transform(y))
y_test = np.asarray(le.transform(y_test))
num_classes = len(np.unique(y))
print(num_classes)
print("Loaded X and y")

X, y = shuffle(X, y, random_state=42)
print("Shuffled")

X = integer_encoding(X)
print("Integer Encoded")
X = pad_sequences(X, maxlen=max_length, padding='post', truncating='post')
print("Padded")

def to_cat(seq_list):
    out = []
    for seq in seq_list:
        arr = np.zeros((max_length, 21))
        for i in range(len(seq)):
            arr[i][seq[i]] = 1
        out.append(arr)

    return out

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

        # X_batch = integer_encoding(X_batch)
        # X_batch = pad_sequences(X_batch, maxlen=max_length, padding='post', truncating='post')
        X_batch = np.asarray(to_cat(X_batch))
        X_batchT = []
        for arr in X_batch:
            X_batchT.append(np.reshape(arr.T, (max_length*21)))
        X_batch = np.asarray(X_batchT)
        # print(X_batch.shape)
        y_batch = np.asarray(y_batch)

        yield X_batch, y_batch

# batch size
bs = 256

# num_classes = 1707

# sensitivity metric
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Keras NN Model
def create_model():
    input_ = Input(shape = (max_length*21,))
    x = Dense(1024, activation = "relu")(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x) 
    out = Dense(num_classes, activation = 'softmax')(x)
    classifier = Model(input_, out)

    return classifier

# training
kf = KFold(n_splits = 5, random_state = 42, shuffle = True)

# training
num_epochs = 200

fold = 1

val_f1score = []
val_acc = []
test_f1score = []
test_acc = []

with tf.device('/gpu:0'):
    for train_index, val_index in kf.split(X):
        print("#############################")
        print("Training with Fold " + str(fold))
        print("#############################")

        # model
        model = create_model()

        # adam optimizer
        opt = keras.optimizers.Adam(learning_rate = 1e-5)
        model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy', sensitivity])

        # callbacks
        mcp_save = keras.callbacks.ModelCheckpoint('saved_models/ann_onehot_' + str(fold) + '.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        callbacks_list = [reduce_lr, mcp_save]

        # test and train generators
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        train_gen = bm_generator(X_train, y_train, bs)
        val_gen = bm_generator(X_val, y_val, bs)
        history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)
        model = load_model('saved_models/ann_onehot_' + str(fold) + '.h5', custom_objects={'sensitivity':sensitivity})

        # print("Validation")
        # X_val = np.asarray(to_cat(X_val))
        # X_valT = []
        # for arr in X_val:
        #     X_valT.append(arr.T)
        # X_val = np.asarray(X_valT)
        # y_pred_val = model.predict(X_val)
        # f1_score_val = f1_score(y_val, y_pred_val.argmax(axis=1), average = 'weighted')
        # acc_score_val = accuracy_score(y_val, y_pred_val.argmax(axis=1))
        # val_f1score.append(f1_score_val)
        # val_acc.append(acc_score_val)
        # print("F1 Score: ", val_f1score)
        # print("Acc Score", val_acc)

        print("Testing")
        X_testT = []
        for arr in X_test:
            X_testT.append(np.reshape(arr.T, (max_length*21)))
        X_test = np.asarray(X_testT)
        y_pred_test = model.predict(X_test)
        f1_score_test = f1_score(y_test, y_pred_test.argmax(axis=1), average = 'weighted')
        acc_score_test = accuracy_score(y_test, y_pred_test.argmax(axis=1))
        test_f1score.append(f1_score_test)
        test_acc.append(acc_score_test)
        print("F1 Score: ", test_f1score)
        print("Acc Score", test_acc)

        fold += 1

print("Validation F1 Score: " + str(np.mean(val_f1score)) + ' +- ' + str(np.std(val_f1score)))
print("Validation Acc Score: " + str(np.mean(val_acc)) + ' +- ' + str(np.std(val_acc)))
print("Test F1 Score: " + str(np.mean(test_f1score)) + ' +- ' + str(np.std(test_f1score)))
print("Test Acc Score: " + str(np.mean(test_acc)) + ' +- ' + str(np.std(test_acc)))

'''
saved_models/ann_onehot.h5
Testing
F1 Score:  [0.5825008197066206, 0.5726313125365259, 0.5726646232340983]
0.5759322518257483 +- 0.004644698799338449
Acc Score [0.5911369332421964, 0.5804283435862383, 0.5804283435862383]
0.5839978734715576 +- 0.0050480775751147325
'''