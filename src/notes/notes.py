###### ANN Notes ######
# ANN1: Val Acc: 0.11162, filename: ann_11-162.h5 
# 50 epochs, 256 bs
input_ = Input(shape = (300,))
x = Dense(1024, activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)

'''
ANN2: loss: 2.8093 - accuracy: 0.3794 - sensitivity: 0.2100 - specificity: 1.0000 - 
val_loss: 1.7272 - val_accuracy: 0.5274 - val_sensitivity: 0.3213 - val_specificity: 1.0000, 
filename: ann_52-738.h5 
Notes: Extremely regularized, too much dropout
50 epochs, 256 bs
'''
input_ = Input(shape = (300,))
x = Dense(1024, activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)

'''
ANN3: loss: 1.3972 - accuracy: 0.6119 - sensitivity: 0.5052 - specificity: 1.0000 - 
val_loss: 1.0776 - val_accuracy: 0.6589 - val_sensitivity: 0.5692 - val_specificity: 1.0000
filename: ann_65-89.h5
Notes: More regularization than needed
50 epochs, 256 bs
'''
input_ = Input(shape = (300,))
x = Dense(1024, activation = "relu")(input_)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x) 
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)

'''
ANN4: loss: 0.9166 - accuracy: 0.6976 - sensitivity: 0.6195 - specificity: 1.0000 - 
val_loss: 0.9535 - val_accuracy: 0.6588 - val_sensitivity: 0.6051 - val_specificity: 1.0000
filename: ann_65-876.h5
Notes: Little Overfitting, but that's chill.
50 epochs, 256 bs
'''
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

'''
CNN1: loss: 2.2019 - accuracy: 0.4569 - sensitivity: 0.2936 - specificity: 1.0000 - val_loss: 1.9840 - 
val_accuracy: 0.4766 - val_sensitivity: 0.3160 - val_specificity: 1.0000 
filename: cnn_47-66.h5
50 epochs, 256 bs
'''
# keras nn model
input_ = Input(shape = (3,100,))
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

'''
ABLE1: loss: 1.3829 - accuracy: 0.6167 - sensitivity: 0.4856 - specificity: 1.0000 - val_loss: 1.7264 - 
val_accuracy: 0.5373 - val_sensitivity: 0.4305 - val_specificity: 1.0000
filename: able_53-73.h5
150 epochs
'''
# LSTM model; cannot use batchnormalization on this
input_ = Input(shape = (3,100,))
x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(input_)
x = Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True))(x)
x = SeqSelfAttention(attention_activation='tanh')(x)
x = Flatten()(x)
out = Dense(num_classes, activation = 'softmax')(x)
model = Model(input_, out)
