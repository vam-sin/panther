'''
CNN1: loss: 2.2019 - accuracy: 0.4569 - sensitivity: 0.2936 - specificity: 1.0000 - val_loss: 1.9840 - 
val_accuracy: 0.4766 - val_sensitivity: 0.3160 - val_specificity: 1.0000 
filename: cnn_47-66.h5
50 epochs, 256 bs
BioVec
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